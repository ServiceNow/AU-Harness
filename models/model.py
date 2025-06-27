import asyncio
import tempfile
import re
import soundfile as sf
import os
from abc import ABC

from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_result,
    stop_after_attempt,
    wait_random,
    wait_random_exponential,
)

import logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    fh = logging.FileHandler("audiobench.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)
logging.basicConfig(level=logging.INFO)
logger.propagate = True
from models.model_response import ErrorTracker, ModelResponse
from models.request_resp_handler import RequestRespHandler
from utils import constants
from utils.input_builder import InputBuilder

class Model(ABC):
    """TODO: Need SME to add."""

    def __init__(self, model_info: dict):
        """Initialize model configuration here.

        Args:
            model_info: model configuration dictionary
        """
        
        self.model_info = model_info
        self._name = model_info.get("name")
        self.model = model_info["model"]
        self.api_key = model_info.get("auth_token", "")
        self.model_url = model_info.get("url", "")
        self.inference_type = model_info["inference_type"]
        self.batch_size = model_info.get("batch_size", 1)
        # sleep before every call - in ms
        self.delay = model_info.get("delay", 100)
        self.timeout = model_info.get("timeout", 30)
        # max_wait for 8 attempts = 2^(8-1) = 128 secs
        self.retry_attempts = model_info.get("retry_attempts", 8)
        # some flow does not work with async client like internal private network
        self.postprocessor_path = model_info.get("postprocessor", [])
        #model_name = model_info.get("model", model_info.get("alias", self.name()))
        self.req_resp_hndlr = RequestRespHandler(
            self.inference_type,
            self.model_info,
            timeout=self.timeout,
        )
        self.input_builder = InputBuilder(self.name())
        # prevent data races when updating self.errors asynchronously
        self.errors_lock = asyncio.Lock()
        self.errors = ErrorTracker()

    def name(self):
        return self._name

    def _is_retryable_error(self, result: ModelResponse):
        """Check if the error is a rate limit error by checking response code."""
        # currently retrying for too many requests error(429)
        # and APIConnectionError(599) returned by OpenAI intermittently
        # 450 is a custom status code used for adjusting max tokens
        # 500 can be "The server had an error while processing your request."
        # 503 can be "The service is temporarily unable to process your request"
        # 504 is a Gateway Timeout - HTTP
        return result.response_code and result.response_code in (408, 429, 450, 500, 503, 599, 504)

    def _set_max_backoff(self, attempt):
        """Set exponential backoff for 429 and 1 second for 599 error codes."""
        if (
            attempt.retry_state.outcome.result().response_code == 429
            or attempt.retry_state.outcome.result().response_code == 504
        ):
            attempt.retry_state.retry_object.wait = wait_random_exponential(multiplier=1, max=100)
        elif attempt.retry_state.outcome.result().response_code == 450:
            # 450 error for adjusting max tokens, don't need to wait
            attempt.retry_state.retry_object.wait = 0
        else:
            attempt.retry_state.retry_object.wait = wait_random()

    def _log_before_retry(self, retry_state):
        """Log retry attempt."""
        resp_code = retry_state.outcome.result().response_code

        # Log the URL switch
        if retry_state.attempt_number == 1 and resp_code == 429:
            logger.warning(
                f"[{self.name()}] Retrying the request with next available URL as it returned {resp_code} code in attempt {retry_state.attempt_number}"
            )
        else:
            logger.warning(
                f"[{self.name()}] Retrying the request in {retry_state.next_action.sleep} seconds as it returned {resp_code} code in attempt {retry_state.attempt_number}"
            )

    async def _mark_errors(self, result: ModelResponse):
        """Update error tracker."""
        if result.response_code != 200:
            async with self.errors_lock:
                self.errors.increment(result.response_code)

    def _parse_token_error_message(self, error_message: str) -> tuple[int | None, int | None]:
        """Parse the error message to extract the maximum context length and the number of tokens in the messages.

        Args:
            error_message: The error message string.

        Returns:
            A tuple containing the maximum context length and the number of tokens in the messages.
        """
        pattern = r"maximum context length is (\d+).*?\((\d+) in the messages"
        match = re.search(pattern, error_message)
        if match:
            try:
                max_context_length = int(match.group(1))
                tokens_in_messages = int(match.group(2))
                return max_context_length, tokens_in_messages
            except ValueError:
                pass
        return None, None

    async def _generate_text_with_retry(
        self, message: dict | str, model_params: dict, run_params: dict
    ) -> ModelResponse:
        max_tokens_adjusted = False
        try:
            async for attempt in AsyncRetrying(
                retry=retry_if_result(self._is_retryable_error),
                # reset wait times based on error condition
                wait=wait_random_exponential(multiplier=1, max=300),
                stop=stop_after_attempt(self.retry_attempts),
                before_sleep=self._log_before_retry,
            ):
                with attempt:
                    # initial delay for each call (in ms)
                    result: ModelResponse = await self._generate_text(message, model_params, run_params)
                    logger.info(f"{self.name()} model response: {str(result.raw_response)}")
                    await self._mark_errors(result)

                    if (
                        result.response_code == 500
                        and hasattr(self, "inference_type")
                        and self.inference_type
                        in [constants.INFERENCE_SERVER_VLLM_COMPLETIONS, constants.INFERENCE_SERVER_VLLM]
                        and "maximum context length" in result.raw_response
                        and not max_tokens_adjusted
                    ):
                        max_context_length, tokens_in_messages = self._parse_token_error_message(result.raw_response)
                        if max_context_length and tokens_in_messages:
                            new_max_tokens = (
                                max_context_length - tokens_in_messages - constants.VLLM_MAX_TOKEN_RETRY_BUFFER
                            )
                            if new_max_tokens > 0:
                                # set response code to custom error for this case, so it is retryable with wait 0
                                result.response_code = 450
                                if self.inference_type == constants.INFERENCE_SERVER_VLLM_COMPLETIONS:
                                    model_params["max_tokens"] = new_max_tokens
                                else:
                                    model_params["max_completion_tokens"] = new_max_tokens
                                # use flag to only retry once when max tokens is adjusted
                                max_tokens_adjusted = True
                                logger.warning(f"Adjusted max_tokens to {new_max_tokens} due to context length error.")

                if not attempt.retry_state.outcome.failed:
                    attempt.retry_state.set_result(result)
                self._set_max_backoff(attempt)
        except RetryError:
            logger.error(
                f"[{self.name()}] Request failed after {self.retry_attempts} attempts for input: {message}..."
            )
        return result

    def apply_formatter(self, messages: list[dict], run_params: dict) -> list[dict] | str:
        """Applies formatting according to representative input builder instance."""
        return self.input_builder.build_conversation(messages, run_params)

    async def _generate_text(self, message: dict | str, model_params: dict, run_params: dict) -> ModelResponse:
        """Generic implementation in this class, override if needed.

        It implements model query by building message header and body with the help of Request Response Handler.

        Args:
            messages: inputs for inference
            model_params: model parameter json
            run_params: params for the run

        Returns:
            Response and http return code
        """
        # Build temp wav if needed then delegate to RequestRespHandler so that all requests flow through common path
        if isinstance(message, list):
            raise ValueError("_generate_text expects a single dict or str, not a list.")
        if not isinstance(message, dict):
            raise ValueError("_generate_text expects a dict input for audio transcription.")
        #print(message.keys())
        audio_array = message["array"]
        sampling_rate = message["sampling_rate"]
        audio_file_path = message.get("path")

        fp = None
        if not self.is_path_supported(audio_file_path):
            fp = tempfile.NamedTemporaryFile(suffix=".wav")
            sf.write(fp, audio_array, sampling_rate)
            audio_file_path = fp.name

        # Build message body for request handler in correct format for HTTP file upload
        files = None
        if audio_file_path:
            f = open(audio_file_path, "rb")
            files = {"file": (os.path.basename(audio_file_path), f, "audio/wav")}
        else:
            raise ValueError("audio_file_path must be provided for OpenAI transcription")

        try:
            model_response: ModelResponse = await self.req_resp_hndlr.request_server(
                url=self.model_url,
                auth=self.api_key,
                msg_body=files,
                formatted_messages=message,
            )
        finally:
            if f:
                f.close()
        return model_response
    @staticmethod
    def is_path_supported(audio_file_path) -> bool:
        """Check if the audio file path can directly be used with the OpenAI API."""
        return (
            audio_file_path
            and any(audio_file_path.endswith(sound_format) for sound_format in OPENAI_SUPPORTED_SOUND_FORMAT)
            and audio_file_path.startswith("/tmp")
        )