import asyncio
import importlib
import importlib.util
import re
import time
from abc import ABC, abstractmethod
from http import HTTPStatus

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

        # sleep before every call - in ms
        self.delay = model_info.get("delay", 100)
        # max_wait for 8 attempts = 2^(8-1) = 128 secs
        self.retry_attempts = model_info.get("retry_attempts", 8)

        # some flow does not work with async client like internal private network
        self.postprocessor_path = model_info.get("postprocessor", [])
        #model_name = model_info.get("model", model_info.get("alias", self.name()))
        self.req_resp_hndlr = RequestRespHandler(
            self.inference_type,
            self.model_info,
        )
        self.input_builder = InputBuilder(self.name())

        #self.weighted_params = self._create_weighted_elements(model_info, model_name)

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
            message: input for inference
            model_params: model parameter json
            run_params: params for the run

        Returns:
            Response and http return code
        """
        if isinstance(message, list):
            raise ValueError("_generate_text expects a single sample (dict or str), not a list.")
        if isinstance(message, dict):
            formatted_message = self.apply_formatter([message], run_params)
        else:
            logger.warning(
                f"Input message is already a string. The formatter will not be applied."
                f"Make sure the message is already formatted according to {self.name()}. Received {message}"
            )
            formatted_message = message
        if isinstance(formatted_message, str):
            formatted_message = self._preprocess_text(formatted_message, run_params)
        msg_body = self.req_resp_hndlr.get_input_msg(formatted_message, model_params, run_params)
        logger.debug(f"{self.name()} input message body: {msg_body}")

        current_url_index = self.weighted_params.get_random_index()
        endpoint_in_use = self.weighted_params.listing[current_url_index]
        model_response: ModelResponse = await self.req_resp_hndlr.request_server(
            url=endpoint_in_use["url"],
            auth=endpoint_in_use["auth"],

            msg_body=msg_body,
            formatted_messages=formatted_messages,
        )
        model_response.model_parameters = model_params
        if model_response.response_code == 200:
            #logger.debug(f"{self.name()} model response: {str(model_response.raw_response)}")
            text_resp = self.req_resp_hndlr.get_response_text(model_response.llm_response)
            if text_resp.strip() != "":
                model_response.response_code = 200

            text_resp = self.postprocess(self.postprocessor_path, text_resp, run_params.get("response_format", ""))
            model_response.llm_response = text_resp if len(text_resp) > 0 else " "
            return model_response
        else:
            logger.error(f"Error: error in the request. Code: {model_response.response_code}.")
            model_response.llm_response = " "
            return model_response

    def postprocess(self, postprocessor_path: list[str], text: str, response_format: str):
        """Postprocess the text using the postprocessor."""
        for path in postprocessor_path:
            module, class_name = path.rsplit(".", 1)
            postprocessor = getattr(importlib.import_module(module), class_name)
            text = postprocessor().postprocess(text, response_format)
        return text
