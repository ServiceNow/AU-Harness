"""Model abstraction for Large Audio Language Model evaluation framework.

This module provides the base Model class that orchestrates inference requests
with retry logic and error handling for different model backends.
"""

import asyncio
import copy
import logging
from abc import ABC

from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_result,
    stop_after_attempt,
    wait_random,
    wait_random_exponential,
)

from models.model_response import ErrorTracker, ModelResponse
from models.request_resp_handler import RequestRespHandler
from utils import constants
from utils.constants import task_temp_map
from utils.multimodal import encode_audio_array_base64, audio_array_to_wav_file

logger = logging.getLogger(__name__)
logger.propagate = True


class Model(ABC):
    """TODO: Need SME to add."""

    def __init__(self, model_info: dict, temperature: float = 0.7):
        """Initialize model configuration here.

        Args:
            model_info: model configuration dictionary
        """
        self.model_info = model_info
        self._name = model_info.get("name")
        self.model = model_info.get("model")
        self.api_key = model_info.get("auth_token", "")
        self.model_url = model_info.get("url")
        self.api_version = model_info.get("api_version")
        self.inference_type = model_info.get("inference_type")
        self.batch_size = model_info.get("batch_size", 1)
        # sleep before every call - in ms
        self.delay = model_info.get("delay", 100)
        self.timeout = model_info.get("timeout", 30)
        # max_wait for 8 attempts = 2^(8-1) = 128 secs
        self.retry_attempts = model_info.get("retry_attempts", 8)
        # chunk_size in seconds (default 30)
        self.chunk_size = model_info.get("chunk_size", 30)
        # temperature for LLM requests (default 0.7)
        self.temperature = temperature
        # some flow does not work with async client like internal private network
        self.postprocessor_path = model_info.get("postprocessor", [])
        # model_name = model_info.get("model", model_info.get("alias", self.name()))
        self.req_resp_hndlr = RequestRespHandler(
            self.inference_type,
            self.model_info,
            timeout=self.timeout,
            temperature=self.temperature
        )
        # prevent data races when updating self.errors asynchronously
        self.errors_lock = asyncio.Lock()
        self.errors = ErrorTracker()

    def name(self):
        """Return the model name.
        
        Returns:
            str: The model name.
        """
        return self._name

    def set_temp(self, task_type: str) -> None:
        """Set temperature based on task type using task_temp_map.
        
        Args:
            task_type: The type of task being performed.
        """
        if task_type in task_temp_map:
            self.temperature = task_temp_map[task_type]
            # Also update the request handler's temperature
            self.req_resp_hndlr.temperature = self.temperature

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
        if attempt.retry_state.outcome.result().response_code in (429, 504):
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
                "[%s] Retrying the request with next available URL as it returned %s code in attempt %s",
                self.name(), resp_code, retry_state.attempt_number
            )
        else:
            logger.warning(
                "[%s] Retrying the request in %s seconds as it returned %s code in attempt %s",
                self.name(), retry_state.next_action.sleep, resp_code, retry_state.attempt_number
            )

    async def _mark_errors(self, result: ModelResponse, error_tracker: ErrorTracker):
        """Update error tracker."""
        if result.response_code != 200:
            # No lock needed since this is a per-call error tracker
            error_tracker.increment(result.response_code)
            # Make sure the error tracker is attached to the ModelResponse
            result.error_tracker = error_tracker

    async def generate_text_with_retry(
            self, message: dict | str, run_params: dict
    ) -> ModelResponse:
        """Generate text with retry logic and error handling.
        
        Args:
            message: Input message for model inference
            run_params: Runtime parameters for the inference request
            
        Returns:
            ModelResponse: Response object containing generated text and metadata
        """
        # Create a new error tracker instance for this specific call
        call_errors = ErrorTracker()
        result = None
        try:
            async for attempt in AsyncRetrying(
                    retry=retry_if_result(self._is_retryable_error),
                    wait=wait_random_exponential(multiplier=1, max=300),
                    stop=stop_after_attempt(self.retry_attempts),
                    before_sleep=self._log_before_retry,
            ):
                with attempt:
                    try:
                        # All data prep is now in _generate_text
                        # Set attempt number for downstream logging
                        self.req_resp_hndlr.current_attempt = attempt.retry_state.attempt_number
                        # Pass the error tracker to _generate_text
                        result: ModelResponse = await self._generate_text(message, run_params, call_errors)
                        # Ensure the result has our error tracker
                        if not result.error_tracker:
                            result.error_tracker = call_errors
                        await self._mark_errors(result, call_errors)
                    except Exception as e:
                        metric_name = run_params.get("metric")
                        logger.error("Exception during text generation for metric %s: %s", metric_name, e)
                        result = ModelResponse(
                            input_prompt=str(message),
                            llm_response="",
                            raw_response=str(e),
                            response_code=500,
                            performance=None,
                            wait_time=0,
                            error_tracker=call_errors,
                        )
                        await self._mark_errors(result, call_errors)
                attempt.retry_state.set_result(result)
                # Set backoff for next retry based on current result
                self._set_max_backoff(attempt)
                if not self._is_retryable_error(result):
                    break
        except KeyError as e:
            logger.error("Missing key while building model_inputs on the fly: %s", e)
            result = ModelResponse(
                input_prompt=message,
                llm_response="",
                raw_response=f"Missing key: {e}",
                response_code=500,
                performance=None,
                wait_time=0,
                error_tracker=call_errors,
            )
            await self._mark_errors(result, call_errors)
        except RetryError:
            logger.error(
                "[%s] Request failed after %s attempts",
                self.name(), self.retry_attempts
            )
            result = ModelResponse(
                input_prompt=message,
                llm_response="",
                raw_response="RetryError: Request failed after max attempts",
                response_code=500,
                performance=None,
                wait_time=0,
                error_tracker=call_errors,
            )
            await self._mark_errors(result, call_errors)
        except Exception as e:
            logger.error("Unexpected error in _generate_text_with_retry: %s", e)
            result = ModelResponse(
                input_prompt=message,
                llm_response="",
                raw_response=str(e),
                response_code=500,
                performance=None,
                wait_time=0,
                error_tracker=call_errors,
            )
            await self._mark_errors(result, call_errors)
        return result






    async def _generate_text(self, message: dict, run_params: dict, error_tracker: ErrorTracker = None) -> ModelResponse:
        """
        Implements model query by building message header and body with the help of Request Response Handler.
        Args:
            message: inputs for inference
            run_params: params for the run
        Returns:
            Response and http return code
        """

        # getting attributes
        audio_array = message.get("array", None)
        sampling_rate = message.get("sampling_rate", 0)
        chunk_seconds: int = int(run_params.get("chunk_size", 30))  # default to 30s
        metric_name: str | None = run_params.get("metric")
        max_samples: int = int(chunk_seconds * sampling_rate) if sampling_rate else 0
        total_samples: int = len(audio_array) if audio_array is not None else 0
        instruction = message.get("instruction")

        tools = copy.deepcopy(message.get('tools', None))
        # If metric is judge types, only use first chunk (30s) regardless of length
        judge_metrics = {"llm_judge_binary", "llm_judge_detailed"}
        if metric_name in judge_metrics:
            total_samples = max_samples  # force single chunk

        # --- CHUNKING LOGIC FIRST ---
        if total_samples > max_samples:
            concatenated_text: str = ""
            responses: list[ModelResponse] = []
            num_chunks = (total_samples + max_samples - 1) // max_samples

            # chat completion – process each chunk and concatenate
            if self.inference_type in (
                    constants.INFERENCE_SERVER_VLLM_CHAT_COMPLETION,
                    constants.OPENAI_CHAT_COMPLETION,
            ):
                for i in range(num_chunks):
                    start = i * max_samples
                    end = min((i + 1) * max_samples, total_samples)
                    chunk_array = audio_array[start:end]
                    encoded = encode_audio_array_base64(chunk_array, sampling_rate)

                    # Compose chunk-specific instruction
                    chunk_instructions = message.get("chunk_instructions")
                    if chunk_instructions and i < len(chunk_instructions):
                        chunk_instruction = chunk_instructions[i]
                        full_instruction = instruction + "\n" + chunk_instruction
                    else:
                        full_instruction = instruction

                    # Prepare messages list starting with system prompt if available
                    messages = []

                    # Add system prompt if available
                    system_prompt = message.get("system_prompt")
                    if system_prompt:
                        messages.append({
                            "role": "system",
                            "content": system_prompt
                        })

                    # Handle text-only vs audio+text scenarios
                    if encoded == "":
                        # Text-only case
                        messages.append({
                            "role": "user",
                            "content": [{"type": "text", "text": full_instruction}]
                        })
                    else:
                        # Audio + text case
                        messages.append({
                            "role": "user",
                            "content": [
                                {"type": "text", "text": full_instruction},
                                {
                                    "type": "input_audio",
                                    "input_audio": {
                                        "data": encoded,
                                        "format": "wav",
                                    },
                                },
                            ],
                        })
                    message["model_inputs"] = messages
                    resp = await self.req_resp_hndlr.request_server(message["model_inputs"], tools=tools, error_tracker=error_tracker)
                    concatenated_text += resp.llm_response or ""
                    responses.append(resp)
                # Merge responses
                final_resp = responses[-1]
                final_resp.llm_response = concatenated_text
                return final_resp

            # audio transcription - append values
            if self.inference_type in (
                    constants.TRANSCRIPTION,
            ):
                for i in range(num_chunks):
                    start = i * max_samples
                    end = min((i + 1) * max_samples, total_samples)
                    chunk_array = audio_array[start:end]
                    wav_path = audio_array_to_wav_file(chunk_array, sampling_rate)
                    # Pass closed file (file path) to request_server
                    resp = await self.req_resp_hndlr.request_server(wav_path, tools=None, error_tracker=error_tracker)
                    concatenated_text += resp.llm_response or ""
                    responses.append(resp)
                # ---------- Merge chunk responses ------------------
                final_resp = responses[-1]
                final_resp.llm_response = concatenated_text
                return final_resp
            raise ValueError("Unsupported inference type")

        # --- SINGLE-CHUNK LOGIC ---
        # chat completion
        if self.inference_type in (
                constants.INFERENCE_SERVER_VLLM_CHAT_COMPLETION,
                constants.OPENAI_CHAT_COMPLETION,
        ):
            # Cut to first 30s, then process as chat completion
            if audio_array is not None and len(audio_array) > 0:
                chunk_array = audio_array[:max_samples]
                encoded = encode_audio_array_base64(chunk_array, sampling_rate)
            else:
                encoded = ""

            # Prepare messages list starting with system prompt if available
            messages = []

            # Add system prompt if available
            system_prompt = message.get("system_prompt")
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })

            # Handle text-only vs audio+text scenarios
            if encoded == "":
                # Text-only case
                messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": instruction}]
                })
            else:
                # Audio + text case
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": encoded,
                                "format": "wav",
                            },
                        }
                    ],
                })
            message["model_inputs"] = messages
            return await self.req_resp_hndlr.request_server(message["model_inputs"], tools=tools, error_tracker=error_tracker)

        # transcription
        if self.inference_type in (
                constants.TRANSCRIPTION,
        ):
            wav_path = audio_array_to_wav_file(audio_array, sampling_rate)
            # Pass closed file (file path) to request_server
            resp = await self.req_resp_hndlr.request_server(wav_path, tools=None, error_tracker=error_tracker)
            return resp
        raise ValueError("Unsupported inference type")
