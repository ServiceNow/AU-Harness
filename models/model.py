import asyncio
import re
from abc import ABC
import os
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_result,
    stop_after_attempt,
    wait_random,
    wait_random_exponential,
)

import logger_setup
logger_setup.configure()
import logging
logger = logging.getLogger(__name__)
logger.propagate = True
from models.model_response import ErrorTracker, ModelResponse
from models.request_resp_handler import RequestRespHandler
from copy import deepcopy
from utils import constants
from utils.multimodal import encode_audio_array_base64, audio_array_to_wav_file

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
        self.api_version = model_info.get("api_version", "")  # Optional API version
        self.inference_type = model_info["inference_type"]
        self.batch_size = model_info.get("batch_size", 1)
        # sleep before every call - in ms
        self.delay = model_info.get("delay", 100)
        self.timeout = model_info.get("timeout", 30)
        # max_wait for 8 attempts = 2^(8-1) = 128 secs
        self.retry_attempts = model_info.get("retry_attempts", 8)
        # chunk_size in seconds (default 30)
        self.chunk_size = model_info.get("chunk_size", 30)
        # some flow does not work with async client like internal private network
        self.postprocessor_path = model_info.get("postprocessor", [])
        #model_name = model_info.get("model", model_info.get("alias", self.name()))
        self.req_resp_hndlr = RequestRespHandler(
            self.inference_type,
            self.model_info,
            timeout=self.timeout,
        )
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


    #pure retry and exception logic
    async def _generate_text_with_retry(
        self, message: dict | str, run_params: dict
    ) -> ModelResponse:
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
                        result: ModelResponse = await self._generate_text(message, run_params)
                        await self._mark_errors(result)
                    except Exception as e:
                        logger.error(f"Exception during text generation: {e}")
                        result = ModelResponse(
                            input_prompt=str(message),
                            llm_response="",
                            raw_response=str(e),
                            response_code=500,
                            performance=None,
                            wait_time=0,
                        )
                attempt.retry_state.set_result(result)
                # Set backoff for next retry based on current result
                self._set_max_backoff(attempt)
                if not self._is_retryable_error(result):
                    break
        except KeyError as e:
            logger.error(f"Missing key while building model_inputs on the fly: {e}")
            result = ModelResponse(
                input_prompt=message,
                llm_response="",
                raw_response=f"Missing key: {e}",
                response_code=500,
                performance=None,
                wait_time=0,
            )
        except RetryError:
            logger.error(
                f"[{self.name()}] Request failed after {self.retry_attempts} attempts for input: {message}..."
            )
            result = ModelResponse(
                input_prompt=message,
                llm_response="",
                raw_response="RetryError: Request failed after max attempts",
                response_code=500,
                performance=None,
                wait_time=0,
            )
        except Exception as e:
            logger.error(f"Unexpected error in _generate_text_with_retry: {e}")
            result = ModelResponse(
                input_prompt=message,
                llm_response="",
                raw_response=str(e),
                response_code=500,
                performance=None,
                wait_time=0,
            ) 
        return result






    async def _generate_text(self, message: dict, run_params: dict) -> ModelResponse:
        """
        Implements model query by building message header and body with the help of Request Response Handler.
        Args:
            message: inputs for inference
            run_params: params for the run
        Returns:
            Response and http return code
        """



        #getting attributes
        audio_array = message.get("array")
        sampling_rate = message.get("sampling_rate")
        chunk_seconds: int = int(run_params.get("chunk_size", 30))  # default to 30s
        metric_name: str | None = run_params.get("metric")
        max_samples: int = int(chunk_seconds * sampling_rate) if sampling_rate else 0
        total_samples: int = len(audio_array) if audio_array is not None else 0
        instruction = message.get("instruction")

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
                    
                    # Add user message with instruction and audio
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
                    
                    resp = await self.req_resp_hndlr.request_server(message["model_inputs"])
                    logger.info(f"Requester response: {resp}")
                    concatenated_text += resp.llm_response or ""
                    responses.append(resp)
                # Merge responses
                final_resp = responses[-1]
                final_resp.llm_response = concatenated_text
                return final_resp

            #audio transcription - append values
            elif self.inference_type in (
                constants.INFERENCE_SERVER_VLLM_TRANSCRIPTION,
                constants.OPENAI_TRANSCRIPTION,
            ):
                for i in range(num_chunks):
                    start = i * max_samples
                    end = min((i + 1) * max_samples, total_samples)
                    chunk_array = audio_array[start:end]
                    wav_path = audio_array_to_wav_file(chunk_array, sampling_rate)
                    # Pass closed file (file path) to request_server
                    resp = await self.req_resp_hndlr.request_server(wav_path)
                    concatenated_text += resp.llm_response or ""
                    responses.append(resp)
                # ---------- Merge chunk responses ------------------
                final_resp = responses[-1]
                final_resp.llm_response = concatenated_text
                return final_resp
            else:
                raise ValueError("Unsupported inference type")




        # --- SINGLE-CHUNK LOGIC ---
        #chat completion
        if self.inference_type in (
            constants.INFERENCE_SERVER_VLLM_CHAT_COMPLETION,
            constants.OPENAI_CHAT_COMPLETION,
        ):
            # Cut to first 30s, then process as chat completion
            chunk_array = audio_array[:max_samples]
            encoded = encode_audio_array_base64(chunk_array, sampling_rate)

            # Prepare messages list starting with system prompt if available
            messages = []
            
            # Add system prompt if available
            system_prompt = message.get("system_prompt")
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            # Add user message with instruction and audio
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
            
            return await self.req_resp_hndlr.request_server(message["model_inputs"])

        #transcription
        elif self.inference_type in (
            constants.INFERENCE_SERVER_VLLM_TRANSCRIPTION,
            constants.OPENAI_TRANSCRIPTION,
        ):
            wav_path = audio_array_to_wav_file(audio_array, sampling_rate)
            # Pass closed file (file path) to request_server
            resp = await self.req_resp_hndlr.request_server(wav_path)
            return resp
        else:
            raise ValueError("Unsupported inference type")