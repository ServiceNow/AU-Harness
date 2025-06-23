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
from utils import constants, util
from utils.input_builder import InputBuilder
from utils.measurement_stats import MeasurementStats
from utils.model_perf_stats import PerfStats
from utils.weighted_map import WeightedMap, WeightedValue


class Model(ABC):
    """TODO: Need SME to add."""

    def __init__(self, model_info: dict):
        """Initialize model configuration here.

        Args:
            model_info: model configuration dictionary
        """
        self.model_info = model_info
        # sleep before every call - in ms
        self.delay = model_info.get("delay", 100)
        # max_wait for 8 attempts = 2^(8-1) = 128 secs
        self.retry_attempts = model_info.get("retry_attempts", 8)

        # some flow does not work with async client like internal private network
        self.postprocessor_path = model_info.get("postprocessor", [])
        model_name = model_info.get("model", model_info.get("alias", self.name()))
        self.req_resp_hndlr = RequestRespHandler(
            self.inference_type,
            self.model_info,
        )
        self.input_builder = InputBuilder(self.name())

        self.weighted_params = self._create_weighted_elements(model_info, model_name)

        # prevent data races when updating self.errors asynchronously
        self.errors_lock = asyncio.Lock()
        self.errors = ErrorTracker()

    

    def _create_weighted_elements(self, model_info: dict, model_name: str) -> WeightedMap:
        """Creates mapping of endpoint info to support multiple endpoints. Backwards compatible."""
        endpoint_lst = self._get_endpoints(model_info, model_name)

        weighted_elements = []
        for endpoint in endpoint_lst:
            param = {
                "url": endpoint.get("url", ""),
                "auth": endpoint.get("auth_token", ""),
                "timeout": self.model_info.get("timeout", 30),
                "max_retries": 0,
            }

            weighted_elements.append(WeightedValue(value=param, weight=endpoint["weight"]))
        return WeightedMap(weighted_elements)

    def _get_endpoints(self, model_info: dict, model_name: str) -> list[dict]:
        """Retries endpoint info and assigns uniform weights if no weight specified."""
        if "," in model_info.get("url", ""):
            raise Exception("Separate URLs with | instead of with a comma.")
        if "," in model_info.get("auth_token", ""):
            raise Exception("Separate tokens with | instead of with a comma.")

        urls = model_info.get("url", "").split("|")
        auth_tokens = model_info.get("auth_token", "").split("|")
        weights = []

        ## Not possible to provide both model_path and a endpoint. Must be one or the other
        ## For localhost, there is no need for authorization tokens. Supplying auth tokens or token types leads to 404 error
        ## Check if localhost in any
        ## Edge Case: Model Endpoint URL does have "localhost:" in which case this would fail.
        ## Sure fire way is to see if model path is provided.
        found_localhost = any("localhost" in url for url in urls)
        auth_tokens = [""] * len(urls) if found_localhost else auth_tokens

        if model_info.get("weights"):
            if "," in model_info["weights"]:
                raise Exception("Separate weights with | instead of with a comma.")
            weights = model_info["weights"].split("|")

        if len(urls) != len(auth_tokens):
            raise Exception(f"Number of URLs does not match number of tokens for model {model_name}")

        if weights:
            if len(urls) != len(weights):
                raise Exception(f"Number of URLs does not match number of weights for model {model_name}")
        else:
            weights = [1 / len(urls) for _ in range(len(urls))]

        if cache_url := model_info.get("cache_url"):
            urls = [util.add_cache_url(cache_url=cache_url, url=url) for url in urls]

        endpoint_info = [{"url": urls[i], "auth_token": auth_tokens[i], "weight": weights[i]} for i in range(len(urls))]
        return endpoint_info

    @abstractmethod
    def name(self):
        """TODO: Need SME to add."""
        pass

    def aliased_name(self) -> str:
        """TODO: Need SME to add."""
        alias = self.model_info.get("alias", None)
        return alias if alias else self.name()

    @staticmethod
    def _get_seed_from_replica_id(record_id: str) -> dict[str, int]:
        """Get seed from replica id.

        Args:
            record_id: The record id.

        Returns:
            A `dict` to override the model params. If `record_id` contains a replica id, the returned `dict` contains a
            `seed` with the replica id. Otherwise, the returned `dict` is empty.
        """
        _, _, replica = record_id.partition(constants.REPLICA)
        return {constants.SEED: int(replica)} if replica else {}

    def generate_text(
        self,
        messages: list[dict] | str,
        run_params: dict,
        model_params: dict[str, object],
        call_id: str,
    ) -> tuple[str, int]:
        """Generates inference against a single text. Run params are defined in runspec, model params are what is passed to model.

        Args:
            messages: List of role based messages for model (or a formatted string)
            run_params: Params to be used for specific run
            model_params: Params defined for the model
            call_id: Unique ID to define the step the model is being called for for a given runspec

        Returns: Model response and response code.
        """
        model_params = model_params or {}
        inference_type = getattr(self, "inference_type", None)
        for var in model_params:
            if var in constants.MODEL_PARAMS_TO_KEEP_IN_RUN_PARAMS:
                run_params[var] = model_params[var]
        mapped_params = util.model_param_mapper(self.name(), model_params, inference_type)
        model_name = str(self.__class__.__name__)
        model_name = f"{model_name}_{call_id}"
        start = time.time()
        # Directly generate text without reasoning-budget wrapper for audio models
        final_model_response = asyncio.run(
            self._generate_text_with_retry(messages, mapped_params, run_params)
        )

        # Set after all errors are tracked
        final_model_response.error_tracker = self.errors

        inference_type = getattr(self, "inference_type", "")
        if call_id != constants.CALL_ID_TEST:
            ps = PerfStats()
            ps.add_all_stats(run_params.get("runspec_name"), model_name, final_model_response)
            ps.add(
                run_params.get("runspec_name"),
                model_name,
                "total_time",
                float(final_model_response.performance.latency)
                if final_model_response.performance
                else time.time() - start,
            )
            MeasurementStats().add_all_stats(
                run_params.get("runspec_name"), model_name, final_model_response, inference_type
            )
        return final_model_response.llm_response, final_model_response.response_code


    

    def _log_performance(self, model_response, model_name, rec_id, run_params, start):
        ps = PerfStats()
        latency = model_response.performance.latency if model_response.performance else (time.time() - start)
        ps.add(run_params.get("runspec_name"), model_name, "id", rec_id)
        ps.add_all_stats(run_params.get("runspec_name"), model_name, model_response)
        ps.add(run_params.get("runspec_name"), model_name, "total_time", latency)

    def _log_measurement(self, model_response, model_name, rec_id, run_params):
        ms = MeasurementStats()
        ms.add(run_params.get("runspec_name"), model_name, "id", rec_id)
        ms.add_all_stats(
            run_params.get("runspec_name"), model_name, model_response, getattr(self, "inference_type", "")
        )


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

    def _process_text(self, text: str, run_params: dict) -> str:
        return text

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

    def _test(self) -> bool:
        """This method is used to ping test the model before running a runspec.

        Returns:
            Returns true if it is able to connect
        """
        resp_text, resp_code = self.generate_text(
            constants.MESSAGES_TEST_AVAILABILITY,
            model_params={"max_tokens": 1},
            run_params={},
            call_id=constants.CALL_ID_TEST,
        )
        return resp_code == 200

    def get_model_info(self):
        """Get model info for this model.

        Returns:
            Model info json
        """
        return self.model_info

    def validate_error_code_message(self, response: str, code: int) -> bool:
        """Validates if the response is an error or not. If the API call is successful or if it is OpenAI ResponsibleAIPolicyViolation then it is not considered an error.

        Args:
            response: response from the API request
            code: response code from the API request

        Returns:
            True for valid error cases and False for invalid error cases.
        """
        if (code == 200) or (
            constants.OPENAI_VIOLATION_SUBSTRING.lower() in response.lower() and code == HTTPStatus.UNAUTHORIZED
        ):
            return False
        return True

    def postprocess(self, postprocessor_path: list[str], text: str, response_format: str):
        """Postprocess the text using the postprocessor."""
        for path in postprocessor_path:
            module, class_name = path.rsplit(".", 1)
            postprocessor = getattr(importlib.import_module(module), class_name)
            text = postprocessor().postprocess(text, response_format)
        return text
