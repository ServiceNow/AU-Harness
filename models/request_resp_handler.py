import time
import httpx
from openai import AsyncOpenAI, OpenAI
from models.model_response import ModelResponse
from utils import constants
import logging
logger = logging.getLogger(__name__)  # handlers configured in logger_setup.py

class RequestRespHandler:
    """Class responsible for creating request and processing response for each type of inference server."""

    def __init__(self, inference_type: str, model_info: dict, timeout: int = 30):
        self.inference_type = inference_type
        self.model_info = model_info
        self.api = model_info.get("url")
        self.auth = model_info.get("auth_token", "")
        self.client = None
        self.timeout = timeout
        self.set_client(verify_ssl=True, timeout=self.timeout)
        # Remove Bearer if present for vllm/openai
        if self.inference_type in [
            constants.INFERENCE_SERVER_VLLM,
            constants.INFERENCE_SERVER_VLLM_COMPLETIONS,
        ] and self.auth.startswith("Bearer"):
            self.auth = self.auth.replace("Bearer ", "")

    def set_client(self, verify_ssl: bool, timeout: int):
        """Create HTTP/vLLM client for audio-to-text APIs."""
        if self.inference_type in [
            constants.INFERENCE_SERVER_VLLM,
            constants.INFERENCE_SERVER_VLLM_COMPLETIONS,
            constants.INFERENCE_SERVER_OPENAI,
        ]:
            self.client = (
                AsyncOpenAI(
                    base_url=self.api,
                    api_key=self.auth,
                    timeout=timeout,
                    max_retries=0,
                    default_headers={"Connection": "close"},
                    http_client=httpx.AsyncClient(verify=verify_ssl),
                )
            )
        else:
            # Generic HTTP client fallback (for OpenAI or similar APIs)
            self.client = httpx.AsyncClient(timeout=timeout, verify=verify_ssl)

    async def request_server(self, url, auth, msg_body, formatted_messages) -> ModelResponse:
        """Send a request to the inference server and return a `ModelResponse`.

        Logic:
        1. vLLM* servers – handled through the OpenAI-compatible SDK (`self.client.chat.completions`).
        2. Generic HTTP endpoint – we issue a POST request with an `Authorization` header derived from
           the supplied ``auth`` (if any).
        3. Any exception is wrapped in a `ModelResponse` with ``response_code = 500``.
        """
        model_name: str | None = self.model_info.get("model")
        params: dict = msg_body.get("parameters", {})
        messages = msg_body.get("messages", msg_body)

        start_time = time.time()
        try:
            # -------- vLLM path --------------------------------------------------
            if self.inference_type in [
                constants.INFERENCE_SERVER_VLLM,
                constants.INFERENCE_SERVER_VLLM_COMPLETIONS,
                constants.INFERENCE_SERVER_OPENAI,
            ]:
                logger.info("vLLM xpath")
                prediction = await self.client.chat.completions.create(
                    model=model_name, messages=messages, **params
                )
                raw_response: str = prediction.choices[0].message.content or " "
                llm_response: str = raw_response
                response_code: int = 200

            # -------- Generic HTTP path -----------------------------------------
            elif self.inference_type == constants.INFERENCE_SERVER:
                #logger.info("audio http")
                #logger.info(f"URL: {url}")
                #logger.info(f"Message body: {msg_body}")
                headers = {"Authorization": auth} if auth else {}
                #logger.info(f"Headers: {headers}")

                resp = await self.client.post(url, headers=headers, files=msg_body)
                raw_response = resp.text
                llm_response = self.get_response_text(raw_response)
                response_code = resp.status_code
            else:
                raise ValueError(f"Invalid inference type: {self.inference_type}")

            elapsed_time: float = time.time() - start_time
            if response_code == 200:
                logger.info(f"Successful Response")
            else:
                logger.error(f"Failed Response with code {response_code} and message {raw_response}")
            return ModelResponse(
                input_prompt=formatted_messages,
                llm_response=llm_response if llm_response else " ",
                raw_response=raw_response,
                response_code=response_code,
                performance=None,
                wait_time=elapsed_time,
                model_parameters=params,
            )

        except Exception as e:
            # inside RequestRespHandler before returning on error
            logger.error(f"audio_raw_response={e}")  
            # Failure – wrap the error and return early
            return ModelResponse(
                input_prompt=formatted_messages,
                llm_response="",
                raw_response=str(e),
                response_code=500,
                performance=None,
                wait_time=None,
                model_parameters=params,
            )

    def get_response_text(self, resp_text):
        """Extract transcription text from the response."""
        import json
        try:
            json_resp = json.loads(resp_text)
            for key in ["text", "generated_text", "transcript", "output"]:
                if key in json_resp:
                    return json_resp[key]
            return resp_text
        except Exception:
            return resp_text