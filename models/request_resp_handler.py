import time
import httpx
from openai import AsyncAzureOpenAI, AsyncOpenAI
from models.model_response import ModelResponse
from utils import constants
import logging
import requests
logger = logging.getLogger(__name__)  # handlers configured in logger_setup.py
import json
import os
import httpx

class RequestRespHandler:
    """Class responsible for creating request and processing response for each type of inference server."""

    def __init__(self, inference_type: str, model_info: dict, timeout: int = 30):
        self.inference_type = inference_type
        self.model_info = model_info
        self.api = model_info.get("url")
        self.auth = model_info.get("auth_token", "")
        self.api_version = model_info.get("api_version", "")
        self.client = None
        self.timeout = timeout
        # current retry attempt (set by caller). Default 1.
        self.current_attempt: int = 1
        # Remove Bearer if present for vllm/openai
        
        # Strip 'Bearer ' ONLY for OpenAI flows – VLLM endpoints expect full 'Bearer <token>'
        if self.inference_type in [
            constants.OPENAI_TRANSCRIPTION,
            constants.OPENAI_CHAT_COMPLETION,
            constants.INFERENCE_SERVER_VLLM_CHAT_COMPLETION,
        ] and self.auth.startswith("Bearer "):
            self.auth = self.auth.replace("Bearer ", "")
        self.set_client(verify_ssl=True, timeout=self.timeout)

    def set_client(self, verify_ssl: bool, timeout: int):
        #use python client wrapper
        #vllm chat completions compatibility
        if self.inference_type in [
            constants.OPENAI_CHAT_COMPLETION,
            constants.OPENAI_TRANSCRIPTION
        ]:
            # Azure OpenAI endpoints
            self.client = (
                AsyncAzureOpenAI(
                    azure_endpoint=self.api,
                    api_key=self.auth,
                    api_version=self.api_version,
                    timeout=timeout,
                    max_retries=0,
                    default_headers={"Connection": "close"},
                    http_client=httpx.AsyncClient(verify=verify_ssl),
                )
            )
        elif self.inference_type == constants.INFERENCE_SERVER_VLLM_CHAT_COMPLETION:
            # vLLM endpoints (OpenAI-compatible, no api_version)
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
        #basic post
        elif self.inference_type in [
            constants.INFERENCE_SERVER_VLLM_TRANSCRIPTION,
        ]:
            self.client = httpx.AsyncClient(timeout=timeout, verify=verify_ssl)
        else:
            raise ValueError(f"Invalid inference type: {self.inference_type}")

    async def request_server(self, msg_body) -> ModelResponse:
        """Send a request to the inference server and return a `Model Response`.

        Logic:
        1. vLLM* servers – handled through the OpenAI-compatible SDK (`self.client.chat.completions`).
        2. Any exception is wrapped in a `ModelResponse` with ``response_code = 500``.
        """
        model_name: str | None = self.model_info.get("model")

        start_time = time.time()
        # Re-create a fresh client for this request to avoid closed-loop issues
        self.set_client(verify_ssl=True, timeout=self.timeout)


        #same input, different calls
        try:
            # -------- vLLM transcription path --------------------------------------------------
            #vllm transcription
            if self.inference_type == constants.INFERENCE_SERVER_VLLM_TRANSCRIPTION:

                # Ensure 'Bearer' prefix for VLLM
                if self.auth and not self.auth.startswith("Bearer "):
                    auth_header = f"Bearer {self.auth}"
                else:
                    auth_header = self.auth
                headers = {"Authorization": auth_header} if auth_header else {}

                if isinstance(msg_body, str) and msg_body.endswith('.wav'):
                    with open(msg_body, "rb") as f:
                        files = {"file": (os.path.basename(msg_body), f, "audio/wav")}
                        async with httpx.AsyncClient(timeout=self.timeout, verify=True) as _client:
                            resp = await _client.post(self.api, headers=headers, files=files)
                else:
                    raise ValueError("Invalid input: msg_body must be a wav file path")
                raw_response = resp.text
                llm_response = self.get_response_text(raw_response)
                response_code = 200
                elapsed_time: float = time.time() - start_time
                #logger.info(f"Successful post request: {response_code}")
                return ModelResponse (
                input_prompt=str(msg_body),
                llm_response=llm_response if llm_response else " ",
                raw_response=raw_response,
                response_code=response_code,
                performance=None,
                wait_time=elapsed_time,
                )

            #openai chat completions, vllm chat completions
            elif self.inference_type in [constants.OPENAI_CHAT_COMPLETION, constants.INFERENCE_SERVER_VLLM_CHAT_COMPLETION]:
                prediction = await self.client.chat.completions.create(
                    model=model_name, messages=msg_body
                )
                response_data = prediction.model_dump()
                raw_response: str = response_data
                llm_response: str = response_data['choices'][0]['message']['content'] or " "
                response_code: int = 200
                logger.info(f"Successful post request: {response_code}")
                logger.info(f"LLM response: {llm_response}")
                elapsed_time: float = time.time() - start_time

                return ModelResponse (
                input_prompt=str(msg_body[0]["content"][0]["text"]),
                llm_response=llm_response if llm_response else " ",
                raw_response=raw_response,
                response_code=response_code,
                performance=None,
                wait_time=elapsed_time,
                )

            #openai transcription
            elif self.inference_type == constants.OPENAI_TRANSCRIPTION:
                prediction = await self.client.audio.transcriptions.create(
                    model=model_name, file=f
                )
                response_data = prediction.model_dump_json()
                raw_response: str = response_data
                llm_response: str = response_data['text'] or " "
                response_code: int = 200
                #logger.info(f"Successful post request: {response_code}")
                elapsed_time: float = time.time() - start_time

                return ModelResponse (
                input_prompt=str(msg_body),
                llm_response=llm_response if llm_response else " ",
                raw_response=raw_response,
                response_code=response_code,
                performance=None,
                wait_time=elapsed_time,
                )

        except Exception as e:
            logger.error(f"Attempt {self.current_attempt}: audio_raw_response={e!r}")
            # First attempt to wrap the error in ModelResponse
            try:
                return ModelResponse(
                    input_prompt=str(msg_body) if msg_body is not None else "error",
                    llm_response="",
                    raw_response=str(e),
                    response_code=500,
                    performance=None,
                    wait_time=0,
                )
            except Exception as inner:
                logger.error(f"ModelResponse construction failed: {inner!r}")
                # Second ultra-safe fallback with hard-coded fields
                return ModelResponse(
                    input_prompt="error",
                    llm_response="",
                    raw_response="special error",
                    response_code=500,
                    performance=None,
                    wait_time=0,
                )
    def get_response_text(self, resp_text):
        """Extract transcription text from the response."""
        try:
            json_resp = json.loads(resp_text)
            for key in ["text", "generated_text", "transcript", "output"]:
                if key in json_resp:
                    return json_resp[key]
            return resp_text
        except Exception:
            return resp_text