import time
import re
import httpx
from openai import AsyncAzureOpenAI, AsyncOpenAI
from torch.utils.hipify.hipify_python import mapping

from models.model_response import ModelResponse
from utils import constants
import logging
import requests
logger = logging.getLogger(__name__)  # handlers configured in utils/logging.py
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
        if self.inference_type in [
            constants.OPENAI_TRANSCRIPTION,
            constants.OPENAI_CHAT_COMPLETION,
            constants.INFERENCE_SERVER_VLLM_CHAT_COMPLETION,
        ] and self.auth.startswith("Bearer "):
            self.auth = self.auth.replace("Bearer ", "")
        self.set_client(verify_ssl=True, timeout=self.timeout)

    def _cast_to_openai_type(self, properties, mapping):
        for key, value in properties.items():
            if "type" not in value:
                properties[key]["type"] = "string"
            else:
                var_type = value["type"]
                if var_type == "float":
                    properties[key]["format"] = "float"
                    properties[key]["description"] += " This is a float type value."
                if var_type in mapping:
                    properties[key]["type"] = mapping[var_type]
                else:
                    properties[key]["type"] = "string"

            # Currently support:
            # - list of any
            # - list of list of any
            # - list of dict
            # - list of list of dict
            # - dict of any

            if properties[key]["type"] == "array" or properties[key]["type"] == "object":
                if "properties" in properties[key]:
                    properties[key]["properties"] = self._cast_to_openai_type(
                        properties[key]["properties"], mapping
                    )
                elif "items" in properties[key]:
                    properties[key]["items"]["type"] = mapping[properties[key]["items"]["type"]]
                    if (
                            properties[key]["items"]["type"] == "array"
                            and "items" in properties[key]["items"]
                    ):
                        properties[key]["items"]["items"]["type"] = mapping[
                            properties[key]["items"]["items"]["type"]
                        ]
                    elif (
                            properties[key]["items"]["type"] == "object"
                            and "properties" in properties[key]["items"]
                    ):
                        properties[key]["items"]["properties"] = self._cast_to_openai_type(
                            properties[key]["items"]["properties"], mapping
                        )
        return properties
    def convert_to_tool(self, functions):
        mapping = {
            "integer": "integer",
            "number": "number",
            "float": "number",
            "string": "string",
            "boolean": "boolean",
            "bool": "boolean",
            "array": "array",
            "list": "array",
            "dict": "object",
            "object": "object",
            "tuple": "array",
            "any": "string",
            "byte": "integer",
            "short": "integer",
            "long": "integer",
            "double": "number",
            "char": "string",
            "ArrayList": "array",
            "Array": "array",
            "HashMap": "object",
            "Hashtable": "object",
            "Queue": "array",
            "Stack": "array",
            "Any": "string",
            "String": "string",
            "Bigint": "integer",
        }
        new_functions = []
        for item in functions:
            item["name"] = re.sub(r"\.", "_", item["name"])
            item["parameters"]["type"] = "object"
            item["parameters"]["properties"] = self._cast_to_openai_type(
                item["parameters"]["properties"], mapping
            )

            new_functions.append({"type": "function", "function": item})

        return new_functions


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

    async def request_server(self, msg_body, tools=None) -> ModelResponse:
        """Send a request to the inference server and return a `Model Response`.

        Logic:
        1. vLLM* servers – handled through the OpenAI-compatible SDK (`self.client.chat.completions`).
        2. Any exception is wrapped in a `ModelResponse` with ``response_code = 500``.
        """
        model_name: str | None = self.model_info.get("model")
        if tools:
            tools = self.convert_to_tool(tools)

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
                    model=model_name, messages=msg_body, tools=tools
                )
                response_data = prediction.model_dump()
                raw_response: str = response_data
                llm_response: str = response_data['choices'][0]['message']['content'] or " "
                response_code: int = 200
                elapsed_time: float = time.time() - start_time

                # Find the user message to extract input prompt
                user_prompt = ""
                for message in msg_body:
                    if message["role"] == "user" and "content" in message and isinstance(message["content"], list):
                        for content_item in message["content"]:
                            if content_item.get("type") == "text":
                                user_prompt = content_item.get("text", "")
                                break
                        if user_prompt:
                            break
                
                return ModelResponse (
                input_prompt=user_prompt,
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