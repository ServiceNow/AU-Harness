"""Request response handler for various inference servers."""
import logging
import re
import time
import inspect
import tempfile
import numpy as np
import soundfile as sf
import os
import httpx
from openai import AsyncAzureOpenAI, AsyncOpenAI
from cartesia import AsyncCartesia
from elevenlabs.client import AsyncElevenLabs
from deepgram import AsyncDeepgramClient
from models.model_response import ModelResponse, ErrorTracker
from utils import constants

logger = logging.getLogger(__name__)  # handlers configured in utils/logging.py


class RequestRespHandler:
    """Class responsible for creating request and processing response for each type of inference server."""

    def __init__(self, inference_type: str, model_info: dict, generation_params: dict, timeout: int = 30):
        self.inference_type = inference_type
        self.model_info = model_info
        self.api = model_info.get("url")
        self.auth = model_info.get("auth_token", "")
        self.api_version = model_info.get("api_version", "")
        self.client = None
        self.timeout = timeout
        # Use provided generation params
        self.generation_params = generation_params
        # current retry attempt (set by caller). Default 1.
        self.current_attempt: int = 1
        # Remove Bearer if present for vllm/openai
        if self.inference_type in [
            constants.TRANSCRIPTION,
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

    def _extract_response_data(self, prediction) -> str:
        """Extract response data from prediction object."""
        response_data = prediction.model_dump()
        return response_data

   

    def convert_to_tool(self, functions):
        """Convert functions to OpenAI tool format."""
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
        """Set up the appropriate client based on inference type."""
        # use python client wrapper
        # vllm chat completions compatibility
        if self.inference_type in [
            constants.OPENAI_CHAT_COMPLETION,
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
        elif self.inference_type == constants.INFERENCE_SERVER_VLLM_CHAT_COMPLETION or self.inference_type == constants.TRANSCRIPTION:
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
        elif self.inference_type == constants.CARTESIA_TTS:
            # Cartesia TTS client
            self.client = AsyncCartesia(api_key=self.auth)

        elif self.inference_type == constants.ELEVENLABS_TTS:
            # ElevenLabs TTS client
            self.client = AsyncElevenLabs(api_key=self.auth)

        elif self.inference_type == constants.DEEPGRAM_TTS:
            # Deepgram TTS async client
            self.client = AsyncDeepgramClient(api_key=self.auth)

    def validated_safe_generation_params(self, generation_params):
        """Validate and sanitize generation parameters for the OpenAI API client.
        
        This function filters out invalid parameters that are not accepted by the OpenAI API client,
        logs a warning for any ignored parameters, and ensures that required parameters like
        temperature and max_completion_tokens have default values if not provided.
        
        Args:
            generation_params (dict): Dictionary of generation parameters to validate
            
        Returns:
            dict: Sanitized dictionary containing only valid parameters with default values added as needed
        """
        valid_params = inspect.signature(self.client.chat.completions.create).parameters
        safe_params = {k: v for k, v in generation_params.items() if k in valid_params}

        if safe_params != generation_params:
            ignored_params = {k: v for k, v in generation_params.items() if k not in safe_params}
            logger.warning("Ignoring invalid generation parameters %s and setting required params to defaults", list(ignored_params.keys()))

        safe_params['temperature'] = safe_params.get('temperature', constants.DEFAULT_TEMPERATURE)
        safe_params['max_completion_tokens'] = safe_params.get('max_completion_tokens', constants.DEFAULT_MAX_COMPLETION_TOKENS)

        return safe_params

    async def request_tts_server(self, text: str, model_name: str, voice_id: str,
                                 start_time: float, error_tracker: ErrorTracker) -> ModelResponse:
        """Helper function for TTS request processing.

        Args:
            text: Text to convert to speech
            model_name: TTS model name
            voice_id: Voice ID for TTS
            start_time: Request start time for performance tracking
            error_tracker: Error tracker for this call

        Returns:
            ModelResponse: Response object with audio file path
        """
        if self.inference_type == constants.CARTESIA_TTS:
            # Cartesia TTS generation
            bytes_iter = self.client.tts.bytes(
                model_id=model_name,
                transcript=text,
                voice={"mode": "id", "id": voice_id},
                output_format={
                    "container": "wav",
                    "sample_rate": 16000,
                    "encoding": "pcm_s16le",
                }
            )

            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", mode='wb') as f:
                async for chunk in bytes_iter:
                    f.write(chunk)
                audio_path = f.name

            elapsed_time = time.time() - start_time

            # Get file size for response
            audio_bytes_len = os.path.getsize(audio_path)

            return ModelResponse(
                input_prompt=text,
                llm_response=audio_path,
                raw_response={"audio_bytes": audio_bytes_len, "audio_path": audio_path},
                response_code=200,
                performance=None,
                wait_time=elapsed_time,
                error_tracker=error_tracker,
            )

        elif self.inference_type == constants.ELEVENLABS_TTS:
            # ElevenLabs TTS generation
            audio = self.client.text_to_speech.convert(
                text=text,
                voice_id=voice_id,
                model_id=model_name,
                output_format="pcm_16000"
            )

            # Collect PCM chunks
            pcm_chunks = []
            async for chunk in audio:
                if isinstance(chunk, bytes):
                    pcm_chunks.append(chunk)

            pcm_data = b''.join(pcm_chunks)

            # Convert PCM to WAV with headers using soundfile
            audio_array = np.frombuffer(pcm_data, dtype=np.int16)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                sf.write(f.name, audio_array, 16000, format='WAV')
                audio_path = f.name

            elapsed_time = time.time() - start_time
            return ModelResponse(
                input_prompt=text,
                llm_response=audio_path,
                raw_response={"audio_bytes": len(pcm_data), "audio_path": audio_path},
                response_code=200,
                performance=None,
                wait_time=elapsed_time,
                error_tracker=error_tracker,
            )

        elif self.inference_type == constants.DEEPGRAM_TTS:
            # Deepgram TTS generation
            response = self.client.speak.v1.audio.generate(
                text=text,
                model=model_name,
            )

            # response is an async generator, collect chunks
            audio_chunks = []
            async for chunk in response:
                if isinstance(chunk, bytes):
                    audio_chunks.append(chunk)

            audio_data = b''.join(audio_chunks)

            # Write to temporary WAV file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", mode='wb') as f:
                f.write(audio_data)
                audio_path = f.name

            elapsed_time = time.time() - start_time

            return ModelResponse(
                input_prompt=text,
                llm_response=audio_path,
                raw_response={"audio_bytes": len(audio_data), "audio_path": audio_path},
                response_code=200,
                performance=None,
                wait_time=elapsed_time,
                error_tracker=error_tracker,
            )

    async def request_server(self, msg_body, tools=None, error_tracker: ErrorTracker = None) -> ModelResponse:
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
        # Skip for TTS clients - they should be reused to avoid file descriptor leaks
        if self.inference_type not in (constants.CARTESIA_TTS, constants.ELEVENLABS_TTS, constants.DEEPGRAM_TTS):
            self.set_client(verify_ssl=True, timeout=self.timeout)

        try:
            # Handle TTS requests
            if self.inference_type in (constants.CARTESIA_TTS, constants.ELEVENLABS_TTS, constants.DEEPGRAM_TTS):
                text = msg_body.get("text")
                voice_id = self.model_info.get("voice_id")
                return await self.request_tts_server(text, model_name, voice_id, start_time, error_tracker)
            elif self.inference_type == constants.OPENAI_CHAT_COMPLETION or self.inference_type == constants.INFERENCE_SERVER_VLLM_CHAT_COMPLETION:
                # openai chat completions, vllm chat completions
                self.generation_params = self.validated_safe_generation_params(self.generation_params)
                prediction = await self.client.chat.completions.create(
                    model=model_name, messages=msg_body, tools=tools, **self.generation_params
                )
                raw_response: str = self._extract_response_data(prediction)
                llm_response: str = raw_response['choices'][0]['message']['content'] or " "

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

            elif self.inference_type == constants.TRANSCRIPTION:
                # msg_body is a file path, need to open it as file object
                with open(msg_body, "rb") as audio_file:
                    prediction = await self.client.audio.transcriptions.create(
                        model=model_name, file=audio_file
                    )
                user_prompt = str(msg_body)
                raw_response: str = self._extract_response_data(prediction)
                llm_response: str = raw_response['text'] or " "

            elapsed_time = time.time() - start_time
            return ModelResponse(
                input_prompt=user_prompt,
                llm_response=llm_response if llm_response else " ",
                raw_response=raw_response,
                response_code=200,
                performance=None,
                wait_time=elapsed_time,
            )

        except (httpx.RequestError, httpx.HTTPStatusError, ValueError, OSError) as e:
            logger.error("Attempt %s", self.current_attempt)
            # First attempt to wrap the error in ModelResponse
            try:
                # Increment the appropriate counter based on error type
                if "Connection error" in str(e):
                    error_tracker.connection_error += 1
                elif "rate limit" in str(e).lower() or "rate_limit" in str(e).lower():
                    error_tracker.rate_limit += 1
                elif "timeout" in str(e).lower():
                    error_tracker.request_timeout += 1
                elif "internal server" in str(e).lower() or "5" == str(e)[0]:
                    error_tracker.internal_server += 1
                elif "4" == str(e)[0]:
                    error_tracker.api_error += 1
                else:
                    error_tracker.other += 1
                return ModelResponse(
                    input_prompt=str(msg_body) if msg_body is not None else "error",
                    llm_response="",
                    raw_response=str(e),
                    response_code=500,
                    performance=None,
                    wait_time=0,
                    error_tracker=error_tracker,
                )
            except (AttributeError, TypeError, ValueError) as inner:
                logger.error("ModelResponse construction failed: %r", inner)
                # Second ultra-safe fallback with hard-coded fields
                # Use provided error_tracker or create a fallback one
                if not error_tracker:
                    error_tracker = ErrorTracker()
                error_tracker.other += 1
                return ModelResponse(
                    input_prompt="error",
                    llm_response="",
                    raw_response="special error",
                    response_code=500,
                    performance=None,
                    wait_time=0,
                    error_tracker=error_tracker
                )
