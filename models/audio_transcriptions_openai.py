import tempfile

import numpy as np
import soundfile as sf
import os

import logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    fh = logging.FileHandler("audiobench.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)
logging.basicConfig(level=logging.INFO)
logger.propagate = True
from models.model import Model
from models.model_response import ModelResponse
from utils import constants

MAX_RETRIES = 2
OPENAI_SUPPORTED_SOUND_FORMAT = ["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"]


class AudioTranscriptionsOpenAI(Model):
    """Model Class for transcription models that use the transcriptions OpenAI API."""

    def name(self):
        """Return the name of the model."""
        return constants.MODEL_TRANSCRIPTIONS_OPENAI

    def __init__(self, model_info):
        # Force inference type for audio model
        model_info.setdefault("inference_type", constants.INFERENCE_SERVER)
        self.inference_type = model_info["inference_type"]
        super().__init__(model_info)
        self.model = model_info["model"]
        self.api_key = model_info.get("auth_token", "")
        self.model_url = model_info.get("url", "")

    @staticmethod
    def is_path_supported(audio_file_path) -> bool:
        """Check if the audio file path can directly be used with the OpenAI API."""
        return (
            audio_file_path
            and any(audio_file_path.endswith(sound_format) for sound_format in OPENAI_SUPPORTED_SOUND_FORMAT)
            and audio_file_path.startswith("/tmp")
        )

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
        language = message.get("language")
        audio_file_path = message.get("path")

        fp = None
        if not self.is_path_supported(audio_file_path):
            fp = tempfile.NamedTemporaryFile(suffix=".wav")
            sf.write(fp, audio_array, sampling_rate)
            audio_file_path = fp.name

        # Build message body for request handler in correct format for HTTP file upload
        params = {"model": self.model, "language": language}
        files = None
        if audio_file_path:
            f = open(audio_file_path, "rb")
            files = {"file": (os.path.basename(audio_file_path), f, "audio/wav")}
        else:
            raise ValueError("audio_file_path must be provided for OpenAI transcription")

        current_index = self.weighted_params.get_random_index()
        endpoint = self.weighted_params.listing[current_index]
        try:
            model_response: ModelResponse = await self.req_resp_hndlr.request_server(
                url=endpoint["url"],
                auth=endpoint["auth"],
                msg_body=files,
                formatted_messages=message,
            )
        finally:
            if f:
                f.close()
        return model_response

    def test(self) -> bool:
        """Call the model to test if it is available."""
        # Create dummy audio file
        resp_text, resp_code = self.generate_text(
            [{"array": np.random.rand(1000), "sampling_rate": 16000}], {}, {}, call_id="Test"
        )
        return resp_code == 200
