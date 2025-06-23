import json
import logging
import tempfile

import numpy as np
import requests
import soundfile as sf

from models.model import Model
from models.model_response import ModelResponse
from utils import constants

MAX_RETRIES = 2
NVIDIA_SUPPORTED_SOUND_FORMAT = ["wav"]
LANGUAGE_MAP = {
    "de": "de-DE",
    "en": "en-US",
    "es": "es-ES",
    "fr": "fr-FR",
    "it": "it-IT",
    "ja": "ja-JP",
    "nl": "nl-NL",
    "pt": "pt-BR",
}


class AudioTranscriptionsNvidia(Model):
    """Model Class for transcription models that use the transcriptions Nvidia API."""

    def name(self):
        """Return the name of the model."""
        return constants.MODEL_TRANSCRIPTIONS_NVIDIA

    def __init__(self, model_info):
        super().__init__(model_info)
        self.model = model_info["model"]
        self.api_key = model_info["auth_token"]
        self.model_url = model_info["url"]

    @staticmethod
    def is_path_supported(audio_file_path) -> bool:
        """Check if the audio file path can directly be used with the NVIDIA API."""
        return (
            audio_file_path
            and any(audio_file_path.endswith(sound_format) for sound_format in NVIDIA_SUPPORTED_SOUND_FORMAT)
            and audio_file_path.startswith("/tmp")
        )

    async def _generate_text(self, messages: list[dict] | str, model_params: dict, run_params: dict) -> ModelResponse:
        """Generic implementation in this class, override if needed.

        It implements model query by building message header and body with the help of Request Response Handler.

        Args:
            messages: inputs for inference
            model_params: model parameter json
            run_params: params for the run

        Returns:
            Response and http return code
        """
        ret_code = 200
        exception = None

        audio_array = messages[0].pop("array")  # We are not saving the audio array because it is too large
        sampling_rate = messages[0]["sampling_rate"]
        language = messages[0].get("language", "en")
        audio_file_path = messages[0].get("path")

        fp = None
        if not self.is_path_supported(audio_file_path):
            fp = tempfile.NamedTemporaryFile(suffix=".wav")
            sf.write(fp, audio_array, sampling_rate)
            audio_file_path = fp.name

        headers = {"Authorization": self.api_key}

        payload = {"language": LANGUAGE_MAP[language]}

        file_extension = audio_file_path.rsplit(".")[-1]

        files = [("file", (f"audio.{file_extension}", open(audio_file_path, "rb"), f"audio/{file_extension}"))]
        try:
            response_text = ""
            for attempt in range(MAX_RETRIES):
                response = requests.request("POST", self.model_url, headers=headers, data=payload, files=files)
                json_response = json.loads(response.text)
                # nvidia output is a list of dicts, we only want the text
                response_text = json_response.get("text", "").strip()
                if response_text:
                    break
                if attempt == MAX_RETRIES - 1:
                    logging.warning("No transcript found after second attempt")
                    logging.warning(f"Response: {response}")
            if fp:
                fp.close()
            return ModelResponse(
                input_prompt=messages,
                llm_response=response_text,
                response_code=200,
                raw_response=response_text,
                performance=None,
                model_parameters=model_params,
            )
        except Exception as e:
            logging.warning(f"Nvidia API error: {e}")
            ret_code = 500
            exception = str(e)

        response_text = {"response": "Nvidia error", "code": ret_code, "message": exception}
        return ModelResponse(
            input_prompt=messages,
            llm_response=json.dumps(response_text),
            raw_response=json.dumps(response_text),
            response_code=ret_code,
            performance=None,
            model_parameters=model_params,
        )

    def test(self) -> bool:
        """Call the model to test if it is available."""
        # Create dummy audio file
        resp_text, resp_code = self.generate_text(
            [{"array": np.random.rand(1000), "sampling_rate": 16000}], {}, {}, call_id="Test"
        )
        return resp_code == 200
