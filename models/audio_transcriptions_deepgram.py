import logging
import tempfile

import numpy as np
import soundfile as sf
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    FileSource,
    PrerecordedOptions,
)

from models.model import Model
from models.model_response import ModelResponse
from utils import constants

# These are listsed on their page, but more are supported
MAX_RETRIES = 2
SUPPORTED_SOUND_FORMAT = [
    "mp3",
    "mp4",
    "mp2",
    "AAC",
    "FLAC",
    "PCM",
    "Ogg",
    "Opus",
    "WebM",
    "mpeg",
    "mpga",
    "m4a",
    "wav",
    "webm",
]


class AudioTranscriptionsDeepgram(Model):
    """Deepgram Audio model class."""

    def name(self):
        """Return the name of the model."""
        return constants.MODEL_TRANSCRIPTIONS_DEEPGRAM

    def __init__(self, model_info):
        super().__init__(model_info)
        self.model = model_info["model"]
        self.api_key = model_info["auth_token"]
        # The Deepgram SDK is used, no model URL required (see pyproject.toml): use URL 'ignore' in toolkit

    @staticmethod
    def is_path_supported(audio_file_path) -> bool:
        """Check if the audio file path can directly be used with the OpenAI API."""
        return (
            audio_file_path
            and any(audio_file_path.endswith(sound_format) for sound_format in SUPPORTED_SOUND_FORMAT)
            and audio_file_path.startswith("/tmp")
        )

    def _extract_transcript(self, response: dict) -> str:
        """Safely extract transcript from Deepgram response.

        Args:
            response: Raw response from Deepgram API

        Returns:
            Extracted transcript or empty string if not found
        """
        try:
            return response["results"]["channels"][0]["alternatives"][0]["transcript"]
        except (KeyError, IndexError):
            return ""

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
        audio_array = messages[0].pop("array")  # We are not saving the audio array because it is too large
        sampling_rate = messages[0]["sampling_rate"]
        language = messages[0].get("language", "en")
        if self.model == "nova-3":
            language = "multi"
        audio_file_path = messages[0].get("path")

        fp = None
        if not self.is_path_supported(audio_file_path):
            fp = tempfile.NamedTemporaryFile(suffix=".wav")
            sf.write(fp, audio_array, sampling_rate)
            audio_file_path = fp.name

        config = DeepgramClientOptions()
        deepgram = DeepgramClient(self.api_key, config)

        with open(audio_file_path, "rb") as file:
            buffer_data = file.read()

        payload: FileSource = {
            "buffer": buffer_data,
        }

        options = PrerecordedOptions(model=self.model, smart_format=True, language=language)

        try:
            for attempt in range(MAX_RETRIES):
                response = deepgram.listen.rest.v("1").transcribe_file(payload, options, timeout=60)
                transcript = self._extract_transcript(response)
                if transcript:
                    break
                if attempt == MAX_RETRIES - 1:
                    logging.warning("No transcript found after second attempt")
                    logging.warning(f"Response: {response}")
            if fp:
                fp.close()
        except Exception as e:
            logging.error(f"Deepgram transcription failed: {str(e)}", exc_info=True)
            return ModelResponse(
                input_prompt=messages,
                llm_response="Deepgram error",
                raw_response=str(e),
                response_code=500,
                performance=None,
                model_parameters=options.to_dict(),
            )
        return ModelResponse(
            input_prompt=messages,
            llm_response=transcript,
            raw_response=transcript,
            response_code=200,
            performance=None,
            model_parameters=options.to_dict(),
        )

    def test(self) -> bool:
        """Call the model to test if it is available."""
        # Create dummy audio file
        resp_text, resp_code = self.generate_text(
            [{"array": np.random.rand(1000), "sampling_rate": 16000}], {}, {}, call_id="Test"
        )
        return resp_code == 200
