import base64
from copy import deepcopy
from io import BytesIO
import librosa
import soundfile as sf

TRUNCATION_SUFFIX = "... (truncated)"
TRUNCATION_LENGTH = 60

#normalize to 16000
def encode_audio_array_base64(audio_array, sampling_rate):
    try:
        if audio_array is None or len(audio_array) == 0:
            return ""
        else:
            audio_array = librosa.resample(audio_array, orig_sr=sampling_rate, target_sr=16000)
            sampling_rate = 16000

            buffer = BytesIO()
            sf.write(buffer, audio_array, sampling_rate, format='WAV')
            buffer.seek(0)
            audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            return audio_base64
    except Exception as e:
        raise RuntimeError(f"Failed to encode audio: {e}")

def audio_array_to_wav_file(audio_array, sampling_rate):
    """Resample to 16kHz and write audio_array to a temporary .wav file. Returns file path."""
    import tempfile
    try:
        audio_array = librosa.resample(audio_array, orig_sr=sampling_rate, target_sr=16000)
        sampling_rate = 16000
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
            sf.write(tmp_wav, audio_array, sampling_rate, format='WAV')
            wav_path = tmp_wav.name
        return wav_path
    except Exception as e:
        raise RuntimeError(f"Failed to write audio to wav file: {e}")

def truncate_values_for_saving(formatted_messages: list[dict] | str) -> list[dict] | str:
    """Recursively truncate all string fields in formatted messages.

    This function recursively traverses the input data structure and truncates any string
    values that exceed the specified length. It handles nested dictionaries, lists, and
    other data structures.

    Args:
        formatted_messages: The messages to process, can be a list of dictionaries or a string.

    Returns:
        The processed messages with truncated string values.
    """
    # Make a deep copy only at the top level
    truncated_messages = deepcopy(formatted_messages)

    def _truncate_recursive(obj):
        """Recursively truncate strings in the object."""
        if isinstance(obj, str) and len(obj) > TRUNCATION_LENGTH + len(TRUNCATION_SUFFIX):
            if (
                any(c.isspace() or (not c.isascii()) for c in obj) or len(obj) < 256
            ):  # natural text will either have whitespace or unicode chars (if foreign langs)
                return obj
            return (
                obj[:TRUNCATION_LENGTH] + TRUNCATION_SUFFIX
            )  # will only truncate base64 type strings (has 0 whitespace/unicode) with length > 256

        elif isinstance(obj, dict):
            for key, value in obj.items():
                obj[key] = _truncate_recursive(value)

        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                obj[i] = _truncate_recursive(item)

        # Handle bytes objects
        elif isinstance(obj, bytes) and len(obj) > TRUNCATION_LENGTH + len(TRUNCATION_SUFFIX):
            # Convert to string representation for truncation
            return str(obj)[:TRUNCATION_LENGTH] + TRUNCATION_SUFFIX

        # Return unchanged for other types (int, float, bool, None, etc.)
        return obj

    # Start the recursion with our copy
    return _truncate_recursive(truncated_messages)
