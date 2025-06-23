import base64
from copy import deepcopy
from io import BytesIO

import soundfile as sf
from PIL import Image

TRUNCATION_SUFFIX = "... (truncated)"
TRUNCATION_LENGTH = 60



def extract_base64_type_and_data(item: dict) -> list[str]:
    """Extract the type and data from a base64 string."""
    for typ in ["image_url", "audio_url"]:
        if typ in item:
            data = item[typ]["url"].removeprefix("data:").split(";")
            data[1] = data[1].split(",")[1]
            return data
    raise ValueError("Invalid base64 datatype.")


def encode_audio_array_base64(audio_array, sampling_rate):
    """Encode audio array into base64 without creating a file."""
    buffer = BytesIO()
    sf.write(buffer, audio_array, sampling_rate, format="WAV")
    buffer.seek(0)
    audio_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:audio/wav;base64,{audio_b64}"



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
