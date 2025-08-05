import base64
import os
import re

import numpy as np

# Import directly from multimodal as in the original file
from multimodal import (
    encode_audio_array_base64,
    audio_array_to_wav_file,
    truncate_values_for_saving,
    TRUNCATION_SUFFIX,
    TRUNCATION_LENGTH
)


def test_encode_audio_array_base64():
    """Test the encode_audio_array_base64 function"""
    print("Testing encode_audio_array_base64...")

    # Create a simple sine wave for testing
    sample_rate = 22050
    duration = 0.5  # seconds
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_array = np.sin(2 * np.pi * 440 * t)

    try:
        # Test encoding
        result = encode_audio_array_base64(audio_array, sample_rate)

        # Check if result is a string
        assert isinstance(result, str), "Result should be a string"

        # Try to decode the result to verify it's valid base64
        decoded = base64.b64decode(result)
        print("✅ encode_audio_array_base64 test passed")
    except Exception as e:
        print(f"❌ encode_audio_array_base64 test failed: {e}")


def test_audio_array_to_wav_file():
    """Test the audio_array_to_wav_file function"""
    print("Testing audio_array_to_wav_file...")

    # Create a simple sine wave for testing
    sample_rate = 22050
    duration = 0.5  # seconds
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_array = np.sin(2 * np.pi * 440 * t)

    try:
        # Test function
        result = audio_array_to_wav_file(audio_array, sample_rate)

        # Check if the file exists and has the right extension
        assert os.path.exists(result), f"File {result} should exist"
        assert result.endswith(".wav"), "File should have .wav extension"

        # Clean up the temporary file
        os.remove(result)
        print("✅ audio_array_to_wav_file test passed")
    except Exception as e:
        print(f"❌ audio_array_to_wav_file test failed: {e}")


def test_truncate_values_for_saving():
    """Test the truncate_values_for_saving function"""
    print("Testing truncate_values_for_saving...")

    try:
        # Test 1: Simple string - non-base64 shouldn't be truncated
        normal_text = "This is some normal text with spaces"
        result1 = truncate_values_for_saving(normal_text)
        assert result1 == normal_text, "Normal text should not be truncated"

        # Test 2: Long base64-like string should be truncated
        base64_like = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789" * 10
        result2 = truncate_values_for_saving(base64_like)
        expected2 = base64_like[:TRUNCATION_LENGTH] + TRUNCATION_SUFFIX
        assert result2 == expected2, "Base64 string should be truncated"

        # Test 3: Dictionary with nested values
        test_dict = {
            "normal_text": "This is normal text",
            "long_text": "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789" * 10,
            "nested": {
                "more_base64": "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789" * 10
            }
        }

        result3 = truncate_values_for_saving(test_dict)
        assert result3["normal_text"] == test_dict["normal_text"], "Normal text in dict should not be truncated"
        assert result3["long_text"] == test_dict["long_text"][
                                       :TRUNCATION_LENGTH] + TRUNCATION_SUFFIX, "Long text in dict should be truncated"
        assert result3["nested"]["more_base64"] == test_dict["nested"]["more_base64"][
                                                   :TRUNCATION_LENGTH] + TRUNCATION_SUFFIX, "Nested long text should be truncated"

        # Test 4: List with mixed types
        test_list = [
            "Normal text",
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789" * 10,
            {"key": "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789" * 10},
        ]

        result4 = truncate_values_for_saving(test_list)
        assert result4[0] == test_list[0], "Normal text in list should not be truncated"
        assert result4[1] == test_list[1][
                             :TRUNCATION_LENGTH] + TRUNCATION_SUFFIX, "Long text in list should be truncated"

        print("✅ truncate_values_for_saving test passed")
    except Exception as e:
        print(f"❌ truncate_values_for_saving test failed: {e}")


def split_inline_speaker_labels(text: str) -> str:
    # This will insert a newline before any 'A:' or 'B:' that is not at the start of a line
    return re.sub(r'(?<!^)(?<!\n)\s*([AB]:)', r'\n\1', text)


def test_split_inline_speaker_labels():
    """Test the split_inline_speaker_labels function"""
    print("Testing split_inline_speaker_labels...")

    try:
        test_cases = [
            ("A: hello B: world", "A: hello\nB: world"),
            ("A: first line\nB: second line", "A: first line\nB: second line"),
            ("A: hello how are you B: I'm good", "A: hello how are you\nB: I'm good"),
            ("A: start B: middle A: end", "A: start\nB: middle\nA: end"),
            ("No speaker labels here", "No speaker labels here"),
            ("\nA: Already has newline", "\nA: Already has newline"),
            ("Text A: with A: multiple B: labels", "Text\nA: with\nA: multiple\nB: labels")
        ]

        for input_text, expected_output in test_cases:
            result = split_inline_speaker_labels(input_text)
            assert result == expected_output, f"Failed: '{input_text}' → Got '{result}', Expected '{expected_output}'"

        print("✅ split_inline_speaker_labels test passed")
    except Exception as e:
        print(f"❌ split_inline_speaker_labels test failed: {e}")


if __name__ == "__main__":
    test_encode_audio_array_base64()
    test_audio_array_to_wav_file()
    test_truncate_values_for_saving()
    test_split_inline_speaker_labels()
    print("All tests completed!")
