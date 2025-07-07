# pip install transformers peft librosa

import transformers
import numpy as np
import librosa
import os
from datasets import load_dataset
import librosa

def data_load(path = "hf", hf_token="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", local_dir="./data/enterprise-audio"):
    """
    Load audio data. If path == 'hf', download from Hugging Face repo using the provided token to a local folder.
    Otherwise, treat path as a local file and load with librosa.
    Returns:
        - If path == 'hf': List of local file paths to downloaded audio files.
        - Else: tuple (audio, sr) from librosa.load
    """
    if path == 'hf':
        dataset = load_dataset(
            "ServiceNow-AI/enterprise-audio",
            split="rows_with_audio",
            token=hf_token,
            cache_dir=local_dir,
            verification_mode="no_checks"  # skip missing-split verification that fails for rows_with_transcript
        )
        # Show available columns in this split
        print("Columns in 'rows_with_audio' split:", dataset.column_names)
        # Extract local file paths from the returned split
        audio_paths = []
        for item in dataset:
            # The field name for audio may vary; try common ones
            if 'audio' in item and isinstance(item['audio'], dict) and 'path' in item['audio']:
                audio_paths.append(item['audio']['path'])
            elif 'file' in item:
                audio_paths.append(item['file'])
        return audio_paths

def infer_on_dataset(audio_paths):
    import transformers
    import librosa
    pipe = transformers.pipeline(model='fixie-ai/ultravox-v0_6-llama-3_1-8b', trust_remote_code=True)
    system_turn = [{
        "role": "system",
        "content": "You are a friendly and helpful character. You love to answer questions for people. You must provide a pure transcription, with no additional text or formatting based on the audio sample."
    }]
    for path in audio_paths:
        print(f"\n--- Inference for: {path} ---")
        audio, sr = librosa.load(path, sr=16000)
        result = pipe({
            'audio': audio,
            'turns': system_turn,
            'sampling_rate': sr
        }, max_new_tokens=30)
        print("Model output:", result)


def main():
    audio_paths = data_load()
    # For a quick test, just run on the first audio file
    infer_on_dataset(audio_paths[:1])