import base64
from io import BytesIO
import soundfile as sf
import librosa
import numpy as np
import json
import requests
from tqdm import tqdm
from datasets import load_from_disk
import os
import argparse
import concurrent.futures

class Inferencer:
    def __init__(self, url_path: str, model_name: str, generation_params: dict = None):
        self.url = url_path
        self.model_name = model_name
        self.generation_params = generation_params if generation_params is not None else {
        }
        self.headers = {'Content-Type': 'application/json'}

    def _encode_audio(self, audio_array: np.ndarray, sampling_rate: int) -> str:
        try:
            audio_array = librosa.resample(audio_array, orig_sr=sampling_rate, target_sr=16000)
            sampling_rate = 16000

            buffer = BytesIO()
            sf.write(buffer, audio_array, sampling_rate, format='WAV')
            buffer.seek(0)
            audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            return audio_base64
        except Exception as e:
            raise RuntimeError(f"Failed to encode audio: {e}")

    def infer_single(self, question: str, audio_array: np.ndarray, sampling_rate: int) -> str:
        try:
            audio_base64 = self._encode_audio(audio_array, sampling_rate)
        except Exception as e:
            return f"Error during audio encoding: {e}"

        chat_input = [
            {
                "role": "system",
                "content": "You are Apriel, a thoughtful and systematic AI assistant built by ServiceNow Language Models (SLAM) lab."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_base64,
                            "format": "wav"
                        }
                    }
                ]
            }
        ]

        payload = {
            "model": self.model_name,
            "messages": chat_input,
            "temperature": 0,
            "max_tokens": 8192,
            "seed": 0
        }

        try:
            response = requests.post(self.url, headers=self.headers, data=json.dumps(payload))
            response.raise_for_status()
            response_data = response.json()

            if 'choices' in response_data and len(response_data['choices']) > 0 and \
               'message' in response_data['choices'][0] and 'content' in response_data['choices'][0]['message']:
                return response_data['choices'][0]['message']['content'].strip()
            else:
                return f"Error: Unexpected response structure. Response: {response_data}"

        except requests.exceptions.RequestException as e:
            return f"HTTP Error: {e}"
        except json.JSONDecodeError:
            return f"Error: Invalid JSON response. Response text: {response.text if response else 'N/A'}"
        except Exception as e:
            return f"Unexpected Error: {e}"