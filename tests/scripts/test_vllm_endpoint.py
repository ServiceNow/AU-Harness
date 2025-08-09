import base64
import json
import requests

# Configuration
MODEL = {
    'name': '<insert_vllm_model_name>',
    'url': '<insert_url>/v1/chat/completions',
}

INPUT_STRING = "Transcribe the audio clip without any additional commentary."
AUDIO_PATH = "./tests/files/sample.wav"
API_KEY = "<insert_api_key>"


def encode_audio_to_base64(file_path: str) -> str:
    """
    Reads an audio file and encodes it to a Base64 string.
    """
    with open(file_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode("utf-8")


def generate_text_with_requests(input_text: str, audio_base64: str, model_url: str, api_key: str):
    """
    Sends a POST request to the model API with the audio and input text, then prints the response.
    """
    payload = {
        "model": MODEL['name'],
        "temperature": 0.01,
        "max_tokens": 512,
        "seed": 0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_base64,
                            "format": "wav"
                        }
                    },
                    {
                        "type": "text",
                        "text": input_text
                    }
                ]
            }
        ]
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(model_url, headers=headers, json=payload)
        response.raise_for_status()
        content = response.json().get("choices")[0].get("message").get("content")
        print("Model Response:\n", content)
    except requests.exceptions.RequestException as e:
        print("Error:\n", e)


if __name__ == "__main__":
    audio_b64 = encode_audio_to_base64(AUDIO_PATH)
    generate_text_with_requests(INPUT_STRING, audio_b64, MODEL["url"], API_KEY)
