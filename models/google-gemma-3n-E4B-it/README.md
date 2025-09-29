# Google Gemma 3n Model API

This folder creates a Docker image that contains the Google Gemma 3n model with audio processing capabilities.

## Overview

The Google Gemma 3n model is a multimodal model that can process both text and audio inputs. This implementation provides an OpenAI-compatible API for easy integration with existing applications.

## Requirements

- Docker
- Hugging Face account with access to the Gemma 3n model
- Hugging Face API token

## Building the Image

To build the Docker image, run:

```bash
docker build -t google-gemma-3n .
```

## Running the Container

To run the container:

```bash
docker run -p 8000:8000 \
  -e HF_TOKEN=your_huggingface_token \
  -e MODEL_PATH=google/gemma-3n-E4B-it \
  google-gemma-3n
```

## API Usage

The API follows the OpenAI chat completions format. You can send audio files as part of your messages.

### Example Request

```python
import requests
import base64

audio_url = "https://example.com/audio.mp3"

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "google/gemma-3n-E4B-it",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio_url",
                        "audio_url": {"url": audio_url}
                    },
                    {
                        "type": "text",
                        "text": "Transcribe this audio."
                    }
                ]
            }
        ],
        "temperature": 0.7,
        "max_tokens": 512
    }
)

print(response.json())
```
