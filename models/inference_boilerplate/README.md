# OpenAI-Compatible FastAPI Inference Server (with Audio Support)

This is a FastAPI boilerplate server that mimics OpenAI’s `/v1/chat/completions` API and supports structured messages including text and `audio_url` inputs (remote URLs or base64-encoded audio).
It can be used when your model does not yet have vLLM support ready.

## ✅ Features:
- OpenAI-style chat completions endpoint
- Support for `audio_url` blocks in messages
- Async, concurrency-safe, GPU-ready inference
- Docker-ready for deployment
- Dummy model you can easily replace

---

## 📁 Project Structure

```text
inference_server/
├── app/
│   ├── main.py          # FastAPI server logic
│   ├── models.py        # Request/response schemas
│   ├── dummy_model.py   # Placeholder dummy model
│   ├── infer.py         # Inference runner
│   └── utils.py         # Token counter and audio extractor
├── requirements.txt
├── Dockerfile
└── README.md
```

## 🔧 Example Client Usage (with `requests`)

### 1. Send a message with a remote audio URL

```python
import requests

url = "http://localhost:8000/v1/chat/completions"
headers = {"Content-Type": "application/json"}

payload = {
    "model": "my-model",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What's in this audio?"
                },
                {
                    "type": "audio_url",
                    "audio_url": {
                        "url": "http://example.com/audio.ogg"
                    }
                }
            ]
        }
    ],
    "max_tokens": 64
}

response = requests.post(url, json=payload, headers=headers)
print(response.json())
```

### 2. Send base64-encoded audio via data: URI
```
import base64
import requests

def encode_audio_base64(audio_url: str) -> str:
    response = requests.get(audio_url)
    response.raise_for_status()
    return base64.b64encode(response.content).decode("utf-8")

base64_audio = encode_audio_base64("http://example.com/audio.ogg")

payload = {
    "model": "my-model",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What do you hear?"
                },
                {
                    "type": "audio_url",
                    "audio_url": {
                        "url": f"data:audio/ogg;base64,{base64_audio}"
                    }
                }
            ]
        }
    ],
    "max_tokens": 64
}


response = requests.post("http://localhost:8000/v1/chat/completions", json=payload)
print(response.json())
```
## ▶️ Run Locally
Install dependencies:
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000

## 🐳 Docker
Build and run:
docker build -t inference-server .
docker run --gpus all -p 8000:8000 inference-server


## 🔄 Replace the Dummy Model
Open app/dummy_model.py and replace the logic inside generate(...) with your own model:
class DummyModel:
    async def generate(self, messages, audio_inputs):
        # Insert your model logic here
