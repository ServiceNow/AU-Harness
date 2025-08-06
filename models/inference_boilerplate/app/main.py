from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import time
import uuid
import logging

from models import ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChoice, ChatMessage
from infer import infer
from utils import count_tokens, extract_audio_urls

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("inference-server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    logger.info(f"Received request: model={request.model}, messages={len(request.messages)}")

    audio_urls = extract_audio_urls(request.messages)
    response_text = await infer(
        model=request.model,
        messages=request.messages,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        audio_inputs=audio_urls,
        stop=request.stop,
    )

    prompt_tokens = count_tokens(request.messages)
    completion_tokens = count_tokens(response_text)
    total_tokens = prompt_tokens + completion_tokens

    response = ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=response_text),
                finish_reason="stop"
            )
        ],
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        }
    )

    return JSONResponse(content=response.dict())
