from typing import List, Optional, Union
from app.models import Message
from app.dummy_model import DummyModel
import asyncio

model_instance = DummyModel()

async def infer(
    model: str,
    messages: List[Message],
    temperature: float,
    max_tokens: int,
    audio_inputs: List[str],
    stop: Optional[Union[str, List[str]]] = None
) -> str:
    await asyncio.sleep(0.05)  # Simulate latency
    return await model_instance.generate(messages, audio_inputs=audio_inputs)
