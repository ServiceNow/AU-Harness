from typing import List, Optional
from app.models import Message

class DummyModel:
    def __init__(self):
        self.device = "cuda"

    async def generate(self, messages: List[Message], audio_inputs: List[str]) -> str:
        text_input = ""
        for m in messages:
            if isinstance(m.content, str):
                text_input += m.content + " "
            elif isinstance(m.content, list):
                for block in m.content:
                    if block.type == "text" and block.text:
                        text_input += block.text + " "
        summary = f"Processed text: '{text_input.strip()}'"
        if audio_inputs:
            summary += f" | Processed {len(audio_inputs)} audio input(s)"
        return summary
