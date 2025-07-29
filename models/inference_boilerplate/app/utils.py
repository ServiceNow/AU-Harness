from typing import List, Union
from app.models import Message

def count_tokens(content: Union[str, List[Message]]) -> int:
    if isinstance(content, str):
        return len(content.split())
    elif isinstance(content, list):
        total = 0
        for m in content:
            if isinstance(m.content, str):
                total += len(m.content.split())
            elif isinstance(m.content, list):
                for block in m.content:
                    if block.type == "text" and block.text:
                        total += len(block.text.split())
        return total
    return 0

def extract_audio_urls(messages: List[Message]) -> List[str]:
    """Extracts all audio URLs or data URIs from the messages."""
    urls = []
    for m in messages:
        if isinstance(m.content, list):
            for block in m.content:
                if block.type == "audio_url" and block.audio_url and block.audio_url.url:
                    urls.append(block.audio_url.url)
    return urls
