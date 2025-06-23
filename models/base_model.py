from __future__ import annotations
from abc import ABC, abstractmethod
from io import BytesIO
import soundfile as sf

class BaseModel(ABC):
    name: str = "base"

    @staticmethod
    def wav_bytes(wave, sr):
        buf = BytesIO()
        sf.write(buf, wave, sr, format="RAW", subtype="PCM_16")
        return buf.getvalue()

    def generate(self, examples: list[dict]):
        return [self._single(ex) for ex in examples]

    @abstractmethod
    def _single(self, ex: dict) -> str:  # pragma: no cover
        ...
