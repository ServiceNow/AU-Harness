import logging
import re
import random
import soundfile as sf
import numpy as np
from tqdm import tqdm
from scipy.signal import resample

from pathlib import Path

logger = logging.getLogger(__name__)

class CallHomePreprocessor:
    def process(self, dataset, properties=None):
        """
        Processes a dataset for CallHome speaker diarization.
        Args:
            dataset: str | Path. Path pointing to the root `CallHome_eng` folder that contains `audio/` and `transcripts/` sub-folders
            properties: dict, optional
        Returns:
            input_data: list of processed dicts
        """
        if properties is None:
            properties = {}

        dataset_path = Path(dataset).resolve()
        transcripts_dir = dataset_path / "transcripts"
        audio_dir = dataset_path / "audio"
        if not transcripts_dir.exists() or not audio_dir.exists():
            logger.error("CallHome dataset must contain 'transcripts' and 'audio' sub-folders")
            raise FileNotFoundError("CallHome dataset must contain 'transcripts' and 'audio' sub-folders")

        base_instruction = (
            "Transcribe this turn based two speaker audio. There may be speaker overlap, and a speaker may go twice. "
            "You should return the output as \n```json\n{\nA: words\nB: more words\nA: even more words\n}\n```\n\n"
            "you MUST return in this exact json format, or the answer is considered invalid"
        )

        input_data = []
        cha_files = sorted(transcripts_dir.glob("*.cha"))
        if not cha_files:
            logger.warning(f"No .cha files found in {transcripts_dir}. Check your dataset path and contents.")
        for cha_path in tqdm(cha_files, desc="Processing CallHome"):
            result = process_one(cha_path, audio_dir, base_instruction)
            if result is not None:
                input_data.append(result)
        logger.info(f"[CallHomePreprocessor] Total samples processed: {len(input_data)}")
        return input_data


def process_one(cha_path, audio_dir, base_instruction):
    logger.info(f"Processing {cha_path}")
    line_re = re.compile(r"^(?P<spkr>[A-Z]+):\s*(?P<txt>.*)$")
    cha_id = cha_path.stem  # e.g. 0638
    wav_path = audio_dir / f"{cha_id}.wav"
    if not wav_path.exists():
        return None
    audio_array, sr = sf.read(wav_path)
    target_sr = 16000
    if sr != target_sr:
        num_samples = int(round(audio_array.shape[0] * target_sr / sr))
        audio_array = resample(audio_array, num_samples)
        sr = target_sr
    transcript_lines = []
    with cha_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = line_re.match(line)
            if not m:
                continue  # skip meta/comment lines
            speaker = m.group("spkr")
            text = m.group("txt").strip()
            text = re.sub(r'&=\w+', '', text)
            transcript_lines.append(f"{speaker}: {text}")
            if not text:
                continue
            transcript_lines.append(f"{speaker}: {text}")
    reference_txt = "\n".join(transcript_lines)
    return {
        "array": audio_array,
        "sampling_rate": sr,
        "instruction": base_instruction,
        "model_target": reference_txt,
        "id": cha_id,
        "task_type": "Transcription",
    }


if __name__ == "__main__":
    preprocessor = CallHomePreprocessor()
    preprocessor.process("../private_datasets/CallHome_eng")