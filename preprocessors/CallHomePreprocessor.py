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
    def process(self, dataset, num_samples=None, properties=None):
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
            "You should return the output as \nA: words\nB: more words\nA: even more words\n"
            "you MUST return in this exact format, or the answer is considered invalid"
        )

        input_data = []
        cha_files = sorted(transcripts_dir.glob("*.cha"))
        if not cha_files:
            logger.warning(f"No .cha files found in {transcripts_dir}. Check your dataset path and contents.")
        if num_samples is not None:
            cha_files = cha_files[:num_samples]
        for cha_path in tqdm(cha_files, desc="Processing CallHome"):
            result = process_one(cha_path, audio_dir, base_instruction)
            if result is not None:
                input_data.append(result)
        logger.info(f"[CallHomePreprocessor] Total samples processed: {len(input_data)}")
        return input_data


def process_one(cha_path, audio_dir, base_instruction):
    logger.info(f"Processing {cha_path}")
    # Allow optional leading '*' and whitespace before the speaker code
    line_re = re.compile(r"^\s*\*?(?P<spkr>[A-Z]+):\s*(?P<txt>.*)$")
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
            logger.info(f"[CallHomePreprocessor] Processing line: {line}")
            m = line_re.match(line)
            logger.info(f"[CallHomePreprocessor] Match result: {m}")
            if not m:
                logger.info(f"[CallHomePreprocessor] Skipping line: {line}")
                continue  # skip meta/comment lines
            speaker = m.group("spkr")
            text = m.group("txt").strip()
            # Remove &=annotation codes
            text = re.sub(r'&=\w+', '', text)
            # Remove \x15timestamp\x15 markers like \x15249470_250380\x15
            text = re.sub(r'\x15[0-9_]+\x15', '', text)
            # Collapse extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            logger.info(f"[CallHomePreprocessor] Speaker: {speaker}, Text: {text}")
            transcript_lines.append(f"{speaker}: {text}")
            if not text:
                logger.info(f"[CallHomePreprocessor] Skipping empty text for speaker: {speaker}")
                continue
            logger.info(f"[CallHomePreprocessor] Added line: {speaker}: {text}")
            transcript_lines.append(f"{speaker}: {text}")
    reference_txt = "\n".join(transcript_lines)
    logger.info(f"[CallHomePreprocessor] Reference firsr 100 characters: {reference_txt[:100]}")
    #logger.info(f"[CallHomePreprocessor] Reference text: {reference_txt}")
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