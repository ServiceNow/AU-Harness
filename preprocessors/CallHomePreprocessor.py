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
            "A is the first speaker, so always start with 'A:'"
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
    # We capture the speaker code and the entire remainder of the line (which may contain a \x15<start>_<end>\x15 timestamp)
    line_re = re.compile(r"^\s*\*?(?P<spkr>[A-Z]+):\s*(?P<txt>.*)$")
    # Regex to capture timestamps of the form \x15<start>_<end>\x15 (numbers are in milliseconds)
    ts_re = re.compile(r"\x15(?P<start>[0-9]+)_(?P<end>[0-9]+)\x15")
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

    # --- NEW: Find min start and max end timestamps ---
    min_start_ms = None
    max_end_ms = None
    cleaned_lines = []  # List of tuples: (orig_spkr, cleaned_text, start_ms, end_ms)

    # Filler word removal logic (copied from CallHomePostprocessor)
    import re as _re
    _FILLER_PATTERNS = [
        r"\buh\b", r"\bum+\b", r"\buhhuh\b", r"\bmhm+\b", r"\bmm+\b", r"\bah+\b", r"\beh+\b", r"\bhmm+\b",
        r"\bh\b", r"\bye\b", r"\byeah yeah\b", r"\bI I\b", r"\bx+\b", r"\bxxx\b",
        r"\bca-\b", r"\be-\b", r"\bI-\b", r"\bm-\b", r"\bw-\b", r"\b\+/, \+\b", r"\b\+\,\b",
        r"\b(hm)+\b", r"\b(um)+\b", r"\b(uh)+\b"
    ]
    def _remove_fillers(text):
        for pat in _FILLER_PATTERNS:
            text = _re.sub(pat, '', text, flags=_re.IGNORECASE)
        text = _re.sub(r'\b[a-zA-Z]-\b', '', text)
        text = _re.sub(r'\s+', ' ', text).strip()
        return text

    # First pass: clean up all text and collect lines with timestamps
    with cha_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = line_re.match(line)
            if not m:
                continue
            orig_spkr = m.group("spkr")
            raw_txt = m.group("txt")
            ts_match = ts_re.search(raw_txt)
            if not ts_match:
                continue  # Only process lines with timestamps
            start_ms = int(ts_match.group("start"))
            end_ms = int(ts_match.group("end"))
            if min_start_ms is None or start_ms < min_start_ms:
                min_start_ms = start_ms
            if max_end_ms is None or end_ms > max_end_ms:
                max_end_ms = end_ms
            text = raw_txt
            text = re.sub(r'&=\w+', '', text)
            text = ts_re.sub('', text)
            text = re.sub(r'\s+', ' ', text).strip()
            text = _remove_fillers(text)
            # Remove if only punctuation or whitespace remains
            if not text or not re.search(r'\w', text) or all(c in '.,;:!?-()[]{}"\'\s' for c in text):
                continue
            cleaned_lines.append((orig_spkr, text, start_ms, end_ms))

    if min_start_ms is None or max_end_ms is None or not cleaned_lines:
        return None

    logger.info("Contents of cleaned_lines:")
    for line in cleaned_lines:
        logger.info(line)
    # Second pass: chunking and speaker resetting per chunk
    chunk_lines = {}  # chunk_idx: list of (orig_spkr, text)
    chunk_order = []
    for orig_spkr, text, start_ms, end_ms in cleaned_lines:
        chunk_idx = (start_ms - min_start_ms) // 30000  # relative to min_start_ms
        if chunk_idx not in chunk_lines:
            chunk_lines[chunk_idx] = []
            chunk_order.append(chunk_idx)
        chunk_lines[chunk_idx].append((orig_spkr, text))

    transcript_lines = []
    chunk_instructions = []
    for chunk_idx in chunk_order:
        spkr_map = {}
        lines = []
        for orig_spkr, text in chunk_lines[chunk_idx]:
            if orig_spkr in spkr_map:
                canon_spkr = spkr_map[orig_spkr]
            else:
                canon_spkr = "A" if not spkr_map else "B"
                spkr_map[orig_spkr] = canon_spkr
            line_str = f"{canon_spkr}: {text}"
            transcript_lines.append(line_str)
            if len(lines) < 2:
                lines.append(line_str)
        chunk_instructions.append("\n".join(lines))
    logger.info(f"Transcript lines: {transcript_lines}")
    # --- Cut audio to [min_start_ms, max_end_ms] ---
    start_sample = int(min_start_ms / 1000 * sr)
    end_sample = int(max_end_ms / 1000 * sr)
    audio_array = audio_array[start_sample:end_sample]

    # Build reference as a single two-line string: A: all A's words\nB: all B's words
    a_words = []
    b_words = []
    for line in transcript_lines:
        line = line.strip()
        if line.startswith('A:'):
            a_words.append(line[2:].strip())
        elif line.startswith('B:'):
            b_words.append(line[2:].strip())
    reference_txt = f"A: {' '.join(a_words)}\nB: {' '.join(b_words)}"
    logger.info(f"[CallHomePreprocessor] Reference first 100000 characters: {reference_txt[:100000]}")
    return {
        "array": audio_array,
        "sampling_rate": sr,
        "instruction": base_instruction,
        "chunk_instructions": chunk_instructions,
        "model_target": reference_txt,
        "id": cha_id,
        "task_type": "Transcription",
    }