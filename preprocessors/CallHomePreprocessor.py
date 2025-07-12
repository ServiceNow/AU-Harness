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
            properties: dict, optional. May include 'length_filter' tuple (min_seconds, max_seconds) to filter samples.
        Returns:
            input_data: list of processed dicts
        """
        if properties is None:
            properties = {}
        metric = properties.get("metric", None)
        length_filter = properties.get("length_filter", None)  # Optional (min_seconds, max_seconds) tuple

        # Set base instruction depending on metric
        if metric and metric.lower() == "word_error_rate":
            base_instruction = (
                "Transcribe the text verbatim. Do not include any extra text or formatting. Include EVERY word."
            )
        else:
            base_instruction = (
                "Transcribe this turn based two speaker audio. There may be speaker overlap, and a speaker may go twice. "
                "You should return the output as \nA: words\nB: more words\nA: even more words\n"
                "you MUST return in this exact format, or the answer is considered invalid"
                "A is the first speaker, so always start with 'A:'"
            )

        dataset_path = Path(dataset).resolve()
        transcripts_dir = dataset_path / "transcripts"
        audio_dir = dataset_path / "audio"
        if not transcripts_dir.exists() or not audio_dir.exists():
            logger.error("CallHome dataset must contain 'transcripts' and 'audio' sub-folders")
            raise FileNotFoundError("CallHome dataset must contain 'transcripts' and 'audio' sub-folders")

        input_data = []
        cha_files = sorted(transcripts_dir.glob("*.cha"))
        if not cha_files:
            logger.warning(f"No .cha files found in {transcripts_dir}. Check your dataset path and contents.")
        if num_samples is not None:
            cha_files = cha_files[:num_samples]
        for cha_path in tqdm(cha_files, desc="Processing CallHome"):
            result = process_one(cha_path, audio_dir, base_instruction, metric, length_filter)
            if result is not None:
                if metric and metric.lower() == "word_error_rate" and isinstance(result, list):
                    input_data.extend(result)  # flatten
                else:
                    input_data.append(result)
        logger.info(f"[CallHomePreprocessor] Total samples processed: {len(input_data)}")
        return input_data


def process_one(cha_path, audio_dir, base_instruction, metric=None, length_filter=None):
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
            # Remove any word immediately following an = sign (e.g., =laughs, =lipsmack)
            text = re.sub(r'=\w+', '', text)
            text = re.sub(r'&\w+', '', text)
            text = ts_re.sub('', text)
            text = re.sub(r'\s+', ' ', text).strip()
            # Remove weird punctuation/annotation tokens
            text = re.sub(r'(?:[&+\-/]+[.,]*)+', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            # Remove if only punctuation or whitespace remains
            if not text or not re.search(r'\w', text) or all(c in '.,;:!?-()[]{}"\'\s' for c in text):
                continue
            cleaned_lines.append((orig_spkr, text, start_ms, end_ms))

    if min_start_ms is None or max_end_ms is None or not cleaned_lines:
        return None

    logger.info("Contents of cleaned_lines:")
    for line in cleaned_lines:
        logger.info(line)
    # Special handling for word error rate metric
    if metric and metric.lower() == "word_error_rate":
        # For each cleaned line, create a separate sample
        samples = []
        filtered_count = 0
        for idx, (orig_spkr, text, start_ms, end_ms) in enumerate(cleaned_lines, 1):
            # Extract audio segment for this line
            start_sample = int(start_ms / 1000 * sr)
            end_sample = int(end_ms / 1000 * sr)
            audio_segment = audio_array[start_sample:end_sample]
            
            # Calculate audio duration in seconds
            duration_seconds = len(audio_segment) / sr
            
            # Apply length filtering if specified
            if length_filter and isinstance(length_filter, tuple) and len(length_filter) == 2:
                min_length, max_length = length_filter
                if duration_seconds < min_length or duration_seconds > max_length:
                    filtered_count += 1
                    continue
            
            sample_id = f"{cha_id}_{idx}"
            samples.append({
                "array": audio_segment,
                "sampling_rate": sr,
                "instruction": base_instruction,
                # No chunk instructions for this case
                "chunk_instructions": [],
                "model_target": text,
                "id": sample_id,
                "task_type": "Turn-based WER",
            })
        if filtered_count > 0:
            logger.info(f"Filtered out {filtered_count} samples that didn't meet length criteria {length_filter}")
        if not samples:
            # All samples were filtered out
            return None
        return samples

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
    
    # Apply length filtering if specified
    audio_duration = len(audio_array) / sr
    if length_filter and isinstance(length_filter, tuple) and len(length_filter) == 2:
        min_length, max_length = length_filter
        if audio_duration < min_length or audio_duration > max_length:
            logger.info(f"Filtered out sample {cha_id} with duration {audio_duration:.2f}s (filter: {length_filter})")
            return None

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
        "task_type": "Turn Transcription",
    }