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
    transcript_lines = []
    current_chunk_idx = -1
    speaker_map = {}

    with cha_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = line_re.match(line)
            if not m:
                continue
            orig_spkr = m.group("spkr")
            raw_txt = m.group("txt")

            ts_match = ts_re.search(raw_txt)
            if ts_match:
                start_ms = int(ts_match.group("start"))
                end_ms = int(ts_match.group("end"))
                if min_start_ms is None or start_ms < min_start_ms:
                    min_start_ms = start_ms
                if max_end_ms is None or end_ms > max_end_ms:
                    max_end_ms = end_ms
                chunk_idx = start_ms // 30000
            else:
                chunk_idx = current_chunk_idx

            if chunk_idx != current_chunk_idx:
                current_chunk_idx = chunk_idx
                speaker_map = {}

            if orig_spkr in speaker_map:
                canon_spkr = speaker_map[orig_spkr]
            else:
                canon_spkr = "A" if not speaker_map else "B"
                speaker_map[orig_spkr] = canon_spkr

            text = raw_txt
            text = re.sub(r'&=\w+', '', text)
            text = ts_re.sub('', text)
            text = re.sub(r'\s+', ' ', text).strip()
            if not text:
                continue
            transcript_lines.append(f"{canon_spkr}: {text}")

    # --- NEW: Cut audio to [min_start_ms, max_end_ms] ---
    if min_start_ms is not None and max_end_ms is not None:
        start_sample = int(min_start_ms / 1000 * sr)
        end_sample = int(max_end_ms / 1000 * sr)
        audio_array = audio_array[start_sample:end_sample]

    reference_txt = "\n".join(transcript_lines)
    logger.info(f"[CallHomePreprocessor] Reference first 100 characters: {reference_txt[:100]}")
    return {
        "array": audio_array,
        "sampling_rate": sr,
        "instruction": base_instruction,
        "model_target": reference_txt,
        "id": cha_id,
        "task_type": "Transcription",
    }


if __name__ == "__main__":
    import textwrap, sys

    # If the script is called with an argument, treat it as a dataset path (original behaviour)
    if len(sys.argv) > 1:
        preprocessor = CallHomePreprocessor()
        preprocessor.process(sys.argv[1])
    else:
        # Demo run on hard-coded CHA snippet provided by the user
        demo_text = textwrap.dedent(r"""@UTF8
@PID:\t11312/t-00001005-1
@Begin
@Languages:\teng
@Participants:\tA Subject, B Subject
@ID:\teng|eng|A|||||Subject|||
@ID:\teng|eng|B|||||Subject|||
@Media:\t0638, audio
*A:\tRight . \x15200460_200690\x15
*B:\tand I . \x15201140_201610\x15
*B:\tI already got another &=yelling apartment . \x15201950_203290\x15
*B:\tfor when I move out &=bang . \x15203880_204930\x15
*A:\toh you did . \x15205210_205610\x15
*B:\tyeah . \x15205780_206050\x15
*A:\tWhere ? \x15206410_206720\x15
*B:\t&=lipsmack it's on the corner of Columbia and Cole . \x15206830_209640\x15
*A:\tuhhuh . \x15210130_210500\x15""")