import logging
import re
import logging
import soundfile as sf
from tqdm import tqdm
from scipy.signal import resample
from preprocessors.base import Preprocessor
from pathlib import Path

logger = logging.getLogger(__name__)

class CallhomePreprocessor(Preprocessor):
    def _load_audio(self, cha_path, audio_dir):
        """
        Load and resample audio file for the given CHA path.
        
        Args:
            cha_path: Path to the CHA file
            audio_dir: Directory containing audio files
            
        Returns:
            Tuple of (audio_array, sample_rate, cha_id) or None if audio file not found
        """
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
            
        return audio_array, sr, cha_id
    
    def load_and_build_prompts(self, base_instruction, user_prompt_add_ons=None, system_prompts=None):
        """
        Load prompt files and build instruction with prompt add-ons and system prompts.
        
        Args:
            base_instruction: Base instruction text
            user_prompt_add_ons: List of prompt add-ons to use
            system_prompts: List of system prompts to use
            
        Returns:
            Tuple of (instruction, system_prompt_text)
        """
        if user_prompt_add_ons is None:
            user_prompt_add_ons = []
        if system_prompts is None:
            system_prompts = []
            
        # Load YAML files using the base class method
        prompt_add_ons = self.load_yaml_file("prompt_add_ons.yaml")
        system_prompts_mapping = self.load_yaml_file("system_prompts.yaml")
        
        # Build instruction with prompt add-ons
        instruction = base_instruction
        for k in user_prompt_add_ons:
            add_on = prompt_add_ons.get(k)
            if add_on:
                instruction = f"{instruction} {add_on}"
                
        # Process system prompts
        system_prompt_text = ""
        for k in system_prompts:
            prompt = system_prompts_mapping.get(k)
            if prompt:
                if system_prompt_text:
                    system_prompt_text += "\n\n" + prompt
                else:
                    system_prompt_text = prompt
                    
        return instruction, system_prompt_text

    def _initial_parse_and_cleanup(self, cha_path):
        """
        Parse the CHA file and clean up the text.
        
        Args:
            cha_path: Path to the CHA file
            
        Returns:
            Tuple of (cleaned_lines, min_start_ms, max_end_ms) or None if no valid lines
        """
        # Allow optional leading '*' and whitespace before the speaker code
        # We capture the speaker code and the entire remainder of the line (which may contain a \x15<start>_<end>\x15 timestamp)
        line_re = re.compile(r"^\s*\*?(?P<spkr>[A-Z]+):\s*(?P<txt>.*)$")
        # Regex to capture timestamps of the form \x15<start>_<end>\x15 (numbers are in milliseconds)
        ts_re = re.compile(r"\x15(?P<start>[0-9]+)_(?P<end>[0-9]+)\x15")
        
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
                if not text or not re.search(r'\w', text) or all(c in '.,;:!?-()[]{}"\'\'\s' for c in text):
                    continue
                cleaned_lines.append((orig_spkr, text, start_ms, end_ms))

        if min_start_ms is None or max_end_ms is None or not cleaned_lines:
            return None
            
        return cleaned_lines, min_start_ms, max_end_ms
        
    def _process_word_error_rate_metric(self, cleaned_lines, audio_array, sr, cha_id, base_instruction, user_prompt_add_ons, system_prompts, length_filter):
        """
        Process data specifically for word error rate metric.
        
        Args:
            cleaned_lines: List of cleaned lines (orig_spkr, text, start_ms, end_ms)
            audio_array: Audio array data
            sr: Sample rate
            cha_id: CHA ID
            base_instruction: Base instruction text
            user_prompt_add_ons: List of prompt add-ons to use
            system_prompts: List of system prompts to use
            length_filter: Optional tuple of (min_length, max_length) for filtering
            
        Returns:
            List of sample dictionaries or None if all samples were filtered
        """
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
            
            instruction, system_prompt_text = self.load_and_build_prompts(
                base_instruction, user_prompt_add_ons, system_prompts
            )
            
            sample_dict = {
                "array": audio_segment,
                "sampling_rate": sr,
                "instruction": instruction,
                # No chunk instructions for this case
                "chunk_instructions": [],
                "model_target": text,
                "id": sample_id,
                "task_type": "Turn-based WER"
            }
            
            if system_prompt_text:
                sample_dict["system_prompt"] = system_prompt_text
                
            samples.append(sample_dict)
        
        if filtered_count > 0:
            logger.info(f"Filtered out {filtered_count} samples that didn't meet length criteria {length_filter}")
        if not samples:
            # All samples were filtered out
            return None
        return samples
    
    def _process_chunking(self, cleaned_lines, min_start_ms):
        """
        Process chunking and create transcript lines with canonical speaker labels.
        
        Args:
            cleaned_lines: List of cleaned lines (orig_spkr, text, start_ms, end_ms)
            min_start_ms: Minimum start timestamp
            
        Returns:
            Tuple of (transcript_lines, chunk_instructions)
        """
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
            
        return transcript_lines, chunk_instructions

    def _process_one(self, cha_path, audio_dir, base_instruction, metric=None, length_filter=None, user_prompt_add_ons=None, system_prompts=None):
        """
        Process one conversation file.
        
        Args:
            cha_path: Path to the CHA file
            audio_dir: Directory containing audio files
            base_instruction: Base instruction text
            metric: Metric to use for processing
            length_filter: Optional tuple of (min_length, max_length) for filtering
            user_prompt_add_ons: List of prompt add-ons to use
            system_prompts: List of system prompts to use
            
        Returns:
            Dictionary with processed data or None if processing failed
        """
        logger.info(f"Processing {cha_path}")
        
        # Step 1: Load audio file
        audio_result = self._load_audio(cha_path, audio_dir)
        if audio_result is None:
            return None
        audio_array, sr, cha_id = audio_result
        
        # Step 2: Initial parse and cleanup
        parse_result = self._initial_parse_and_cleanup(cha_path)
        if parse_result is None:
            return None
        cleaned_lines, min_start_ms, max_end_ms = parse_result

        # Step 3: Special handling for word error rate metric
        if metric and metric.lower() == "word_error_rate":
            return self._process_word_error_rate_metric(
                cleaned_lines, audio_array, sr, cha_id, base_instruction,
                user_prompt_add_ons, system_prompts, length_filter
            )

        # Step 4: Process chunking
        transcript_lines, chunk_instructions = self._process_chunking(cleaned_lines, min_start_ms)
        
        # Cut audio to [min_start_ms, max_end_ms]
        start_sample = int(min_start_ms / 1000 * sr)
        end_sample = int(max_end_ms / 1000 * sr)
        audio_array = audio_array[start_sample:end_sample]
        
        # Apply length filtering if specified
        audio_duration = len(audio_array) / sr
        if length_filter and isinstance(length_filter, tuple) and len(length_filter) == 2:
            min_length, max_length = length_filter
            if audio_duration < min_length or audio_duration > max_length:
                return None
                
        # Step 5: Convert transcript_lines to reference text (preserving speaker resets at each chunk)
        reference_txt = "\n".join(transcript_lines)
        
        # Step 6: Load and build prompts in one step
        instruction, system_prompt_text = self.load_and_build_prompts(
            base_instruction, user_prompt_add_ons, system_prompts
        )
        
        result = {
            "array": audio_array,
            "sampling_rate": sr,
            "instruction": instruction,
            "chunk_instructions": chunk_instructions,
            "model_target": reference_txt,
            "id": cha_id,
            "task_type": "Turn Transcription",
        }
        
        if system_prompt_text:
            result["system_prompt"] = system_prompt_text
            
        return result

    def process(self, dataset, num_samples=None, properties=None):
        """
        Processes a dataset for CallHome speaker diarization.
        Args:
            dataset: str | Path. Path pointing to the root `CallHome_eng` folder that contains `audio/` and `transcripts/` sub-folders
            properties: dict, optional. May include 'length_filter' tuple (min_seconds, max_seconds) to filter samples.
        Returns:
            input_data: list of processed dicts
        """
        # Extract common properties using base class method
        props = self.extract_properties(properties)
        metric = props["metric"]
        user_prompt_add_ons = props["user_prompt_add_ons"]
        system_prompts = props["system_prompts"]
        length_filter = props["length_filter"]

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
        logger.info(f"[CallhomePreprocessor] Resolved dataset path: {dataset_path}")
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
            result = self._process_one(cha_path, audio_dir, base_instruction, metric, length_filter, user_prompt_add_ons, system_prompts)
            if result is not None:
                if metric and metric.lower() == "word_error_rate" and isinstance(result, list):
                    input_data.extend(result)  # flatten
                else:
                    input_data.append(result)
        logger.info(f"[CallHomePreprocessor] Total samples processed: {len(input_data)}")
        return input_data