"""Callhome preprocessor module for LALMEval framework.

This module provides a preprocessor for the Callhome dataset, designed for
speaker diarization tasks with support for both audio and text modalities.
"""
import logging
import random
from pathlib import Path
from typing import Dict, List, Any

import soundfile as sf
import numpy as np
from scipy.signal import resample
from tqdm import tqdm
from datasets import Dataset

from preprocessors.base import Preprocessor

logger = logging.getLogger(__name__)


class CallhomePreprocessor(Preprocessor):
    """Preprocessor for Callhome dataset for speaker diarization tasks.
    
    This preprocessor handles the Callhome dataset which contains conversational
    audio with speaker diarization annotations. It supports both turn-based
    transcription and word error rate evaluation modes.
    """
    def process_asr(self, cleaned_lines, audio_array, sr, conv_id,
                                        instruction, length_filter, num_turns_per_conversation):
        """
        Process data specifically for word error rate metric.
        
        Args:
            cleaned_lines: List of cleaned lines (orig_spkr, text, start_ms, end_ms)
            audio_array: Audio array data
            sr: Sample rate
            conv_id: Conversaiton ID
            instruction: Instruction text
            length_filter: Optional tuple of (min_length, max_length) for filtering
            num_turns_per_conversation: Number of turns per conversation
            
        Returns:
            List of sample dictionaries or None if all samples were filtered
        """
        samples = []
        sample_count = 0
        seed = 42
        total_duration = 0

        # Shuffle the turns
        random.seed(seed)
        random.shuffle(cleaned_lines)

        for idx, cleaned_line in enumerate(cleaned_lines):
            text = cleaned_line['text']
            start_ms = cleaned_line['start_ms']
            end_ms = cleaned_line['end_ms']
            # Extract audio segment for this line
            start_sample = int(start_ms / 1000 * sr)
            end_sample = int(end_ms / 1000 * sr)
            audio_segment = audio_array[start_sample:end_sample]

            # Apply length filtering if specified
            if  (length_filter):
                if not self.check_audio_length(audio_array, sr, length_filter):
                    continue
            
            if sample_count > num_turns_per_conversation:
                break
        
            # Calculate audio duration in seconds
            duration_seconds = len(audio_segment) / sr
            total_duration+= duration_seconds

            sample_id = f"{conv_id}_{idx}"

            sample_dict = {
                "id": sample_id,
                "array": audio_segment,
                "sampling_rate": sr,
                "instruction": instruction,
                "model_target": text,
            }
            samples.append(sample_dict)
            sample_count += 1

        return samples, total_duration

    def load_audio(self, audio_path):
        """
        Load and resample audio file
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Tuple of audio_array, sample_rate
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            return None

        audio_array, sr = sf.read(audio_path)
        if (audio_array.ndim > 1):
            audio_array = np.mean(audio_array, axis=1)
        
        target_sr = 16000
        if sr != target_sr:
            num_samples = int(round(audio_array.shape[0] * target_sr / sr))
            audio_array = resample(audio_array, num_samples)
            sr = target_sr

        return audio_array, sr

    def process_chunking(self, cleaned_lines, min_start_ms):
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

        for cleaned_line in cleaned_lines:
            orig_spkr = cleaned_line['speaker']
            text = cleaned_line['text']
            start_ms = cleaned_line['start_ms']
         
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

    def process(self, dataset: Dataset, task_config: Dict[str, Any], 
                run_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Processes CallHome dataset
        Args:
            dataset: The task dataset to pre-process
            task_config: Dictionary containing task configuration parameters
            run_config: Dictionary containing run configuration parameters
        Returns:
            List of dictionaries where each dictionary represents a pre-processed sample
        """
        
        # Get dataset info
        dataset_keys = list(dataset.features.keys())
        dataset_size = len(dataset)
        self.log_dataset_info(dataset_keys, dataset_size)

        # Get dataset filters
        length_filter, num_samples_filter = self.get_dataset_filters(run_config.get('filter', None), dataset_size)
        task_instruction_prompt = task_config.get("user_prompt", "")

        processed_data = []
        total_duration = 0
        sample_count = 0

        task_type = "asr" if "asr" in task_config.get("task_name") else "diarization"

        if task_type == "asr":
            num_conversations = dataset_size
            total_num_samples = num_samples_filter if num_samples_filter else 1500
            num_turns_per_conversation = max(1, total_num_samples // num_conversations)
            filtered_dataset_size = dataset_size if total_num_samples > dataset_size else total_num_samples
        else:
            filtered_dataset_size = num_samples_filter if num_samples_filter else dataset_size

        for row in tqdm(dataset, desc="Processing samples", total=filtered_dataset_size):
            turns = row['turns']
            audio_array, sr = self.load_audio(row['audio_path'])

            if task_type == "asr":
                processed_results, duration = self.process_asr(
                        turns,
                        audio_array, 
                        sr,
                        row['id'],
                        task_instruction_prompt,
                        length_filter,
                        num_turns_per_conversation
                )
                processed_data.extend(processed_results)
                total_duration += duration
                if len(processed_data) > total_num_samples:
                    break
            else:
                # Diarization
                min_start_ms = row['conversation_audio_start_time']
                max_end_ms = row['conversation_audio_end_time']
                transcript_lines, chunk_instructions = self.process_chunking(turns, min_start_ms)

                # Cut audio to [min_start_ms, max_end_ms]
                start_sample = int(min_start_ms / 1000 * sr)
                end_sample = int(max_end_ms / 1000 * sr)
                audio_array = audio_array[start_sample:end_sample]

                if (length_filter):
                    if not self.check_audio_length(audio_array, sr, length_filter):
                        continue
                if (num_samples_filter):
                    if sample_count > num_samples_filter:
                        break
                
                # Calculate audio duration in seconds
                audio_duration = len(audio_array) / sr
                total_duration += audio_duration
            
                # Convert transcript_lines to reference text (preserving speaker resets at each chunk)
                reference_txt = "\n".join(transcript_lines)

                result = {
                    "array": audio_array,
                    "sampling_rate": sr,
                    "instruction": task_instruction_prompt,
                    "chunk_instructions": chunk_instructions,
                    "model_target": reference_txt,
                    "id": row['id']
                }
                processed_data.append(result)
                sample_count+=1

        self.log_dataset_info(dataset_keys, dataset_size, sample_count, total_duration)    
        return processed_data