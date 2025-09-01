"""BigBench Audio preprocessor module for LALMEval framework.

This module provides a preprocessor for the BigBenchAudio dataset, designed for
Speech Query Question Answering (SQQA) tasks with audio processing capabilities.
"""

import logging
from typing import Dict, List, Any

import numpy as np
from tqdm import tqdm

from preprocessors.base import Preprocessor

logger = logging.getLogger(__name__)


class BigBenchAudioPreprocessor(Preprocessor):
    """
    A preprocessor for the BigBenchAudio dataset, designed for
    Speech Query Question Answering (SQQA) tasks.
    """

    def process(self, dataset: Dict[str, List[Any]], task_config: Dict[str, Any], 
                run_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process the BigBenchAudio dataset to ensure consistent audio format and structured data.

        Parameters:
        - dataset (Dict[str, List[Any]])
            Expected keys: 'id', 'audio', 'category', 'official_answer', 'transcript'.
        - task_config: Dictionary containing task configuration parameters
        - run_config: Dictionary containing run configuration parameters
            
        Returns:
        - List[Dict[str, Any]]: A list of dictionaries where each dictionary represents a sample,
          including the audio array resampled to 16kHz, metadata, and target label.
        """

        # Get dataset info
        dataset_keys = list(dataset.keys())
        dataset_size = len(dataset.get("id", []))
        self.log_dataset_info(dataset_keys, dataset_size)

        # Get dataset filters
        length_filter, num_samples_filter = self.get_dataset_filters(run_config.get('filter', None), dataset_size)

        processed_data = []
        indices = range(dataset_size)
        total_duration = 0
        sample_count = 0

        # Extract relevant information from task_config
        modality = task_config.get('modality', 'audio')
        target_column_name = task_config.get('target_column', None)
        sample_instruction_column_name = task_config.get('instruction_column', None)
        user_prompt = task_config.get('user_prompt', '')

        filters = run_config.get('filter', None)
        length_filter = None
        if (filters):
            length_filter = filters.get('length', None)

        if (not target_column_name):
            raise ValueError("[_big_bench_audio_preprocessor_] Target column name is missing. Preprocessing needs reference answers. Aborting!")

        for i in tqdm(indices, desc="Processing samples"):
            # Create record by accessing each feature by index
            sample_id = dataset["id"][i]

            # Ensure official answer exists. If not, skip!
            if not dataset[target_column_name][i]:
                logger.warning("[%d] Missing official answer. Skipping sample.", i)
                continue

            model_target = dataset[target_column_name][i].strip()
            audio_content_in_text = dataset[sample_instruction_column_name][i].strip()
            if modality == "text":
                audio_data = {
                    "array": np.array([]),  # Placeholder, not used in text-only evals
                    "sampling_rate": 16000
                }
                instruction = user_prompt + audio_content_in_text
            else:
                audio_data = dataset["audio"][i]

                # Validate audio data structure
                if not isinstance(audio_data, dict):
                    logger.warning(f"[{sample_id}] Invalid audio format. Skipping sample.")
                    continue

                # Convert to NumPy array
                audio_array = np.array(audio_data.get("array"))
                sr = audio_data.get("sampling_rate")

                if sr is None:
                    logger.warning(f"[{sample_id}] Sampling rate missing. Assuming 16kHz.")
                    sr = 16000

                # Use base class method to resample audio
                audio_array, sr = self.resample_audio(audio_array, sr)

                # Calculate audio duration in seconds
                audio_duration = len(audio_array) / sr
                total_duration += audio_duration

                # Apply length filtering if specified
                if (length_filter):
                    if not self.check_audio_length(audio_array, sr, length_filter):
                        continue
                if (num_samples_filter):
                    if sample_count >= num_samples_filter:
                        break
                
                # For audio modality, we can define a generic instruction
                # TODO: An override will need to pass this added instruction too. Consider a cleaner way to handle this.
                instruction = user_prompt + f"Answer the question provided in the audio."

            # Create structured sample
            sample = {
                "id": sample_id,
                "category": dataset["category"][i],
                "audio_content_in_text": audio_content_in_text,
                "array": audio_array if modality == "audio" else audio_data["array"],
                "sampling_rate": sr if modality == "audio" else audio_data["sampling_rate"],
                "model_target": model_target,
                "instruction": instruction,
            }

            processed_data.append(sample)
            sample_count += 1

        self.log_dataset_info(dataset_keys, dataset_size, sample_count, total_duration)
        return processed_data
