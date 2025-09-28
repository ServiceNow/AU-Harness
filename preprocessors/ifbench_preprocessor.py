"""IFBench preprocessor module for evaluation framework.

This module provides a preprocessor for the IFBench datasets, designed for
instruction following evaluation of audio LLMs and more.
"""
import logging
from typing import Dict, List, Optional, Any

from datasets import Dataset
import numpy as np
from tqdm import tqdm

from preprocessors.base import Preprocessor

logger = logging.getLogger(__name__)


class IfbenchPreprocessor(Preprocessor):
    """
    A preprocessor for the IFBench datasets, designed for
    instruction following evaluation of audio LLMs.
    """

    def process(self, dataset: Dataset, task_config: Dict[str, Any], 
                run_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process the IFBench datasets.

        Args:
            dataset: The task dataset to pre-process
            task_config: Dictionary containing task configuration parameters
            run_config: Dictionary containing run configuration parameters
            
        Returns:
            List of dictionaries where each dictionary represents a pre-processed sample
        """

        modality = task_config.get('modality', 'audio')
        audio_column_name = task_config.get('audio_column', None)
        sample_instruction_column_name = task_config.get('instruction_column', None)

        # Obtain task-specific prompt (if provided)
        user_prompt = task_config.get('user_prompt', '')

        # Get dataset info
        dataset_keys = list(dataset.features.keys())
        dataset_size = len(dataset)
        self.log_dataset_info(dataset_keys, dataset_size)

        # Get dataset filters
        length_filter, num_samples_filter = self.get_dataset_filters(run_config.get('filter', None), dataset_size)

        processed_data = []
        total_duration = 0
        sample_count = 0

        for i, row in enumerate(tqdm(dataset, desc="Processing samples")):

            if num_samples_filter and sample_count >= num_samples_filter:
                break

            # Ensure prompt exists. Otherwise, move onto the next sample.
            prompt = row[sample_instruction_column_name]

            # Extract IFBench specific columns
            key = row['key']
            instruction_id_list = row["instruction_id_list"]
            kwargs = row["kwargs"]

            # Handle instruction for different modalities
            if modality == "text":
                instruction = user_prompt + prompt
                audio_data = {
                    "array": np.array([]),  # Placeholder, not used in text-only evals
                    "sampling_rate": 16000
                }
            else:
                # For audio modality, we can define a generic instruction
                instruction = user_prompt + "Answer the question provided in the audio."
                audio_data = row[audio_column_name]

                # Validate audio data structure
                if not isinstance(audio_data, dict):
                    logger.warning("Sample %d: Invalid audio format. Skipping sample.", i)
                    continue

                # Convert to NumPy array
                audio_array = np.array(audio_data.get("array"))
                sr = audio_data.get("sampling_rate")

                if sr is None:
                    logger.warning("Sample %d: Sampling rate missing. Assuming 16kHz.", i)
                    sr = 16000
                
                # Use base class method to resample audio
                audio_array, sr = self.resample_audio(audio_array, sr)

                # Calculate audio duration in seconds
                audio_duration = len(audio_array) / sr
                total_duration += audio_duration

                # Apply dataset filtering
                if (length_filter):
                    if not self.check_audio_length(audio_array, sr, length_filter):
                        continue
                if (num_samples_filter):
                    if sample_count >= num_samples_filter:
                        break

            # Create structured sample with IFBench specific fields
            sample = {
                "array": audio_array if modality == "audio" else audio_data["array"],
                "sampling_rate": sr if modality == "audio" else audio_data["sampling_rate"],
                "audio_content_in_text": prompt,
                "instruction": instruction,
                "id": key,
                "instruction_id_list": instruction_id_list,
                "kwargs": kwargs,
                "model_target": (instruction_id_list, kwargs, prompt)
            }

            processed_data.append(sample)
            sample_count += 1

        self.log_dataset_info(dataset_keys, dataset_size, sample_count, total_duration)

        return processed_data
