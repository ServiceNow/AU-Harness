"""Covost2 preprocessor module for LALMEval framework.

This module provides a preprocessor for the Covost2 dataset, designed for
translation tasks with support for both audio and text modalities.
"""

import logging
from typing import Dict, List, Any

from datasets import Dataset
from tqdm import tqdm

from preprocessors.base import Preprocessor

logger = logging.getLogger(__name__)
from utils.constants import language_map  # Import language_map from constants

class Covost2Preprocessor(Preprocessor):
    """Preprocessor for Covost2 dataset from fixie-ai/covost2."""

    def process(self, dataset: Dataset, task_config: Dict[str, Any], 
                run_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run pre-processing on Covost2 type of datasets.
        
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

        processed_data = []
        total_duration = 0
        sample_count = 0

        for row in tqdm(dataset, desc="Processing samples"):
            sample_id = row['id']
            translation = row['translation']
            source_sentence = row['sentence']
            audio = row['audio']
            audio_array = audio['array']
            sampling_rate = audio['sampling_rate']

            if sampling_rate is None:
                logger.warning("[%s] Sampling rate missing. Assuming 16kHz.", sample_id)
                sampling_rate = 16000

            # Resample if needed using base class method
            audio_array, sampling_rate = self.resample_audio(audio_array, sampling_rate)

            # Calculate audio duration in seconds
            audio_duration = len(audio_array) / sampling_rate
            total_duration += audio_duration

            # Apply length filtering if specified
            if (length_filter):
                if not self.check_audio_length(audio_array, sampling_rate, length_filter):
                    continue
            if (num_samples_filter):
                if sample_count >= num_samples_filter:
                    break

            # Get the target language from dataset_info
            try:
                target_language_code = task_config.get("target_language")
                # Convert language code to full language name using the language_map
                target_language_name = language_map.get(target_language_code, target_language_code)
                target_language_name = target_language_name.capitalize()
            except KeyError as exc:
                raise ValueError("Target language not found. Please specify target_language in the task config") from exc
            
            # Get the source language from dataset_info
            try:
                source_language_code = task_config.get("source_language")
                # Convert language code to full language name using the language_map
                source_language_name = language_map.get(source_language_code, source_language_code)
                source_language_name = source_language_name.capitalize()
            except KeyError as exc:
                raise ValueError("Source language not found. Please specify source_language in the task config") from exc

            task_instruction_prompt = task_config.get("user_prompt", "")

            # Create structured sample
            sample = {
                "id": sample_id,
                "array": audio_array,
                "sampling_rate": sampling_rate,
                "model_target": translation.strip(),
                "source_sentence": source_sentence.strip(),
                "instruction": task_instruction_prompt,
                "source_language_name": source_language_name,
                "target_language_name": target_language_name
            }
            processed_data.append(sample)
            sample_count += 1

        self.log_dataset_info(dataset_keys, dataset_size, sample_count, total_duration)
        
        return processed_data