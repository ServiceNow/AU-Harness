"""General preprocessor module for LALMEval framework.

This module provides a general-purpose preprocessor for audio benchmarks
from AudioLLMs and other HuggingFace datasets, with support for various
modalities and filtering options.
"""

import logging
from typing import Dict, List, Any

import numpy as np
from tqdm import tqdm
from datasets import Dataset
from preprocessors.base import Preprocessor

logger = logging.getLogger(__name__)


class GeneralPreprocessor(Preprocessor):
    """Preprocessor for standard Audio benchmarks where output references are ALWAYS expected."""

    def process(self, dataset: Dataset, task_config: Dict[str, Any], 
                run_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run pre-processing on standard/ general Audio datasets.
        
        Args:
            dataset: The task dataset to pre-process
            task_config: Dictionary containing task configuration parameters
            run_config: Dictionary containing run configuration parameters
            
        Returns:
            List of dictionaries where each dictionary represents a pre-processed sample
        """
        
        # Extract common properties using base class method
        modality = task_config.get('modality', 'audio')
        audio_column_name = task_config.get('audio_column', None)
        target_column_name = task_config.get('target_column', None)
        choices_column_name = task_config.get('choices_column', None)
        sample_instruction_column_name = task_config.get('instruction_column', None)
        user_query_column_name = task_config.get('textual_input_column', None)

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
            instruction = user_prompt
            # Create record by accessing each feature by index
            record = {k: row[k] for k in dataset_keys}

            # Extract audio information - if not found, extractor will try audio then context
            self.extract_audio_info(record, audio_column_name=audio_column_name)

            if modality == "text":
                record["array"] = np.array([])  # Placeholder, not used in text-only evals
                record["sampling_rate"] = 16000
                instruction = record.get(user_query_column_name, "")

            # Calculate audio duration in seconds
            audio_duration = len(record["array"]) / record["sampling_rate"]
            total_duration += audio_duration

            # Apply dataset filtering
            if (length_filter):
                if not self.check_audio_length(record["array"], record["sampling_rate"], length_filter):
                    continue
            if (num_samples_filter):
                if sample_count >= num_samples_filter:
                    break

            # General processor requires reference. Otherwise, implement your own preprocessor.
            if target_column_name and target_column_name in record:
                record["model_target"] = record.get(target_column_name, None)
            else:
                raise ValueError("No valid target key found in record")

            # Add sample-specific instructions if they exist in the dataset
            if sample_instruction_column_name and sample_instruction_column_name in record:
                instruction += record.get(sample_instruction_column_name, "")
            
            # Append any user-specified prompt add-ons and choices
            if choices_column_name and choices_column_name in record:
                choices = record.get(choices_column_name, [])
                if isinstance(choices, list):
                    choices_text = " ".join(choices)
                else:
                    choices_text = str(choices)
                instruction += "\n Choices: " + choices_text
            
            # Warning users if no instruction is provided. This can cause evaluated models to hallucinate.
            if not instruction:
                logger.warning("Instruction is empty for sample %d, add user_prompt for instruction insertion", i)
            record["instruction"] = instruction.strip()

            metric_name = task_config.get('metrics')
            if ('judge' in metric_name):
                judge_type = metric_name.split('_')[-1]
                record['judge_type'] = judge_type
            else:
                record['judge_type'] = 'detailed'
            processed_data.append(record)
            sample_count += 1

        self.log_dataset_info(dataset_keys, dataset_size, sample_count, total_duration)

        return processed_data
