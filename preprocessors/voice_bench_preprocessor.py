"""VoiceBench preprocessor module for evaluation framework.

This module provides a preprocessor for the VoiceBench datasets, designed for
instruction following evaluation of audio LLMs and more.
"""
import logging
from typing import Dict, List, Optional, Any

import numpy as np
from tqdm import tqdm

from preprocessors.base import Preprocessor

logger = logging.getLogger(__name__)


class VoiceBenchPreprocessor(Preprocessor):
    """
    A preprocessor for the VoiceBench datasets, designed for
    instruction following evaluation of audio LLMs.
    """

    def process(self, dataset: Dict[str, List[Any]], task_config: Dict[str, Any], 
                run_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process the VoiceBench datasets, most of which do not have sample references.

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
        subset_name = task_config.get('subset', '')

        # Obtain task-specific prompt (if provided)
        user_prompt = task_config.get('user_prompt', '')

        # Obtain length filter (if exists)
        filters = run_config.get('filter', None)
        length_filter = None
        if (filters):
            length_filter = filters.get('length', None)

        # Get dataset info
        dataset_keys = list(dataset.keys())
        dataset_size = len(dataset[dataset_keys[0]]) if dataset_keys else 0
        self.log_dataset_info(dataset_keys, dataset_size)

        processed_data = []
        indices = range(dataset_size)

        for i in tqdm(indices, desc="Processing samples"):

            # Ensure prompt exists. Otherwise, move onto the next sample.
            prompt = dataset[sample_instruction_column_name][i]
            if not prompt:
                logger.warning("[%s] Missing prompt. Skipping sample.", key)
                continue

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
                audio_data = dataset[audio_column_name][i]

                # Validate audio data structure
                if not isinstance(audio_data, dict):
                    logger.warning("[%s] Invalid audio format. Skipping sample.", key)
                    continue

                # Convert to NumPy array
                audio_array = np.array(audio_data.get("array"))
                sr = audio_data.get("sampling_rate")

                if sr is None:
                    logger.warning("[%s] Sampling rate missing. Assuming 16kHz.", key)
                    sr = 16000
                
                # Use base class method to resample audio
                audio_array, sr = self.resample_audio(audio_array, sr)

                # Apply length filtering if specified
                if (length_filter):
                    if not self.check_audio_length(audio_array, sr, length_filter):
                        continue

            # Create structured sample
            sample = {
                "array": audio_array if modality == "audio" else audio_data["array"],
                "sampling_rate": sr if modality == "audio" else audio_data["sampling_rate"],
                "audio_content_in_text": prompt,
                "instruction": instruction,
                "model_target": (prompt)
            }

            # Handle additional keys for IFEval (overwrite if needed)
            if (subset_name == 'ifeval'):
                key = dataset['key'][i]
                instruction_id_list = dataset["instruction_id_list"][i]
                kwargs = dataset["kwargs"][i]

                sample.update({
                    'id': key,
                    'instruction_id_list': instruction_id_list,
                    'kwargs': kwargs,
                    'model_target': (instruction_id_list, kwargs, prompt)
                })

            elif (subset_name == 'advbench'):
                sample["model_target"] = ""

            processed_data.append(sample)

        self.log_dataset_info(dataset_keys, dataset_size, len(processed_data))
        return processed_data
