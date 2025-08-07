"""VoiceBench IFEval preprocessor module for LALMEval framework.

This module provides a preprocessor for the VoiceBench IFEval dataset, designed for
instruction following evaluation of audio LLMs.
"""
import logging
from typing import Dict, List, Optional, Any

import numpy as np
from tqdm import tqdm

from preprocessors.base import Preprocessor

logger = logging.getLogger(__name__)


class VoiceBenchIfevalPreprocessor(Preprocessor):
    """
    A preprocessor for the VoiceBench IFEval dataset, designed for
    instruction following evaluation of audio LLMs.
    """

    def process(
            self,
            dataset: Dict[str, List[Any]],
            num_samples: Optional[int] = None,
            properties: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Process the VoiceBench IFEval dataset.

        Args:
            dataset: Dictionary containing audio data
            num_samples: Optional number of samples to process
            properties: Optional dict of properties

        Returns:
            A list of dictionaries where each dictionary represents a sample
        """


        # Extract properties using the base class method
        props = self.extract_properties(properties)
        # Dataset subset of VoiceBench (i.e. ifeval, advbench)
        subset_name = props.get("dataset_info",{}).get("subset",'') 
        modality = props.get("dataset_info", {}).get("modality", "audio")

        # Get dataset info using base class method
        dataset_keys = list(dataset.keys())
        dataset_size = len(dataset.get('prompt', []))
        self.log_dataset_info(dataset_keys, dataset_size)

        processed_data = []
        indices = range(dataset_size if num_samples is None else min(dataset_size, num_samples))

        for i in tqdm(indices, desc="Processing samples"):
            if modality == "text":
                audio_data = {
                    "array": np.array([]),  # Placeholder, not used in text-only evals
                    "sampling_rate": 16000
                }
            else:
                audio_data = dataset["audio"][i]

            prompt = dataset["prompt"][i]

            if modality == "audio":
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

            # Ensure prompt exists
            if not prompt:
                logger.warning("[%s] Missing prompt. Skipping sample.", key)
                continue

            if modality == "text":
                instruction = prompt
            else:
                # For audio modality, we can define a generic instruction
                instruction = "Answer the question provided in the audio."

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

                sample['id'] = key
                sample['instruction_id_list'] = instruction_id_list
                sample['kwargs'] = kwargs
                sample["model_target"] = (instruction_id_list, kwargs, prompt)

            elif (subset_name == 'advbench'):
                sample["model_target"] = ""

            processed_data.append(sample)

        self.log_dataset_info(dataset_keys, dataset_size, len(processed_data))
        return processed_data
