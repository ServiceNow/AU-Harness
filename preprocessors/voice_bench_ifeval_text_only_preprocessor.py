import logging
from typing import Dict, List, Optional, Any
import numpy as np
from scipy.signal import resample
from tqdm import tqdm

logger = logging.getLogger(__name__)

class VoiceBenchIfevalTextOnlyPreprocessor:
    """
    A preprocessor for the VoiceBench IFEval dataset, designed for
    Speech Query Question Answering (SQQA) tasks.

    NOTE: Only processes text and metadata, ignoring audio data.
    """

    def process(
        self, 
        dataset: Dict[str, List[Any]], 
        num_samples: Optional[int] = None, 
        properties: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Process the VoiceBench IFEval datasets.

        Args:
            dataset: Dictionary containing audio data
            num_samples: Optional number of samples to process
            properties: Optional dict of properties

        Returns:
            A list of dictionaries where each dictionary represents a sample
        """
        
        logger.info("In [VoiceBenchIfevalTextOnlyPreprocessor] Processing dataset...")

        dataset_keys = list(dataset.keys())
        dataset_size = len(dataset.get("key", []))
        logger.info(f"Dataset keys: {dataset_keys}, total samples: {dataset_size}")

        processed_data: List[Dict[str, Any]] = []

        for i in tqdm(range(dataset_size), desc="Processing samples"):
            key = dataset["key"][i]
            audio_data = {
                "array": np.array([]), # Placeholder, not used in text-only evals
                "sampling_rate": 16000
            }
            prompt = dataset["prompt"][i]
            instruction_id_list = dataset["instruction_id_list"][i]
            kwargs = dataset["kwargs"][i]

            # Ensure prompt exists
            if not prompt:
                logger.warning(f"[{sample_id}] Missing prompt. Skipping sample.")
                continue

            # Create structured sample
            sample = {
                "id": key,
                "array": audio_data["array"],   # Placeholder, not used in text-only evals
                "sampling_rate": audio_data["sampling_rate"],   # Placeholder, not used in text-only evals
                "audio_content_in_text": prompt,
                "instruction": prompt,  # Using prompt as instruction for text-only evals
                "instruction_id_list": instruction_id_list,
                "kwargs": kwargs,
                "model_target": (instruction_id_list, kwargs, prompt),
            }

            processed_data.append(sample)

        logger.info(f"Processed dataset size: {len(processed_data)}")
        return processed_data
