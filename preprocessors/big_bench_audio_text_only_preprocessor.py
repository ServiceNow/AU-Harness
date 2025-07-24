import logging
from typing import Dict, List, Optional, Any
import numpy as np
from preprocessors.base import Preprocessor
import tqdm

logger = logging.getLogger(__name__)

class BigBenchAudioTextOnlyPreprocessor(Preprocessor):
    """
    A preprocessor for the BigBenchAudio dataset, designed for
    Speech Query Question Answering (SQQA) tasks. 

    NOTE: Only processes text and metadata, ignoring audio data.

    This class converts a columnar dataset format (dictionary of lists)
    into a row-wise list of dictionaries suitable for model training or inference.
    """

    def process(
        self, 
        dataset: Dict[str, List[Any]], 
        num_samples: Optional[int] = None, 
        properties: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Process the BigBenchAudio dataset to ensure consistent audio format and structured data.

        Parameters:
        - dataset (Dict[str, List[Any]]): A columnar-format dataset where each key maps to a list of values.
            Expected keys: 'id', 'audio', 'category', 'official_answer', 'transcript'.
        - num_samples (Optional[int]): Not used. Reserved for future functionality (e.g., truncating dataset).
        - properties (Optional[Dict[str, Any]]): Not used. Reserved for additional metadata or preprocessing options.

        Returns:
        - List[Dict[str, Any]]: A list of dictionaries where each dictionary represents a sample,
          including the audio array resampled to 16kHz, metadata, and target label.
        """
        
        logger.info("In [BigBenchAudioTextOnlyPreprocessor] Processing dataset...")

        dataset_keys = list(dataset.keys())
        dataset_size = len(dataset.get("id", []))
        self.log_dataset_info(dataset_keys, dataset_size)

        # Direct iteration through the columnar dataset
        processed_data = []
        dataset_size = len(dataset.get("id", []))
        indices = range(dataset_size if num_samples is None else min(dataset_size, num_samples))
        
        for i in tqdm(indices, desc="Processing samples"):
            sample_id = dataset["id"][i]
            audio_data = {
                "array": np.array([]), # Placeholder, not used in text-only evals
                "sampling_rate": 16000
            }
            transcript = dataset["transcript"][i]

            # Ensure transcript exists
            if not transcript:
                logger.warning(f"[{sample_id}] Missing transcript for text-only evals. Skipping sample.")
                continue

            # Ensure official answer exists
            if not dataset["official_answer"][i]:
                logger.warning(f"[{sample_id}] Missing official answer. Skipping sample.")
                continue

            # Create structured sample
            sample = {
                "id": sample_id,
                "category": dataset["category"][i],
                "transcript": transcript,
                "array": audio_data["array"],  # Placeholder, not used in text-only evals
                "sampling_rate": audio_data["sampling_rate"],   # Placeholder, not used in text-only evals
                "model_target": dataset["official_answer"][i].strip(),
                "instruction": transcript,
            }

            processed_data.append(sample)

        self.log_dataset_info(dataset_keys, dataset_size, len(processed_data))
        return processed_data