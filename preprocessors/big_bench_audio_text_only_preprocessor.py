import logging
from typing import Dict, List, Optional, Any
import numpy as np
from preprocessors.base import Preprocessor

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

        # Convert from columnar to row-wise format
        row_data = self.columnar_to_row_wise(dataset)
        processed_data = []

        for record in row_data:
            sample_id = record["id"]
            audio_data = {
                "array": np.array([]), # Placeholder, not used in text-only evals
                "sampling_rate": 16000
            }
            transcript = record["transcript"]

            # Ensure transcript exists
            if not transcript:
                logger.warning(f"[{sample_id}] Missing transcript for text-only evals. Skipping sample.")
                continue

            # Ensure official answer exists
            if not record["official_answer"]:
                logger.warning(f"[{sample_id}] Missing official answer. Skipping sample.")
                continue

            # Create structured sample
            sample = {
                "id": sample_id,
                "category": record["category"],
                "transcript": transcript,
                "array": audio_data["array"],  # Placeholder, not used in text-only evals
                "sampling_rate": audio_data["sampling_rate"],   # Placeholder, not used in text-only evals
                "model_target": record["official_answer"].strip(),
                "instruction": transcript,
            }

            processed_data.append(sample)

        self.log_dataset_info(dataset_keys, dataset_size, len(processed_data))
        return processed_data