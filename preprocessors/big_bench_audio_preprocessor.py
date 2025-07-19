import logging
from typing import Dict, List, Optional, Any
import numpy as np
from scipy.signal import resample
from tqdm import tqdm
from preprocessors.base import Preprocessor

logger = logging.getLogger(__name__)

class BigBenchAudioPreprocessor(Preprocessor):
    """
    A preprocessor for the BigBenchAudio dataset, designed for
    Speech Query Question Answering (SQQA) tasks.

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
        
        logger.info("In [BigBenchAudioPreprocessor] Processing dataset...")

        dataset_keys = list(dataset.keys())
        dataset_size = len(dataset.get("id", []))
        logger.info(f"Dataset keys: {dataset_keys}, total samples: {dataset_size}")

        processed_data: List[Dict[str, Any]] = []

        for i in tqdm(range(dataset_size), desc="Processing samples"):
            sample_id = dataset["id"][i]
            audio_data = dataset["audio"][i]
            category = dataset["category"][i]
            official_answer = dataset["official_answer"][i]
            transcript = dataset["transcript"][i]

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

            # Resample if needed
            if sr != 16000:
                target_length = int(16000 * len(audio_array) / sr)
                audio_array = resample(audio_array, target_length)
                sr = 16000

            # Ensure official answer exists
            if not official_answer:
                logger.warning(f"[{sample_id}] Missing official answer. Skipping sample.")
                continue

            # Create structured sample
            sample = {
                "id": sample_id,
                "category": category,
                "transcript": transcript,
                "array": audio_array,
                "sampling_rate": sr,
                "model_target": official_answer.strip(),
                "instruction": "Answer the question provided in the audio.",
            }

            processed_data.append(sample)

        logger.info(f"Processed dataset size: {len(processed_data)}")
        return processed_data
