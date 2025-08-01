import logging
from typing import Dict, List, Optional, Any
import numpy as np
from preprocessors.base import Preprocessor
from tqdm import tqdm

logger = logging.getLogger(__name__)

class BigBenchAudioPreprocessor(Preprocessor):
    """
    A preprocessor for the BigBenchAudio dataset, designed for
    Speech Query Question Answering (SQQA) tasks.
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
        - dataset (Dict[str, List[Any]])
            Expected keys: 'id', 'audio', 'category', 'official_answer', 'transcript'.
        - num_samples (Optional[int]): Not used. Reserved for future functionality (e.g., truncating dataset).
        - properties (Optional[Dict[str, Any]]): Not used. Reserved for additional metadata or preprocessing options.

        Returns:
        - List[Dict[str, Any]]: A list of dictionaries where each dictionary represents a sample,
          including the audio array resampled to 16kHz, metadata, and target label.
        """
        
        logger.info("In [BigBenchAudioPreprocessor] Processing dataset...")

        # Extract properties using the base class method
        props = self.extract_properties(properties)
        modality = props.get("modality", "audio")
        logger.info(f"Processing modality: {modality}")

        dataset_keys = list(dataset.keys())
        dataset_size = len(dataset.get("id", []))
        self.log_dataset_info(dataset_keys, dataset_size)
        
        processed_data = []
        dataset_size = len(dataset.get("id", []))
        indices = range(dataset_size if num_samples is None else min(dataset_size, num_samples))
        
        for i in tqdm(indices, desc="Processing samples"):
            # Create record by accessing each feature by index
            sample_id = dataset["id"][i]

            if modality == "text":
                audio_data = {
                    "array": np.array([]),  # Placeholder, not used in text-only evals
                    "sampling_rate": 16000
                }
            else:
                audio_data = dataset["audio"][i]
            
            if modality == "audio":
                # Validate audio data structure
                if not isinstance(audio_data, dict):
                    logger.warning(f"[{key}] Invalid audio format. Skipping sample.")
                    continue

                # Convert to NumPy array
                audio_array = np.array(audio_data.get("array"))
                sr = audio_data.get("sampling_rate")

                if sr is None:
                    logger.warning(f"[{key}] Sampling rate missing. Assuming 16kHz.")
                    sr = 16000

                # Use base class method to resample audio
                audio_array, sr = self.resample_audio(audio_array, sr)

            # Ensure official answer exists
            if not dataset["official_answer"][i]:
                logger.warning(f"[{sample_id}] Missing official answer. Skipping sample.")
                continue

            if modality == "text":
                instruction = dataset["transcript"][i]
            else:
                # For audio modality, we can define a generic instruction
                instruction = f"Answer the question provided in the audio."

            # Create structured sample
            sample = {
                "id": sample_id,
                "category": dataset["category"][i],
                "audio_content_in_text": dataset["transcript"][i],
                "array": audio_array if modality == "audio" else audio_data["array"],
                "sampling_rate": sr if modality == "audio" else audio_data["sampling_rate"],
                "model_target": dataset["official_answer"][i].strip(),
                "instruction": instruction,
            }

            processed_data.append(sample)

        self.log_dataset_info(dataset_keys, dataset_size, len(processed_data))
        return processed_data
