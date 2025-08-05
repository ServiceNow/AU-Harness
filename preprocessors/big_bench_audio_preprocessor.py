"""BigBench Audio preprocessor module for LALMEval framework.

This module provides a preprocessor for the BigBenchAudio dataset, designed for
Speech Query Question Answering (SQQA) tasks with audio processing capabilities.
"""

import logging
from typing import Dict, List, Optional, Any

import numpy as np
from tqdm import tqdm

from preprocessors.base import Preprocessor

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
        - num_samples (Optional[int]): Not used. Reserved for future functionality 
          (e.g., truncating dataset).
        - properties (Optional[Dict[str, Any]]): Not used. Reserved for additional 
          metadata or preprocessing options.

        Returns:
        - List[Dict[str, Any]]: A list of dictionaries where each dictionary represents a sample,
          including the audio array resampled to 16kHz, metadata, and target label.
        """


        dataset_keys = list(dataset.keys())
        dataset_size = len(dataset.get("id", []))

        processed_data = []
        dataset_size = len(dataset.get("id", []))
        indices = range(dataset_size if num_samples is None else min(dataset_size, num_samples))

        for i in tqdm(indices, desc="Processing samples"):
            # Create record by accessing each feature by index
            sample_id = dataset["id"][i]
            audio_data = dataset["audio"][i]

            # Validate audio data structure
            if not isinstance(audio_data, dict):
                logger.warning("[%d] Invalid audio format. Skipping sample.", i)
                continue

            # Convert to NumPy array
            audio_array = np.array(audio_data.get("array"))
            sr = audio_data.get("sampling_rate")

            if sr is None:
                logger.warning("[%d] Sampling rate missing. Assuming 16kHz.", i)
                sr = 16000

            # Resample if needed
            audio_array, sr = self.resample_audio(audio_array, sr)

            # Ensure official answer exists
            if not dataset["official_answer"][i]:
                logger.warning("[%d] Missing official answer. Skipping sample.", i)
                continue

            # Create structured sample
            sample = {
                "id": sample_id,
                "category": dataset["category"][i],
                "transcript": dataset["transcript"][i],
                "array": audio_array,
                "sampling_rate": sr,
                "model_target": dataset["official_answer"][i].strip(),
                "instruction": "Answer the question provided in the audio.",
            }

            processed_data.append(sample)

        self.log_dataset_info(dataset_keys, dataset_size, len(processed_data))
        return processed_data
