"""TTS preprocessor module for AU-Harness framework.

This module provides a preprocessor for Text-to-Speech tasks that extracts
text from datasets for audio generation.
"""

import logging
from typing import Dict, List, Any
import numpy as np
from tqdm import tqdm
from datasets import Dataset
from preprocessors.base import Preprocessor

logger = logging.getLogger(__name__)


class TtsPreprocessor(Preprocessor):
    """Preprocessor for TTS tasks - extracts text for generation."""

    def process(self, dataset: Dataset, task_config: Dict[str, Any],
                run_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract text from dataset for TTS generation.

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
        length_filter, num_samples_filter = self.get_dataset_filters(
            run_config.get('filter', None), dataset_size
        )

        processed_data = []
        sample_count = 0

        for i, row in enumerate(tqdm(dataset, desc="Processing TTS samples")):
            record = {k: row[k] for k in dataset_keys}

            # Find text column (priority order: normalized_text -> text_normalized -> text)
            text = None
            for key in ['normalized_text', 'text_normalized', 'text']:
                if key in record:
                    text = record[key]
                    break

            if not text:
                logger.warning("No text column found in sample %d, skipping", i)
                continue

            # Store text in standardized column
            record["ground_truth_text"] = text

            # Placeholder audio (not used for TTS generation)
            record["array"] = np.array([])
            record["sampling_rate"] = 16000
            record["instruction"] = ""  # Not used for TTS
            record["model_target"] = text  # For compatibility with postprocessor

            # Apply sample count filter
            if num_samples_filter and sample_count >= num_samples_filter:
                break

            processed_data.append(record)
            sample_count += 1

        self.log_dataset_info(dataset_keys, dataset_size, sample_count)

        return processed_data
