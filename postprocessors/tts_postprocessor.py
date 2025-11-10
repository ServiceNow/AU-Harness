"""TTS postprocessor module for AU-Harness framework.

This module provides a postprocessor for Text-to-Speech tasks that extracts
generated audio paths and ground truth text for evaluation.
"""

import logging
from postprocessors.base import Postprocessor

logger = logging.getLogger(__name__)


class TtsPostprocessor(Postprocessor):
    """Postprocessor for TTS - extracts audio paths and ground truth text."""

    def process(self, dataset: list[dict], predictions, metric) -> dict:
        """Process TTS predictions (audio file paths).

        Args:
            dataset: List of preprocessed input samples
            predictions: Dictionary mapping model names to lists of ModelResponse objects
            metric: Evaluation metric

        Returns:
            dict: Dictionary containing processed data for evaluation
        """

        # Extract ground truth text as targets
        targets = [sample.get("ground_truth_text", "") for sample in dataset]

        # Extract generated audio paths from predictions
        processed_predictions = {}
        for model_name, model_predictions in predictions.items():
            # Each prediction is a ModelResponse with audio path in llm_response
            audio_paths = [
                pred.llm_response if pred else ""
                for pred in model_predictions
            ]
            processed_predictions[model_name] = audio_paths

        # Create standardized output
        return self.create_output(
            model_targets=targets,
            processed_predictions=processed_predictions
        )
