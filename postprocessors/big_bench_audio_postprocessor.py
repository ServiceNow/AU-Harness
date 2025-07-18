import re
import logging
from utils.logging import configure
from postprocessors.base import Postprocessor

configure()
logger = logging.getLogger(__name__)
logger.propagate = True


class BigBenchAudioPostprocessor(Postprocessor):
    """
    Postprocessor for BigBenchAudio model predictions.
    
    This class is responsible for cleaning up model outputs (e.g., removing 
    <think>...</think> tags) and structuring data for scoring and evaluation.
    """

    @staticmethod
    def remove_thinking_content(sample: str) -> str:
        """
        Removes special formatting tags and extraneous content from a model prediction.

        Specifically removes:
        - Text enclosed in <think>...</think>
        - The special token <|end|>
        
        Args:
            sample (str): Raw model prediction string.

        Returns:
            str: Cleaned version of the prediction.
        """
        cleaned = re.sub(r'<think>.*?</think>', '', sample, flags=re.DOTALL)
        cleaned = cleaned.replace("<|end|>", "")
        return cleaned.strip()

    def process(
        self,
        dataset: list[dict],
        predictions: dict[str, list[str]],
        metric
    ) -> dict:
        """
        Process and clean model predictions and prepare target-label pairs.

        Args:
            dataset (list[dict]): List of preprocessed input samples.
            predictions (dict[str, list[str]]): Dictionary mapping model names to lists of predictions.
            metric: Placeholder for evaluation metric (not used in current implementation).

        Returns:
            tuple:
                - targets (list[tuple[str, str]]): Tuples of (transcript, official answer).
                - processed_predictions (dict[str, list[str]]): Cleaned predictions per model.
                - list: Placeholder (currently empty).
                - list: Placeholder (currently empty).
        """
        logger.info("Processing predictions with BigBenchAudioPostprocessor...")

        processed_predictions: dict[str, list[str]] = {}

        for model_name, preds in predictions.items():
            logger.debug(f"Processing predictions for model: {model_name}")
            processed = [self.remove_thinking_content(pred) for pred in preds]
            processed_predictions[model_name] = processed
            logger.debug(f"Cleaned {len(processed)} predictions for model: {model_name}")

        # Prepare (transcript, target) pairs
        targets = [
            (record["transcript"], record["model_target"])
            for record in dataset
            if "model_target" in record
        ]

        logger.info(f"Extracted {len(targets)} target-reference pairs from dataset.")

        output = {
            "model_targets": targets,
            "processed_predictions": processed_predictions,
        }
        
        self.validate_output(output)
        return output
