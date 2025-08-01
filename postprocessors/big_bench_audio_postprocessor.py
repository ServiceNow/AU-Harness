import logging
from utils.custom_logging import configure
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

    # Using remove_thinking_content from base class

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
            dict: Dictionary containing processed data for evaluation including targets and processed predictions.
        """
        logger.info("Processing predictions with BigBenchAudioPostprocessor...")

        # Process predictions using base class method
        processed_predictions = self.process_predictions(predictions)

        # Prepare (transcript, target) pairs
        targets = [
            (record["audio_content_in_text"], record["model_target"])
            for record in dataset
            if "model_target" in record and "audio_content_in_text" in record
        ]
        
        # Extract instructions using base class method
        instructions = self.extract_instructions(dataset)

        logger.info(f"Extracted {len(targets)} target-reference pairs from dataset.")

        # Create standardized output using base class method
        return self.create_output(
            model_targets=targets,
            processed_predictions=processed_predictions,
            instructions=instructions
        )
