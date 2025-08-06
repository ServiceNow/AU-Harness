import logging

logger = logging.getLogger(__name__)
logger.propagate = True
from postprocessors.base import Postprocessor


class _SimpleMeta:  # stand-in for MetricMetadata
    def __init__(self, name, display_name=None, description=""):
        self.name = name
        self.display_name = display_name or name
        self.description = description


MetricMetadata = _SimpleMeta


class ReportingMetrics(dict):
    """Placeholder so downstream code still works."""


class Covost2Postprocessor(Postprocessor):
    """Postprocessor class to calculate the model scores for the model predictions."""
    # Override the REQUIRED_KEYS from the base class to suit translation task.
    REQUIRED_KEYS = {"model_targets", "processed_predictions", "source_sentences"}

    def create_output(self, model_targets, processed_predictions, source_sentences, instructions=None) -> dict:
        """
        Create a standardized output dictionary for postprocessors.
        
        Args:
            model_targets: Target values for evaluation
            processed_predictions: Dictionary of processed model predictions
            instructions (optional): List of instructions
            
        Returns:
            dict: Output dictionary with standard keys
        """
        output = {
            "model_targets": model_targets,
            "processed_predictions": processed_predictions,
            "source_sentences": source_sentences,
        }

        if instructions is not None:
            output["instructions"] = instructions

        self.validate_output(output)
        return output
    
    def process(self, dataset: list[dict], predictions, metric) -> dict:
        """
        Process and clean model predictions and prepare targets for evaluation.
        
        Args:
            dataset (list[dict]): List of preprocessed input samples
            predictions (dict): Dictionary mapping model names to lists of predictions
            metric: Evaluation metric
            
        Returns:
            dict: Dictionary containing processed data for evaluation
        """

        # Process predictions using base class method
        processed_predictions = self.process_predictions(predictions)
        
        # Extract source sentences by using different target_key (used for COMET metric)
        sources = self.extract_targets(dataset, target_key="source_sentence")

        # Extract targets using base class method
        targets = self.extract_targets(dataset, target_key="model_target")

        # Extract instructions using base class method (optional)
        instructions = self.extract_instructions(dataset)

        # Create standardized output using base class method
        return self.create_output(
            model_targets=targets,
            processed_predictions=processed_predictions,
            instructions=instructions,
            source_sentences=sources,
        )