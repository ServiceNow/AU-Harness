"""
General postprocessor module for standard prediction processing.
"""
import logging

from postprocessors.base import Postprocessor

logger = logging.getLogger(__name__)
logger.propagate = True


class _SimpleMeta:  # stand-in for MetricMetadata
    def __init__(self, name, display_name=None, description=""):
        self.name = name
        self.display_name = display_name or name
        self.description = description


MetricMetadata = _SimpleMeta


class ReportingMetrics(dict):
    """Placeholder so downstream code still works."""


class GeneralPostprocessor(Postprocessor):
    """Postprocessor class to calculate the model scores for the model predictions."""

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

        # Process predictions (in this case, just pass them through)
        processed_predictions = self.process_predictions(predictions)
        # Extract targets using base class method
        targets = self.extract_targets(dataset)

        # Extract instructions using base class method
        instructions = self.extract_instructions(dataset)

        # Create standardized output using base class method
        return self.create_output(
            model_targets=targets,
            processed_predictions=processed_predictions,
            instructions=instructions
        )
