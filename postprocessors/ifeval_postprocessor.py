from utils.custom_logging import configure

configure()
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


class IfevalPostprocessor(Postprocessor):
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
        logger.info("Processing predictions with IfevalPostprocessor...")
        
        # Extract prompts as targets (specific to ifeval format)
        input_prompts = [record["prompt"] for record in dataset if "prompt" in record]
        
        # Extract supporting instructions (specific to ifeval format)
        instructions = [record.get("supporting_instructions", "") for record in dataset]

        # Create standardized output using base class method
        return self.create_output(
            model_targets=input_prompts,
            processed_predictions=predictions,
            instructions=instructions
        )