import logger_setup
logger_setup.configure()
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
    def process(self, dataset: list[dict], predictions, metric) -> tuple:
        logger.info("Processing predictions with Covost2Postprocessor...")

        processed_predictions: dict[str, list[str]] = {}

        for model_name, preds in predictions.items():
            processed = [self.remove_thinking_content(pred) for pred in preds]
            processed_predictions[model_name] = processed

        return [record["model_target"] for record in dataset if "model_target" in record], processed_predictions, [], []