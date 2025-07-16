import logger_setup
logger_setup.configure()
import logging
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


class AudiobenchPostprocessor():
    """Postprocessor class to calculate the model scores for the model predictions."""
    def process(self, dataset: list[dict], predictions, metric) -> tuple:
        return [record["model_target"] for record in dataset if "model_target" in record], predictions, [], []