import re
import logging
from utils.custom_logging import configure
from postprocessors.base import Postprocessor

configure()
logger = logging.getLogger(__name__)
logger.propagate = True

class VoiceBenchIfevalPostprocessor(Postprocessor):
    """
    Postprocessor for VoiceBench IFEval model predictions.
    """

    def process(
        self,
        dataset: list[dict],
        predictions: dict[str, list[str]],
        metric
    ) -> tuple[list[tuple[str, str]], dict[str, list[str]], list, list]:
        """
        Process and clean model predictions and prepare target-label pairs.
        """
        logger.info("Processing predictions with VoiceBenchIfevalPostprocessor...")

        processed_predictions: dict[str, list[str]] = {}
        for model_name, preds in predictions.items():
            processed = [self.remove_thinking_content(pred) for pred in preds]
            processed_predictions[model_name] = processed

        output = {
            "instruction": [record.get("instruction", "") for record in dataset],
            "model_targets": [record["model_target"] for record in dataset if "model_target" in record],
            "processed_predictions": processed_predictions,
        }
        self.validate_output(output)
        return output
