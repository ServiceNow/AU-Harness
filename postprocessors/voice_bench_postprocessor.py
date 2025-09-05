"""VoiceBench IFEval postprocessor for processing instruction following task predictions."""
import logging

from postprocessors.base import Postprocessor

logger = logging.getLogger(__name__)
logger.propagate = True


class VoiceBenchPostprocessor(Postprocessor):
    """
    Postprocessor for VoiceBench Eval model predictions.
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

        processed_predictions = self.process_predictions(predictions)
        output = {
            "instruction": [record.get("instruction", "") for record in dataset],
            "model_targets": [record["model_target"] for record in dataset if "model_target" in record],
            "processed_predictions": processed_predictions,
        }
        self.validate_output(output)
        return output
