import logging

from models.model_response import ModelResponse
from postprocessors.base import Postprocessor

logger = logging.getLogger(__name__)
logger.propagate = True


class MultiturnPostprocessor(Postprocessor):
    """
    Postprocessor for bfcl predictions.
    """

    def process(
            self,
            dataset: list[dict],
            predictions: ModelResponse,
            metric
    ) -> tuple[list[tuple[str, str]], dict[str, list[str]], list, list] | dict:
        """
        Process and clean model predictions and prepare target-label pairs.
        """
        logger.info("Processing predictions with Multi-Turn Postprocessor...")

        processed_predictions: dict[str, list[str]] = {}
        for model_name, preds in predictions.items():
            processed = []
            for i, (pred, dataset_row) in enumerate(zip(preds, dataset)):
                id = dataset_row["id"]
                instructions = dataset_row["instruction"] if dataset_row["instruction"] else dataset_row["textual_input"]
                targets = dataset_row["model_target"]
                category = dataset_row.get("category", "default")

                if targets:
                    if len(targets)!=len(preds):
                        ValueError("Targets and predictions must have same length")
                else:
                    targets = [None] * len(preds)

                if isinstance(pred.raw_response, list):
                    predictions = [i.get('choices', [])[0].get('message', {}).get('content', "") for i in
                                   pred.raw_response]
                else:
                    predictions = pred.raw_response if isinstance(pred.raw_response, str) else None
                if predictions:
                    processed_pred = {'id': id,
                                      'category': category,
                                      'instructions': instructions,
                                      'responses': predictions,
                                      'targets': targets,
                                      'turns': list(range(len(predictions)))}
                    processed.append(processed_pred)
            processed_predictions[model_name] = processed

        output = {
            "ids": [record.get("id") for record in dataset],
            "instructions": [record.get("instruction", "") for record in dataset],
            "model_targets": [record["model_target"] for record in dataset if "model_target" in record],
            "processed_predictions": processed_predictions,
        }
        self.validate_output(output)
        return output
