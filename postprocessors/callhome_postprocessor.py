from utils.custom_logging import configure
configure()
import logging
from postprocessors.base import Postprocessor
logger = logging.getLogger(__name__)
logger.propagate = True
import re

class CallhomePostprocessor(Postprocessor):
    """Postprocessor class to calculate the model scores for the model predictions."""
    def split_inline_speaker_labels(self, text: str) -> str:
        # This will insert a newline before any 'A:' or 'B:' that is not at the start of a line
        return re.sub(r'(?<!^)(?<!\n)\s*([AB]:)', r'\n\1', text)

    def process(self, dataset: list[dict], predictions, metric=None) -> dict:
        # Process model targets directly with split_inline_speaker_labels (removed process_sample middleman)
        model_targets = [self.split_inline_speaker_labels(record.get("model_target", "")) for record in dataset]
        processed_predictions = {
            model_name: [self.split_inline_speaker_labels(pred) for pred in preds]
            for model_name, preds in predictions.items()
        }
        # Special handling for word_error_rate metric
        if metric == "word_error_rate":
            # More robust handling of IDs and audio lengths
            ids = []
            lengths = []
            for record in dataset:
                ids.append(record.get("id", "unknown")[:4])
                array = record.get("array")
                sampling_rate = record.get("sampling_rate", 16000)
                length = len(array) / sampling_rate if array is not None else 0
                lengths.append(length)
                
            output = {
                "model_targets": model_targets,
                "processed_predictions": processed_predictions,
                "ids": ids,
                "lengths": lengths
            }
            
            self.validate_output(output)
            return output

        output = {
            "model_targets": model_targets,
            "processed_predictions": processed_predictions
        }
        
        self.validate_output(output)
        return output