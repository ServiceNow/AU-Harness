import logger_setup
logger_setup.configure()
import logging
import numpy as np
from collections import defaultdict
logger = logging.getLogger(__name__)
logger.propagate = True

class CallHomePostprocessor():
    """Postprocessor class to calculate the model scores for the model predictions."""
    def process(self, dataset: list[dict], predictions, metric=None) -> tuple:

        import re
        def split_inline_speaker_labels(text: str) -> str:
            # This will insert a newline before any 'A:' or 'B:' that is not at the start of a line
            # Negative lookbehind for start of line or newline, positive lookahead for 'A:' or 'B:'
            return re.sub(r'(?<!^)(?<!\n)\s*([AB]:)', r'\n\1', text)

        def process_sample(sample: str) -> str:
            # Optionally, split inline speaker labels onto new lines, but do not combine or restructure
            return split_inline_speaker_labels(sample)

        print ("Check samples", record['model_target'])
        assert 22 == 1
        model_targets = [process_sample(record["model_target"]) for record in dataset if "model_target" in record]
        processed_predictions = {}
        for model_name, preds in predictions.items():
            processed_predictions[model_name] = [process_sample(pred) for pred in preds]
        # Special handling for word_error_rate metric
        if metric == "word_error_rate":
            # Extract first 4 letters of ID for each sample
            ids = [record["id"][:4] if "id" in record else "unknown" for record in dataset]
            
            # Calculate length of each audio sample
            lengths = []
            for record in dataset:
                if "array" in record:
                    # Get the length in seconds (assuming standard sample rate)
                    length = len(record["array"]) / record.get("sampling_rate", 16000)
                    lengths.append(length)
                else:
                    lengths.append(0)
            return model_targets, processed_predictions, ids, lengths

        return model_targets, processed_predictions, [], []