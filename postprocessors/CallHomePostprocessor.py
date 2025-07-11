import logger_setup
logger_setup.configure()
import logging
logger = logging.getLogger(__name__)
logger.propagate = True

class CallHomePostprocessor():
    """Postprocessor class to calculate the model scores for the model predictions."""
    def process(self, dataset: list[dict], predictions) -> tuple:
        """Process both model_target and predictions lists:
        - Concatenate all lines starting with 'A' together (no separator)
        - Concatenate all lines starting with 'B' together (space-separated)

        Args:
            dataset: List of sample dicts with key 'model_target'.
            predictions: List of prediction strings.

        Returns:
            tuple: (model_targets, predictions)
        """
        import re
        def split_inline_speaker_labels(text: str) -> str:
            # This will insert a newline before any 'A:' or 'B:' that is not at the start of a line
            # Negative lookbehind for start of line or newline, positive lookahead for 'A:' or 'B:'
            return re.sub(r'(?<!^)(?<!\n)\s*([AB]:)', r'\n\1', text)

        def process_sample(sample: str) -> str:
            # Optionally, split inline speaker labels onto new lines, but do not combine or restructure
            return split_inline_speaker_labels(sample)

        model_targets = [process_sample(record["model_target"]) for record in dataset if "model_target" in record]
        processed_predictions = {}
        logger.info(f"[CallHomePostprocessor.process] Predictions: {predictions}")
        for model_name, preds in predictions.items():
            processed_predictions[model_name] = [process_sample(pred) for pred in preds]

        return model_targets, processed_predictions