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
        def process_sample(sample: str) -> str:
            lines = sample.splitlines()
            a_words = []
            b_words = []
            for line in lines:
                line = line.strip()
                if line.startswith('A:'):
                    a_words.append(line[2:].strip())
                elif line.startswith('B:'):
                    b_words.append(line[2:].strip())
            return f"A: {' '.join(a_words)}\nB: {' '.join(b_words)}"

        model_targets = [process_sample(record["model_target"]) for record in dataset if "model_target" in record]
        processed_predictions = {}
        logger.info(f"[CallHomePostprocessor.process] Predictions: {predictions}")
        for model_name, preds in predictions.items():
            processed_predictions[model_name] = [process_sample(pred) for pred in preds]

        return model_targets, processed_predictions