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

        import re
        def filter_filler_words(text: str) -> str:
            # List of filler words/sounds and regex patterns for stutters and common transcribed fillers
            fillers = [
                r"\buh\b", r"\bum+\b", r"\buhhuh\b", r"\bmhm+\b", r"\bmm+\b", r"\bah+\b", r"\beh+\b", r"\bhmm+\b",
                r"\bh\b", r"\bye\b", r"\byeah yeah\b", r"\bI I\b", r"\bx+\b", r"\bxxx\b",
                r"\bca-\b", r"\be-\b", r"\bI-\b", r"\bm-\b", r"\bw-\b", r"\b\+/, \+\b", r"\b\+\,\b",
                r"\b(hm)+\b", r"\b(um)+\b", r"\b(uh)+\b"
            ]
            # Remove fillers using regex
            for filler in fillers:
                text = re.sub(filler, '', text, flags=re.IGNORECASE)
            # Remove repeated single letters with dash (e.g., 'e-', 'm-', 'w-')
            text = re.sub(r'\b[a-zA-Z]-\b', '', text)
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            return text

        model_targets = [filter_filler_words(process_sample(record["model_target"])) for record in dataset if "model_target" in record]
        processed_predictions = {}
        logger.info(f"[CallHomePostprocessor.process] Predictions: {predictions}")
        for model_name, preds in predictions.items():
            processed_predictions[model_name] = [filter_filler_words(process_sample(pred)) for pred in preds]

        return model_targets, processed_predictions