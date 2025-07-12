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
        """Process both model_target and predictions lists:
        - Concatenate all lines starting with 'A' together (no separator)
        - Concatenate all lines starting with 'B' together (space-separated)

        Args:
            dataset: List of sample dicts with key 'model_target'.
            predictions: List of prediction strings.
            metric: Optional name of metric being used.

        Returns:
            tuple: If metric is word_error_rate, returns (model_targets, predictions, ids, lengths).
                  Otherwise, returns (model_targets, predictions).
        """
        logger.info(f"full dataset: {dataset}")
        logger.info(f"full predictions: {predictions}")
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
        logger.info(f"metric: {metric}")
        # Special handling for word_error_rate metric
        if metric == "word_error_rate":
            logger.info(f"[CallHomePostprocessor.process] Hit word error rate metric")
            # Extract first 4 letters of ID for each sample
            ids = [record["id"][:4] if "id" in record else "unknown" for record in dataset]
            
            # Calculate length of each audio sample
            lengths = []
            logger.info(f"[CallHomePostprocessor.process] dataset[0] keys: {dataset[0].keys()}")
            for record in dataset:
                if "array" in record:
                    # Get the length in seconds (assuming standard sample rate)
                    length = len(record["array"]) / record.get("sampling_rate", 16000)
                    lengths.append(length)
                else:
                    lengths.append(0)
            
            # Print distribution of sample lengths
            if lengths:
                logger.info(f"Audio sample length statistics:")
                logger.info(f"Min length: {np.min(lengths):.2f} seconds")
                logger.info(f"Max length: {np.max(lengths):.2f} seconds")
                logger.info(f"Mean length: {np.mean(lengths):.2f} seconds")
                logger.info(f"Median length: {np.median(lengths):.2f} seconds")
                logger.info(f"Standard deviation: {np.std(lengths):.2f} seconds")
                
                # Create a simple histogram distribution
                hist, bins = np.histogram(lengths, bins=5)
                logger.info("Sample length distribution:")
                for i in range(len(hist)):
                    logger.info(f"{bins[i]:.1f}-{bins[i+1]:.1f}s: {hist[i]} samples")
            
            return model_targets, processed_predictions, ids, lengths

        return model_targets, processed_predictions, [], []