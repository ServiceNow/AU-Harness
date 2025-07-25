import re
import logging
from utils.custom_logging import configure
from postprocessors.base import Postprocessor

configure()
logger = logging.getLogger(__name__)
logger.propagate = True

class CallhomePostprocessor(Postprocessor):
    """Postprocessor class to calculate the model scores for the model predictions."""
    def split_inline_speaker_labels(self, text: str) -> str:
        # This will insert a newline before any 'A:' or 'B:' that is not at the start of a line
        return re.sub(r'(?<!^)(?<!\n)\s*([AB]:)', r'\n\1', text)

    def process_predictions(self, predictions: dict[str, list[str]]) -> dict[str, list[str]]:
        """
        Process model predictions by applying speaker label splitting.
        Overrides the base class method to add specialized behavior.
        
        Args:
            predictions (dict[str, list[str]]): Dictionary mapping model names to lists of predictions
            
        Returns:
            dict[str, list[str]]: Dictionary with processed predictions
        """
        logger.info("Processing predictions with CallhomePostprocessor...")
        processed_predictions = {}
        
        for model_name, preds in predictions.items():
            logger.debug(f"Processing predictions for model: {model_name}")
            # Apply CallhomePostprocessor-specific processing
            processed = [self.split_inline_speaker_labels(pred) for pred in preds]
            processed_predictions[model_name] = processed
            logger.debug(f"Cleaned {len(processed)} predictions for model: {model_name}")
            
        return processed_predictions
        
    def extract_targets(self, dataset: list[dict], target_key="model_target") -> list:
        """
        Extract targets from dataset and apply speaker label splitting.
        Overrides the base class method to add specialized behavior.
        
        Args:
            dataset (list[dict]): List of preprocessed input samples
            target_key (str): Key to extract from each sample
            
        Returns:
            list: List of extracted and processed targets
        """
        targets = [self.split_inline_speaker_labels(record.get(target_key, "")) for record in dataset]
        logger.info(f"Extracted and processed {len(targets)} targets from dataset")
        return targets
    
    def extract_audio_metadata(self, dataset: list[dict]) -> tuple[list, list]:
        """
        Extract audio metadata (IDs and lengths) from dataset.
        
        Args:
            dataset (list[dict]): List of preprocessed input samples
            
        Returns:
            tuple[list, list]: Tuple containing lists of IDs and lengths
        """
        ids = []
        lengths = []
        for record in dataset:
            ids.append(record.get("id", "unknown")[:4])
            array = record.get("array")
            sampling_rate = record.get("sampling_rate", 16000)
            length = len(array) / sampling_rate if array is not None else 0
            lengths.append(length)
        return ids, lengths
        
    def process(self, dataset: list[dict], predictions, metric=None) -> dict:
        """
        Process and clean model predictions and prepare target-label pairs.
        Special handling for word_error_rate metric to include audio metadata.
        
        Args:
            dataset (list[dict]): List of preprocessed input samples
            predictions (dict): Dictionary mapping model names to lists of predictions
            metric (str, optional): Evaluation metric name
            
        Returns:
            dict: Dictionary containing processed data for evaluation
        """
        # Process predictions using our overridden method
        processed_predictions = self.process_predictions(predictions)
        
        # Extract targets using our overridden method
        model_targets = self.extract_targets(dataset)
        
        # Special handling for word_error_rate metric
        output = {
            "model_targets": model_targets,
            "processed_predictions": processed_predictions
        }
        if metric == "word_error_rate":
            # Extract audio metadata
            ids, lengths = self.extract_audio_metadata(dataset)
            
            # Create output with additional metadata
            output = {
                "model_targets": model_targets,
                "processed_predictions": processed_predictions,
                "ids": ids,
                "lengths": lengths
            }
            
            self.validate_output(output)
            return output

        # For other metrics, use standard output format
        return self.create_output(
            model_targets=model_targets,
            processed_predictions=processed_predictions
        )
