import logging
import re

logger = logging.getLogger(__name__)


class Postprocessor():
    REQUIRED_KEYS = {"model_targets", "processed_predictions"}

    def validate_output(self, output: dict):
        """Validate that output contains required keys.
        
        Args:
            output (dict): The output dictionary to validate
        
        Raises:
            ValueError: If any required keys are missing
        """
        missing = self.REQUIRED_KEYS - output.keys()
        if missing:
            raise ValueError(f"Postprocessor output missing keys: {missing}")

    @staticmethod
    def remove_thinking_content(sample: str) -> str:
        """
        Removes special formatting tags and extraneous content from a model prediction.

        Specifically removes:
        - Text enclosed in <think>...</think>
        - The special token <|end|>
        
        Args:
            sample (str): Raw model prediction string.

        Returns:
            str: Cleaned version of the prediction.
        """
        cleaned = re.sub(r'<think>.*?</think>', '', sample, flags=re.DOTALL)
        return cleaned.strip()

    def process_predictions(self, predictions: dict[str, list]) -> dict[str, list[str]]:
        """
        Process model predictions by removing thinking content and other artifacts.
        Handles only ModelResponse objects.
        
        Args:
            predictions (dict[str, list]): Dictionary mapping model names to lists of ModelResponse objects
            
        Returns:
            dict[str, list[str]]: Dictionary with processed predictions as strings
        """
        logger.info("Processing predictions...")
        processed_predictions = {}

        for model_name, preds in predictions.items():
            processed = []
            for pred in preds:
                if pred is None:
                    # Handle None values
                    processed.append("")
                else:
                    # Only handle ModelResponse objects
                    text = pred.llm_response if pred.llm_response else ""
                    processed.append(self.remove_thinking_content(text))
            
            processed_predictions[model_name] = processed

        return processed_predictions

    def extract_targets(self, dataset: list[dict], target_key="model_target") -> list:
        """
        Extract targets from dataset using the specified key.
        
        Args:
            dataset (list[dict]): List of preprocessed input samples
            target_key (str): Key to extract from each sample
            
        Returns:
            list: List of extracted targets
        """
        targets = [record[target_key] for record in dataset if target_key in record]
        logger.info(f"Extracted {len(targets)} targets from dataset")
        return targets

    def extract_instructions(self, dataset: list[dict], instruction_key="instruction") -> list:
        """
        Extract instructions from dataset using the specified key.
        
        Args:
            dataset (list[dict]): List of preprocessed input samples
            instruction_key (str): Key to extract from each sample
            
        Returns:
            list: List of extracted instructions
        """
        instructions = [record.get(instruction_key, "") for record in dataset]
        return instructions

    def create_output(self, model_targets, processed_predictions, instructions=None) -> dict:
        """
        Create a standardized output dictionary for postprocessors.
        
        Args:
            model_targets: Target values for evaluation
            processed_predictions: Dictionary of processed model predictions
            instructions (optional): List of instructions
            
        Returns:
            dict: Output dictionary with standard keys
        """
        output = {
            "model_targets": model_targets,
            "processed_predictions": processed_predictions
        }

        if instructions is not None:
            output["instructions"] = instructions

        self.validate_output(output)
        return output

    def process(self, dataset: list[dict], predictions, metric) -> dict:
        """Process and clean model predictions and prepare target-label pairs.
        
        Args:
            dataset (list[dict]): List of preprocessed input samples
            predictions (dict): Dictionary mapping model names to lists of predictions
            metric: Evaluation metric
            
        Returns:
            dict: Dictionary containing processed data for evaluation
        """
        raise NotImplementedError
