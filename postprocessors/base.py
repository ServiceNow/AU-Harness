import re

class Postprocessor():
    REQUIRED_KEYS = {"model_targets", "processed_predictions"}
    
    def validate_output(self, output: dict):
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
        cleaned = cleaned.replace("<|end|>", "")
        return cleaned.strip()

    def process(self, dataset: list[dict], predictions, metric) -> dict:
        raise NotImplementedError
