class Postprocessor():
    REQUIRED_KEYS = {"model_targets", "processed_predictions"}
    
    def validate_output(self, output: dict):
        missing = self.REQUIRED_KEYS - output.keys()
        if missing:
            raise ValueError(f"Postprocessor output missing keys: {missing}")
    
    def process(self, dataset: list[dict], predictions, metric) -> dict:
        raise NotImplementedError
