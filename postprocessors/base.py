from typing import Dict, List, Any, Union

class Postprocessor:
    def extract_model_targets(self, dataset: List[Dict]) -> List:
        raise NotImplementedError("Subclasses must implement extract_model_targets")
    
    def process(self, results: List[Dict]) -> Dict[str, Union[float, List]]:
        raise NotImplementedError("Subclasses must implement process")