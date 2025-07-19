import logging
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)

class Preprocessor():
    def process(self, dataset: list[dict], metric) -> tuple:
        raise NotImplementedError
        
    def load_yaml_file(self, file_name):
        """
        Load a YAML file from the prompts directory.
        
        Args:
            file_name (str): Name of the YAML file in the prompts directory
            
        Returns:
            dict: Contents of the YAML file, or empty dict if file not found
        """
        yaml_path = Path(__file__).resolve().parent.parent / "prompts" / file_name
        try:
            with open(yaml_path, "r") as f:
                content = yaml.safe_load(f) or {}
            return content
        except FileNotFoundError:
            logger.warning(f"File not found at {yaml_path}. Returning empty dictionary.")
            return {}
        
    def extract_properties(self, properties=None):
        """
        Extract common properties from the properties dictionary with default values.
        
        Args:
            properties (dict, optional): Dictionary containing properties for preprocessing.
                                         Defaults to None (empty dict).
                                         
        Returns:
            dict: Dictionary containing extracted properties with defaults applied.
        """
        if properties is None:
            properties = {}
            
        extracted = {
            "metric": properties.get("metric", None),
            "user_prompt_add_ons": properties.get("user_prompt_add_ons", []),
            "system_prompts": properties.get("system_prompts", []),
            "length_filter": properties.get("length_filter", None)  # Optional (min_seconds, max_seconds) tuple
        }
        
        logger.debug(f"Extracted properties: {extracted}")
        return extracted