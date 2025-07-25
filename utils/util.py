
from typing import Any, Dict
import os
import yaml

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from . import constants

def smart_round(val: float, precision: int = constants.ROUND_DIGITS) -> float:
    """Round off metrics to global precision value.

    References:
        1. https://bugs.python.org/msg358467
        2. https://en.wikipedia.org/wiki/IEEE_754

    Args:
    ----
        precision: int: Precision up to which value should be rounded off.
        val: float: Value

    Returns:
    -------
        float: Rounded off value
    """
    if not isinstance(precision, int) or precision <= 0:
        logger.warning(
            f"Invalid precision provided: {precision}. Using the default precision: {constants.ROUND_DIGITS}"
        )
        precision = constants.ROUND_DIGITS
    rounded_off_val = round(val * 10**precision) / 10**precision
    return rounded_off_val

def get_context_indices_for_filter(key: str, value: Any, contexts: list[dict]) -> list[int]:
    """Get indices for rows satisfying the given filter.

    Given key-value pair, it returns the list of indices of contexts satisfying key = value.

    Args:
        key: The key to match against in each row of context/data.
        value: The value to compare against
        contexts: list of dictionaries containing additional key-value pairs in data.

    Returns:
        List of integer indices.

    """
    indices = [_ for _, c in enumerate(contexts) if c[key] == value]
    return indices

def validate_config(config_path: str) -> Dict:
    """Validate configuration file against expected structure and types.
    
    Args:
        config_path: Path to the config.yaml file
        
    Returns:
        Dict: The validated configuration
        
    Raises:
        ValueError: If the configuration is invalid
    """
    if not os.path.exists(config_path):
        raise ValueError(f"Config file does not exist: {config_path}")
        
    try:
        # Check if it's a well-formatted YAML file
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            
        if config is None:
            raise ValueError(f"Config file is empty: {config_path}")
        
        # Validate dataset_metric
        if 'dataset_metric' not in config:
            raise ValueError("'dataset_metric' is required")
        if not isinstance(config['dataset_metric'], list):
            raise ValueError("'dataset_metric' must be a list")
        if len(config['dataset_metric']) == 0:
            raise ValueError("'dataset_metric' must have at least one element")
        
        # Check that each dataset_metric entry has a value
        for i, entry in enumerate(config['dataset_metric']):
            if not entry or not isinstance(entry, str) or len(entry.strip()) == 0:
                raise ValueError(f"Dataset-metric entry {i+1} must be a non-empty string")
        
        # Validate num_samples if it exists
        if 'num_samples' in config:
            if not isinstance(config['num_samples'], int):
                raise ValueError("'num_samples' must be an integer")
        
        # Validate judge_concurrency if it exists
        if 'judge_concurrency' in config:
            if not isinstance(config['judge_concurrency'], int):
                raise ValueError("'judge_concurrency' must be an integer")
        
        # Validate judge_model if it exists
        if 'judge_model' in config:
            if not isinstance(config['judge_model'], str):
                raise ValueError("'judge_model' must be a string")
        
        # Validate user_prompt_add_ons if it exists
        if 'user_prompt_add_ons' in config:
            if not isinstance(config['user_prompt_add_ons'], list) or len(config['user_prompt_add_ons']) < 1:
                raise ValueError("'user_prompt_add_ons' must be a list with at least one value")
        
        # Validate length_filter if it exists
        if 'length_filter' in config:
            if not isinstance(config['length_filter'], list) or len(config['length_filter']) != 2:
                raise ValueError("'length_filter' must be a list with exactly 2 elements")
            for value in config['length_filter']:
                if not isinstance(value, (int, float)):
                    raise ValueError("'length_filter' elements must be numbers")
        
        # Validate accented if it exists
        if 'accented' in config:
            if not isinstance(config['accented'], bool):
                raise ValueError("'accented' must be a boolean")
        
        # Validate language if it exists
        if 'language' in config:
            if not isinstance(config['language'], str):
                raise ValueError("'language' must be a string")
        
        # Validate models section
        _validate_models(config)
        
        # Validate aggregate section if it exists
        if 'aggregate' in config:
            _validate_aggregate(config['aggregate'])
        
        # Validate temperature_overrides section if it exists
        if 'temperature_overrides' in config:
            _validate_temperature_overrides(config['temperature_overrides'])
        
        return config
        
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML format: {str(e)}")

def _validate_models(config: Dict) -> None:
    """Validate the models section of the configuration.
    
    Args:
        config: The configuration dictionary
        
    Raises:
        ValueError: If the models section is invalid
    """
    if 'models' not in config:
        raise ValueError("'models' section is required")
    
    if not isinstance(config['models'], list):
        raise ValueError("'models' must be a list")
    
    for i, model_entry in enumerate(config['models']):
        if not isinstance(model_entry, dict) or 'info' not in model_entry:
            raise ValueError(f"Model entry {i+1} must have an 'info' object")
        
        info = model_entry['info']
        required_fields = ['name', 'model', 'inference_type', 'url']
        
        # Check required fields exist and have values
        for field in required_fields:
            if field not in info:
                raise ValueError(f"Model {i+1} is missing required field: '{field}'")
            if info[field] is None or (isinstance(info[field], str) and len(info[field].strip()) == 0):
                raise ValueError(f"Model {i+1}: '{field}' must have a non-empty value")
        
        # Validate field types
        if not isinstance(info['name'], str):
            raise ValueError(f"Model {i+1}: 'name' must be a string")
        
        if not isinstance(info['model'], str):
            raise ValueError(f"Model {i+1}: 'model' must be a string")
        
        if not isinstance(info['inference_type'], str):
            raise ValueError(f"Model {i+1}: 'inference_type' must be a string")
        
        if not isinstance(info['url'], str):
            raise ValueError(f"Model {i+1}: 'url' must be a string")
        
        # Optional fields with type validation
        if 'delay' in info and not isinstance(info['delay'], int):
            raise ValueError(f"Model {i+1}: 'delay' must be an integer")
            
        if 'retry_attempts' in info and not isinstance(info['retry_attempts'], int):
            raise ValueError(f"Model {i+1}: 'retry_attempts' must be an integer")
            
        if 'timeout' in info and not isinstance(info['timeout'], int):
            raise ValueError(f"Model {i+1}: 'timeout' must be an integer")
            
        if 'auth_token' in info and not isinstance(info['auth_token'], str):
            raise ValueError(f"Model {i+1}: 'auth_token' must be a string")
            
        if 'api_version' in info and not isinstance(info['api_version'], str):
            raise ValueError(f"Model {i+1}: 'api_version' must be a string")
            
        if 'batch_size' in info and not isinstance(info['batch_size'], int):
            raise ValueError(f"Model {i+1}: 'batch_size' must be an integer")
            
        if 'chunk_size' in info and not isinstance(info['chunk_size'], int):
            raise ValueError(f"Model {i+1}: 'chunk_size' must be an integer")


def _validate_aggregate(aggregate_section) -> None:
    """
    Validate the aggregate section of the configuration.
    
    Structure should be:
    aggregate:
      - ["name", [["dataset1", "metric1"], ["dataset2", "metric2"]]]
    
    Args:
        aggregate_section: The aggregate section to validate
        
    Raises:
        ValueError: If the aggregate section is invalid
    """
    if not isinstance(aggregate_section, list):
        raise ValueError("'aggregate' must be a list")
    
    for i, aggregate_entry in enumerate(aggregate_section):
        # Check that the aggregate entry is a list with exactly 2 elements
        if not isinstance(aggregate_entry, list):
            raise ValueError(f"Aggregate entry {i+1} must be a list")
            
        if len(aggregate_entry) != 2:
            raise ValueError(f"Aggregate entry {i+1} must have exactly 2 elements")
            
        # First element should be a string (name) and non-empty
        if not isinstance(aggregate_entry[0], str):
            raise ValueError(f"Aggregate entry {i+1}: first element must be a string name")
        if len(aggregate_entry[0].strip()) == 0:
            raise ValueError(f"Aggregate entry {i+1}: name cannot be empty")
            
        # Second element should be a list of dataset-metric pairs
        if not isinstance(aggregate_entry[1], list):
            raise ValueError(f"Aggregate entry {i+1}: second element must be a list of dataset-metric pairs")
            
        dataset_metrics = aggregate_entry[1]
        for j, pair in enumerate(dataset_metrics):
            if not isinstance(pair, list):
                raise ValueError(f"Aggregate entry {i+1}, pair {j+1} must be a list")
                
            if len(pair) != 2:
                raise ValueError(f"Aggregate entry {i+1}, pair {j+1} must have exactly 2 elements (dataset, metric)")
                
            if not isinstance(pair[0], str) or not isinstance(pair[1], str):
                raise ValueError(f"Aggregate entry {i+1}, pair {j+1}: both dataset and metric must be strings")
            if len(pair[0].strip()) == 0 or len(pair[1].strip()) == 0:
                raise ValueError(f"Aggregate entry {i+1}, pair {j+1}: dataset and metric must be non-empty strings")


def _validate_temperature_overrides(temperature_overrides) -> None:
    """
    Validate the temperature_overrides section of the configuration.
    
    Structure should be:
    temperature_overrides:
      - model: "model_name" (optional)
        task: "task_name" (optional)
        temperature: 0.5 (required)
    
    Either model or task (or both) must be present.
    
    Args:
        temperature_overrides: The temperature_overrides section to validate
        
    Raises:
        ValueError: If the temperature_overrides section is invalid
    """
    if not isinstance(temperature_overrides, list):
        raise ValueError("'temperature_overrides' must be a list")
    
    for i, override in enumerate(temperature_overrides):
        if not isinstance(override, dict):
            raise ValueError(f"Temperature override {i+1} must be a dictionary")
            
        # Check for required temperature field
        if 'temperature' not in override:
            raise ValueError(f"Temperature override {i+1} is missing required field: 'temperature'")
            
        if not isinstance(override['temperature'], (int, float)):
            raise ValueError(f"Temperature override {i+1}: 'temperature' must be a number")
            
        # Check that at least one of model or task is present
        if 'model' not in override and 'task' not in override:
            raise ValueError(f"Temperature override {i+1} must have at least one of 'model' or 'task'")
            
        # Validate types if present and ensure non-empty values
        if 'model' in override:
            if not isinstance(override['model'], str):
                raise ValueError(f"Temperature override {i+1}: 'model' must be a string")
            if len(override['model'].strip()) == 0:
                raise ValueError(f"Temperature override {i+1}: 'model' cannot be empty")
            
        if 'task' in override:
            if not isinstance(override['task'], str):
                raise ValueError(f"Temperature override {i+1}: 'task' must be a string")
            if len(override['task'].strip()) == 0:
                raise ValueError(f"Temperature override {i+1}: 'task' cannot be empty")

