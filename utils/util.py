
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
    rounded_off_val = round(val * 10 ** precision) / 10 ** precision
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
    """Validate configuration file against expected structure and types."""
    if not os.path.exists(config_path):
        raise ValueError(f"Config file does not exist: {config_path}")
        
    try:
        # Load and validate YAML file
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            
        if config is None:
            raise ValueError(f"Config file is empty: {config_path}")
        
        # Define validation functions
        def validate_required_section(section_name: str, expected_type: type, extra_check=None) -> None:
            if section_name not in config:
                raise ValueError(f"'{section_name}' is required")
            if not isinstance(config[section_name], expected_type):
                raise ValueError(f"'{section_name}' must be a {expected_type.__name__}")
            if extra_check:
                extra_check(config[section_name], section_name)
        
        def validate_optional_section(section_name: str, expected_type: type, extra_check=None) -> None:
            if section_name in config:
                if not isinstance(config[section_name], expected_type):
                    raise ValueError(f"'{section_name}' must be a {expected_type.__name__}")
                if extra_check:
                    extra_check(config[section_name], section_name)
        
        def validate_non_empty_list(items: list, section_name: str) -> None:
            if len(items) == 0:
                raise ValueError(f"'{section_name}' must have at least one element")
        
        def validate_list_entries_are_non_empty_strings(items: list, section_name: str) -> None:
            for i, entry in enumerate(items, start=1):
                if not entry or not isinstance(entry, str) or not entry.strip():
                    raise ValueError(f"{section_name} entry {i} must be a non-empty string")
        
        def validate_length_filter(filter_list: list, section_name: str) -> None:
            if len(filter_list) != 2:
                raise ValueError(f"'{section_name}' must have exactly 2 elements")
            if not all(isinstance(value, (int, float)) for value in filter_list):
                raise ValueError(f"'{section_name}' elements must be numbers")

        def validate_list_of_strings(items: list, section_name: str) -> None:
            for i, item in enumerate(items, start=1):
                if not isinstance(item, str):
                    raise ValueError(f"'{section_name}' item {i} must be a string, not {type(item).__name__}")

        # Required sections
        validate_required_section('dataset_metric', list, 
                                lambda x, name: validate_non_empty_list(x, name) or 
                                validate_list_entries_are_non_empty_strings(x, name))

        # Optional sections with simple type validation
        type_validations = {
            'num_samples': int,
            'judge_concurrency': int,
            'judge_model': str,
            'accented': bool,
            'language': str
        }
        
        for field, field_type in type_validations.items():
            validate_optional_section(field, field_type)
        

        validate_optional_section('user_prompt_add_ons', list, 
                                lambda x, name: validate_list_of_strings(x, name) if x else None)
        validate_optional_section('system_prompts', list, 
                                lambda x, name: validate_list_of_strings(x, name) if x else None)
        
        validate_optional_section('length_filter', list,
                                lambda x, name: validate_length_filter(x, name))
        
        # Delegate validation for complex sections
        _validate_models(config)
        
        if 'aggregate' in config:
            _validate_aggregate(config['aggregate'])
        
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
    def validate_required_fields(info: Dict, index: int) -> None:
        required_fields = ['name', 'model', 'inference_type', 'url']
        for field in required_fields:
            if not info.get(field) or not isinstance(info[field], str) or not info[field].strip():
                raise ValueError(f"Model {index}: '{field}' must be a non-empty string")
    def validate_optional_fields(info: Dict, index: int) -> None:
        optional_fields = {
            'delay': int, 'retry_attempts': int, 'timeout': int,
            'auth_token': str, 'api_version': str, 'batch_size': int, 'chunk_size': int
        }
        for field, field_type in optional_fields.items():
            if field in info and not isinstance(info[field], field_type):
                raise ValueError(f"Model {index}: '{field}' must be of type {field_type.__name__}")
    if 'models' not in config or not isinstance(config['models'], list):
        raise ValueError("'models' section is required and must be a list")
    for i, model_entry in enumerate(config['models'], start=1):
        if not isinstance(model_entry, dict) or 'info' not in model_entry:
            raise ValueError(f"Model entry {i} must have an 'info' object")
        info = model_entry['info']
        validate_required_fields(info, i)
        validate_optional_fields(info, i)


def _validate_aggregate(aggregate_section) -> None:
    """
    Validate the aggregate section of the configuration.
    
    Structure should be:
    aggregate:
      - ["name", [["dataset1", "metric1"], ["dataset2", "metric2"]]]
    """
    def validate_entry_structure(entry: list, entry_idx: int) -> None:
        if not isinstance(entry, list) or len(entry) != 2:
            raise ValueError(f"Aggregate entry {entry_idx} must be a list with exactly 2 elements")
            
        if not isinstance(entry[0], str) or not entry[0].strip():
            raise ValueError(f"Aggregate entry {entry_idx}: first element must be a non-empty string name")
            
        if not isinstance(entry[1], list):
            raise ValueError(f"Aggregate entry {entry_idx}: second element must be a list of dataset-metric pairs")
            
    def validate_dataset_metric_pair(pair: list, entry_idx: int, pair_idx: int) -> None:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"Aggregate entry {entry_idx}, pair {pair_idx} must be a list with exactly 2 elements")
        
        if not all(isinstance(item, str) and item.strip() for item in pair):
            raise ValueError(f"Aggregate entry {entry_idx}, pair {pair_idx}: both dataset and metric must be non-empty strings")
            
    if not isinstance(aggregate_section, list):
        raise ValueError("'aggregate' must be a list")
    
    for i, aggregate_entry in enumerate(aggregate_section, start=1):
        validate_entry_structure(aggregate_entry, i)
        
        dataset_metrics = aggregate_entry[1]
        for j, pair in enumerate(dataset_metrics, start=1):
            validate_dataset_metric_pair(pair, i, j)


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

