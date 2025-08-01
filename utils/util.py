
from typing import Any, Dict
import os
import yaml
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
from pathlib import Path
import logging
import json
from . import constants

def find_runspec_files(base_dir="runspecs"):
    """
    Find all runspec JSON files in the base directory and its subdirectories.
    
    Args:
        base_dir: Base directory to search in, defaults to "runspecs"
        
    Returns:
        List of Path objects pointing to runspec JSON files
    """
    runspecs_dir = Path(base_dir)
    if not runspecs_dir.exists():
        logger.warning(f"[find_runspec_files] Runspecs directory not found: {runspecs_dir}")
        return []
        
    # Get all category directories in the runspecs directory
    category_dirs = [d for d in runspecs_dir.iterdir() if d.is_dir()]
    
    # Get list of all runspec files in all category directories plus root
    runspec_files = list(runspecs_dir.glob("*.json"))
    for category_dir in category_dirs:
        category_json_files = list(category_dir.glob("*.json"))
        runspec_files.extend(category_json_files)
    
    if not runspec_files:
        logger.warning(f"[find_runspec_files] No runspec files found in {runspecs_dir}")
        
    return runspec_files

def _find_runspec_by_name(dataset_name, runspec_files):
    """
    Find a runspec file by exact name match.
    
    Args:
        dataset_name: Name of the dataset to find
        runspec_files: List of runspec files to search in
        
    Returns:
        tuple: (found_runspec, selected_datasets, matching_file)
    """
    for runspec_file in runspec_files:
        runspec_name = runspec_file.stem
        
        # Check if dataset name exactly matches the runspec file name
        if dataset_name == runspec_name:            
            # Load the runspec file
            with open(runspec_file, 'r') as f:
                runspec_db = json.load(f)
            
            # Use all datasets in this runspec
            return True, runspec_db, runspec_file
    
    return False, {}, None

def _find_dataset_in_runspecs(dataset_name, runspec_files):
    """
    Search for a dataset within all runspec files.
    
    Args:
        dataset_name: Name of the dataset to find
        runspec_files: List of runspec files to search in
        
    Returns:
        tuple: (found_runspec, selected_datasets, matching_file)
    """
    logger.info(f"[find_dataset_in_runspecs] No matching runspec file for '{dataset_name}'. Searching within individual runspec files...")
    
    # Search through all runspec files to find the dataset
    for runspec_file in runspec_files:
        with open(runspec_file, 'r') as f:
            runspec_db = json.load(f)
        
        if dataset_name in runspec_db:
            logger.info(f"[find_dataset_in_runspecs] Found dataset '{dataset_name}' in {runspec_file}")
            # Use only this specific dataset
            return True, {dataset_name: runspec_db[dataset_name]}, runspec_file
    
    logger.warning(f"[find_dataset_in_runspecs] Dataset '{dataset_name}' not found in any runspec file")
    return False, {}, None

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
        
        # Check required sections
        if 'dataset_metric' not in config:
            raise ValueError("'dataset_metric' is required")
            
        if not isinstance(config['dataset_metric'], list):
            raise ValueError("'dataset_metric' must be a list")
            
        # Validate dataset_metric structure
        dataset_metric = config['dataset_metric']
        if len(dataset_metric) == 0:
            raise ValueError("'dataset_metric' must have at least one element")
            
        for i, item in enumerate(dataset_metric):
            if not isinstance(item, list):
                raise ValueError(f"'dataset_metric' item {i+1} must be a list, not {type(item).__name__}")
            
            if len(item) == 0:
                raise ValueError(f"'dataset_metric' item {i+1} must not be an empty list")
            
            for j, element in enumerate(item):
                if not isinstance(element, str) or not element.strip():
                    raise ValueError(f"'dataset_metric' item {i+1}, element {j+1} must be a non-empty string")
        
        # Check optional sections with simple type validations
        type_validations = {
            'num_samples': int,
            'judge_concurrency': int,
            'judge_model': str,
            'accented': bool,
            'language': str
        }
        
        for field, expected_type in type_validations.items():
            if field in config and not isinstance(config[field], expected_type):
                raise ValueError(f"'{field}' must be a {expected_type.__name__}")
        
        # Validate user_prompt_add_ons and system_prompts if present
        for field in ['user_prompt_add_ons', 'system_prompts']:
            if field in config:
                if not isinstance(config[field], list):
                    raise ValueError(f"'{field}' must be a list")
                    
                for i, item in enumerate(config[field]):
                    if not isinstance(item, str):
                        raise ValueError(f"'{field}' item {i+1} must be a string, not {type(item).__name__}")
        
        # Validate length_filter if present
        if 'length_filter' in config:
            if not isinstance(config['length_filter'], list):
                raise ValueError("'length_filter' must be a list")
                
            filter_list = config['length_filter']
            if len(filter_list) != 2:
                raise ValueError("'length_filter' must have exactly 2 elements")
                
            if not all(isinstance(value, (int, float)) for value in filter_list):
                raise ValueError("'length_filter' elements must be numbers")
        
        # Delegate validation for complex sections
        _validate_models(config)
        
        if 'aggregate' in config:
            _validate_aggregate(config['aggregate'])
        
        if 'temperature_overrides' in config:
            _validate_temperature_overrides(config['temperature_overrides'])
            
        # Validate dataset-metric pairs against allowed task metrics
        _validate_dataset_metric_pairs(config.get('dataset_metric', []))
        
        return config
        
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML format: {str(e)}")


def _validate_dataset_metric_pairs(dataset_metric_pairs):
    """
    Validate that each dataset's associated metric is allowed for its task type.
    
    Args:
        dataset_metric_pairs: List of [dataset_name, metric_name] pairs from the config
    """
    # Find all runspec files using the helper function
    runspec_files = find_runspec_files()
    if not runspec_files:
        return
        
    logger.info(f"[validate_config] Validating dataset-metric pairs against {len(runspec_files)} runspec files")
    
    for dataset_pair in dataset_metric_pairs:
        # Check if this is a valid pair format
        if not isinstance(dataset_pair, list) or len(dataset_pair) != 2:
            raise ValueError(f"Invalid dataset_metric pair format: {dataset_pair}")
            
        dataset_name, metric_name = dataset_pair
        
        found_runspec, runspec_data, matching_runspec_file = _find_runspec_by_name(dataset_name, runspec_files)
        
        if not found_runspec:
            found_runspec, runspec_data, matching_runspec_file = _find_dataset_in_runspecs(dataset_name, runspec_files)
        
        if not found_runspec or not matching_runspec_file:
            raise ValueError("Dataset not found or no runspec file determined")
        
        # The task type is simply the JSON filename without extension
        task_type = matching_runspec_file.stem
        
        # Check if the metric is allowed for this task type
        valid_metric = False
        
        # Check if the task_type exists in allowed_task_metrics
        if task_type in constants.allowed_task_metrics:
            allowed_metrics = constants.allowed_task_metrics[task_type]
            
            if metric_name in allowed_metrics:
                valid_metric = True
        
        if not valid_metric:
            # If the task_type doesn't exist in allowed_task_metrics or metric is not allowed
            allowed_metrics = constants.allowed_task_metrics.get(task_type, [])
            raise ValueError(f"Invalid metric '{metric_name}' for dataset '{dataset_name}' with task type '{task_type}'. " 
                             f"Allowed metrics for this task type: {sorted(allowed_metrics)}")

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
      - ["metric_name", ["dataset1", "dataset2", ...]]
    """
    # Validate top-level structure is a list
    if not isinstance(aggregate_section, list):
        raise ValueError("'aggregate' must be a list")
    
    # Validate each aggregate entry
    for i, entry in enumerate(aggregate_section, start=1):
        # Check entry is a list with exactly 2 elements
        if not isinstance(entry, list) or len(entry) != 2:
            raise ValueError(f"Aggregate entry {i} must be a list with exactly 2 elements")
        
        # First element must be a non-empty string (metric name)
        if not isinstance(entry[0], str) or not entry[0].strip():
            raise ValueError(f"Aggregate entry {i}: first element must be a non-empty string representing the metric name")
        
        # Second element must be a list of dataset names
        if not isinstance(entry[1], list):
            raise ValueError(f"Aggregate entry {i}: second element must be a list of dataset names")
        
        # Validate each dataset name in the list
        for j, dataset in enumerate(entry[1], start=1):
            if not isinstance(dataset, str) or not dataset.strip():
                raise ValueError(f"Aggregate entry {i}, dataset {j} must be a non-empty string")


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

