
import importlib
import json
import logging
import os
import statistics
import yaml
from pathlib import Path
from typing import Any, Dict
from . import constants
from utils.custom_logging import configure

logger = logging.getLogger(__name__)

def get_class_from_module(module_prefix, module_name):
    try:
        # Convert class name (CamelCase) to filename (snake)
        # Get pre or post processor
        module_filename = ''.join(['_' + c.lower() if c.isupper() else c for c in module_name]).lstrip('_')
        module = importlib.import_module(f"{module_prefix}.{module_filename}")
        return getattr(module, module_name)
    except Exception as e:
        logger.warning(f"Could not import {module_name} from {module_prefix}: {e}")
        return None

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
        logger.warning("[find_runspec_files] Runspecs directory not found: %s", runspecs_dir)
        return []

    # Get all category directories in the runspecs directory
    category_dirs = [d for d in runspecs_dir.iterdir() if d.is_dir()]

    # Get list of all runspec files in all category directories plus root
    runspec_files = list(runspecs_dir.glob("*.json"))
    for category_dir in category_dirs:
        category_json_files = list(category_dir.glob("*.json"))
        runspec_files.extend(category_json_files)

    if not runspec_files:
        logger.warning("[find_runspec_files] No runspec files found in %s", runspecs_dir)

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
            with open(runspec_file, 'r', encoding='utf-8') as f:
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

    # Search through all runspec files to find the dataset
    for runspec_file in runspec_files:
        with open(runspec_file, 'r', encoding='utf-8') as f:
            runspec_db = json.load(f)

        if dataset_name in runspec_db:
            # Use only this specific dataset
            return True, {dataset_name: runspec_db[dataset_name]}, runspec_file

    return False, {}, None

def _get_task_type_datasets(task_type: str, runspec_files):
    """
    Get all datasets that belong to a specific task type (directory name).
    
    Args:
        task_type: Name of the task type directory (e.g., "paralinguistics")
        runspec_files: List of runspec files to search in
    
    Returns:
        Set of dataset names that belong to this task type
    """
    datasets = set()
    
    for runspec_file in runspec_files:
        # Check if this runspec file is in the specified task type directory
        if runspec_file.parent.name == task_type:
            try:
                with open(runspec_file, 'r', encoding='utf-8') as f:
                    runspec_db = json.load(f)
                # Add all dataset names from this runspec
                datasets.update(runspec_db.keys())
            except Exception as e:
                continue
    
    return datasets

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
            "Invalid precision provided: %s. Using the default precision: %s",
            precision, constants.ROUND_DIGITS
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
        with open(config_path, 'r', encoding='utf-8') as file:
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
                raise ValueError(
                    f"'dataset_metric' item {i+1} must be a list, not {type(item).__name__}"
                )

            if len(item) == 0:
                raise ValueError(f"'dataset_metric' item {i+1} must not be an empty list")

            for j, element in enumerate(item):
                if not isinstance(element, str) or not element.strip():
                    raise ValueError(
                        f"'dataset_metric' item {i+1}, element {j+1} must be a non-empty string"
                    )

        # Validate filters as a dictionary
        if 'filters' in config:
            if not isinstance(config['filters'], dict):
                raise ValueError("'filters' must be a dictionary")
            _validate_filter_values(config['filters'])
        
        # Validate judge_properties as a dictionary
        if 'judge_properties' in config:
            if not isinstance(config['judge_properties'], dict):
                raise ValueError("'judge_properties' must be a dictionary")
            _validate_judge_properties(config['judge_properties'])

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
        raise ValueError(f"Invalid YAML format: {str(e)}") from e


def _validate_filter_values(filters: Dict) -> None:
    """Validate the values in the filters dictionary.
    
    Args:
        filters: Dictionary of filter values to validate
    
    Raises:
        ValueError: If any filter value is invalid
    """
    # Validate num_samples if present
    if 'num_samples' in filters and not isinstance(filters['num_samples'], int):
        raise ValueError("'num_samples' must be an integer")
    
    # Validate user_prompt_add_ons if present
    if 'user_prompt_add_ons' in filters:
        if not isinstance(filters['user_prompt_add_ons'], list):
            raise ValueError("'user_prompt_add_ons' must be a list")
        for i, item in enumerate(filters['user_prompt_add_ons']):
            if not isinstance(item, list) or len(item) != 2:
                raise ValueError(f"'user_prompt_add_ons' item {i+1} must be a list with exactly 2 elements [key, [datasets]]")
            
            # First element must be a non-empty string (prompt key)
            if not isinstance(item[0], str) or not item[0].strip():
                raise ValueError(f"'user_prompt_add_ons' item {i+1}: first element must be a non-empty string")
            
            # Second element must be a list of dataset/runspec/category names
            if not isinstance(item[1], list):
                raise ValueError(f"'user_prompt_add_ons' item {i+1}: second element must be a list of dataset/runspec/category names")
            
            # Each dataset/runspec/category name must be a non-empty string
            for j, dataset_name in enumerate(item[1]):
                if not isinstance(dataset_name, str) or not dataset_name.strip():
                    raise ValueError(f"'user_prompt_add_ons' item {i+1}, dataset {j+1} must be a non-empty string")
    
    # Validate system_prompts if present
    if 'system_prompts' in filters:
        if not isinstance(filters['system_prompts'], list):
            raise ValueError("'system_prompts' must be a list")
        for i, item in enumerate(filters['system_prompts']):
            if not isinstance(item, list) or len(item) != 2:
                raise ValueError(f"'system_prompts' item {i+1} must be a list with exactly 2 elements")
            
            # First element must be a non-empty string (prompt key)
            if not isinstance(item[0], str) or not item[0].strip():
                raise ValueError(f"'system_prompts' item {i+1}: first element must be a non-empty string")
            
            # Second element must be a list with exactly 2 elements [model_name, dataset_name]
            if not isinstance(item[1], list) or len(item[1]) != 2:
                raise ValueError(f"'system_prompts' item {i+1}: second element must be a list with exactly 2 elements")
            
            # Both model_name and dataset_name must be non-empty strings
            for j, element in enumerate(item[1]):
                if not isinstance(element, str) or not element.strip():
                    raise ValueError(f"'system_prompts' item {i+1}, criteria {j+1} must be a non-empty string")
    
    # Validate length_filter if present
    if 'length_filter' in filters:
        if not isinstance(filters['length_filter'], list):
            raise ValueError("'length_filter' must be a list")
        filter_list = filters['length_filter']
        if len(filter_list) != 2:
            raise ValueError("'length_filter' must have exactly 2 elements")
        if not all(isinstance(value, (int, float)) for value in filter_list):
            raise ValueError("'length_filter' elements must be numbers")
    
    # Validate accented if present
    if 'accented' in filters and not isinstance(filters['accented'], bool):
        raise ValueError("'accented' must be a boolean")
    
    # Validate language if present
    if 'language' in filters and not isinstance(filters['language'], str):
        raise ValueError("'language' must be a string")


def _validate_judge_properties(judge_props: Dict) -> None:
    """Validate the values in the judge_properties dictionary.
    
    Args:
        judge_props: Dictionary of judge properties to validate
    
    Raises:
        ValueError: If any judge property is invalid
    """
    # Validate judge_concurrency if present
    if 'judge_concurrency' in judge_props and not isinstance(judge_props['judge_concurrency'], int):
        raise ValueError("'judge_concurrency' must be an integer")
    
    # Validate judge_model if present
    if 'judge_model' in judge_props and not isinstance(judge_props['judge_model'], str):
        raise ValueError("'judge_model' must be a string")
    
    # Validate judge_type if present
    if 'judge_type' in judge_props:
        if not isinstance(judge_props['judge_type'], str):
            raise ValueError("'judge_type' must be a string")
        if judge_props['judge_type'] not in ['vllm', 'openai']:
            raise ValueError("'judge_type' must be either 'vllm' or 'openai'")
    
    # Validate string properties
    string_props = ['judge_api_version', 'judge_api_endpoint', 'judge_api_key', 'judge_prompt_model_override']
    for prop in string_props:
        if prop in judge_props and not isinstance(judge_props[prop], str):
            raise ValueError(f"'{prop}' must be a string")
    
    # Validate judge_temperature if present
    if 'judge_temperature' in judge_props and not isinstance(judge_props['judge_temperature'], (int, float)):
        raise ValueError("'judge_temperature' must be a number")


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

    for dataset_pair in dataset_metric_pairs:
        # Check if this is a valid pair format
        if not isinstance(dataset_pair, list) or len(dataset_pair) != 2:
            raise ValueError(f"Invalid dataset_metric pair format: {dataset_pair}")

        dataset_name, metric_name = dataset_pair

        found_runspec, _, matching_runspec_file = _find_runspec_by_name(dataset_name, runspec_files)

        if not found_runspec:
            found_runspec, _, matching_runspec_file = _find_dataset_in_runspecs(dataset_name, runspec_files)

        if not found_runspec or not matching_runspec_file:
            # Check if it's a category (directory name) before raising error
            category_datasets = _get_task_type_datasets(dataset_name, runspec_files)
            if not category_datasets:
                raise ValueError(f"Dataset not found or no runspec file determined: {dataset_name}")
            # For categories, we need to validate each dataset individually
            # Skip validation here since expand_dataset_metric_pairs will handle the expansion
            continue

        # The task type is simply the JSON filename without extension
        task_type = matching_runspec_file.stem

        # Check if the metric is allowed for this task type
        valid_metric = False

        # Special handling for callhome datasets - allow specific metrics regardless of task type
        if dataset_name.startswith('callhome'):
            callhome_allowed_metrics = ['word_error_rate', 'diarization_metrics', 'llm_judge_detailed']
            if metric_name in callhome_allowed_metrics:
                valid_metric = True

        # Check if the task_type exists in allowed_task_metrics
        if not valid_metric and task_type in constants.allowed_task_metrics:
            allowed_metrics = constants.allowed_task_metrics[task_type]

            if metric_name in allowed_metrics:
                valid_metric = True

        if not valid_metric:
            # If the task_type doesn't exist in allowed_task_metrics or metric is not allowed
            if dataset_name.startswith('callhome'):
                allowed_metrics = ['word_error_rate', 'diarization_metrics', 'llm_judge_detailed']
            else:
                allowed_metrics = constants.allowed_task_metrics.get(task_type, [])
            raise ValueError(
                f"Invalid metric '{metric_name}' for dataset '{dataset_name}' with task type '{task_type}'. "
                f"Allowed metrics for this dataset: {sorted(allowed_metrics)}"
            )

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
            raise ValueError(
                f"Aggregate entry {i}: first element must be a non-empty string representing the metric name"
            )

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
            raise ValueError(
                f"Temperature override {i+1} must have at least one of 'model' or 'task'"
            )

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

def setup_logging(log_file: str):
    """
    Set up logging with default.log
    """
    
    # Configure logging using the custom_logging module
    configure(log_file)
    
    # Set root logger level to INFO
    logging.getLogger().setLevel(logging.INFO)
    
    # Set httpx logger to WARNING level to reduce noise
    logging.getLogger("httpx").setLevel(logging.WARNING)

def read_config(cfg_path: str):
    """
    Read configuration file, set up logging, validate config, and process config dictionaries.
    
    Args:
        cfg_path: Path to configuration file
        
    Returns:
        Tuple of (cfg, judge_properties, filters, temperature_overrides)
    """
    # Set up logging
    with open(cfg_path) as f:
        raw_cfg = yaml.safe_load(f)
    log_file = raw_cfg.get("logging", {}).get("log_file", "default.log")
    setup_logging(log_file)
    
    # Validate the configuration file
    try:
        cfg = validate_config(cfg_path)
        logger.info(f"[read_config] Config file validation successful")
    except ValueError as e:
        logger.error(f"[read_config] Config validation error: {e}")
        raise
    
    # Convert judge_properties list of one-item dicts to a simple dict
    judge_properties = cfg.get("judge_properties", {})
    filters = cfg.get("filters", {})
    temperature_overrides = cfg.get("temperature_overrides", None)
    aggregates = cfg.get("aggregate", [])

    
    return cfg, judge_properties, filters, temperature_overrides, aggregates
    
def expand_dataset_metric_pairs(cfg: dict) -> list[tuple[str, str, dict, str]]:
    """
    Expand dataset-metric pairs from config, finding runspecs and expanding them.
    
    Args:
        cfg: Configuration dictionary
        
    Returns:
        List of tuples (dataset_name, metric_name, dataset_info, task_type)
    """
    # Load runspec files using the utility function
    runspec_files = find_runspec_files()
    
    # Get dataset-metric pairs from config.yaml
    dataset_metric_pairs = []
    for pair in cfg.get("dataset_metric", []):
        # Validate the pair format - should be a list with two elements
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"Invalid dataset_metric pair: {pair}. Must be a list with two elements [dataset, metric]")
        
        dataset_name, metric_name = pair
        dataset_metric_pairs.append((dataset_name, metric_name))

    # Expand each dataset-metric pair by finding runspecs
    expanded_pairs = []
    
    for dname, metric_name in dataset_metric_pairs:
        # Step 1: Look for a matching runspec file
        found_runspec, selected_datasets, matching_runspec_file = _find_runspec_by_name(dname, runspec_files)
        
        # Step 2: If no matching runspec file by name, search within all runspec files for the specific dataset
        if not found_runspec:
            found_runspec, selected_datasets, matching_runspec_file = _find_dataset_in_runspecs(dname, runspec_files)
            
            # Step 3: If still not found, check if it's a category (directory name)
            if not found_runspec:
                category_datasets = _get_task_type_datasets(dname, runspec_files)
                if category_datasets:
                    # Process all datasets in this category
                    for dataset_name in category_datasets:
                        # Find the specific dataset info and runspec file
                        _, dataset_dict, dataset_runspec_file = _find_dataset_in_runspecs(dataset_name, runspec_files)
                        if dataset_dict and dataset_runspec_file:
                            task_type = dataset_runspec_file.stem
                            for ds_name, ds_info in dataset_dict.items():
                                expanded_pairs.append((ds_name, metric_name, ds_info, task_type))
                    continue
                else:
                    logger.info(f"[expand_dataset_metric_pairs] Dataset/runspec/category not found, skipping: {dname}")
                    continue
        
        # Extract task_type from the matching runspec file name (stem)
        task_type = matching_runspec_file.stem if matching_runspec_file else None
        
        # Process each selected dataset (if whole runspec could be multiple per dname/metric pair)
        for dataset_name, dataset_info in selected_datasets.items():
            expanded_pairs.append((dataset_name, metric_name, dataset_info, task_type))
    
    return expanded_pairs

def _calculate_aggregates(aggregates, all_scores, model_configs):
    """
    Process aggregate metrics by calculating means across multiple datasets for a specific metric.
    
    Args:
        aggregates: List of aggregate configurations from the config file in format [metric_name, [dataset1, dataset2, ...]]
        all_scores: Dictionary of scores keyed by dataset_metric pairs
        model_configs: List of model configurations used for evaluation
    """
    logger.info("[calculate_aggregates] Processing aggregate metrics...")

    aggregate_scores = {}
    
    # Get unique model types
    model_types = set()
    for model_config in model_configs:
        model_type = model_config["info"].get("model")  # The model type (e.g., "gpt-4o-mini-audio-preview")
        if model_type:
            model_types.add(model_type)
        
    # Load all runspec files using the utility function
    runspec_files = find_runspec_files()
    
    for agg_item in aggregates:
        # Skip invalid aggregates
        if not isinstance(agg_item, (list, tuple)) or len(agg_item) != 2:
            logger.warning(f"[calculate_aggregates] Invalid aggregate format: {agg_item}")
            continue
            
        metric_name, dataset_specs = agg_item
        if not isinstance(dataset_specs, list) or not dataset_specs:
            logger.warning(f"[calculate_aggregates] Invalid dataset specs list in aggregate for metric '{metric_name}'")
            continue
        
        # Step 1: Look up metric keys from constants.py
        if metric_name not in constants.metric_output:
            logger.warning(f"[calculate_aggregates] Metric '{metric_name}' not found in metric_output dict")
            continue
        
        metric_keys = constants.metric_output[metric_name]
        
        # Step 2: Process each dataset/runspec entry
        processed_datasets = []  # For actual calculations
        display_names = []  # For display purposes (runspecs or dataset names)
        
        for dataset_spec in dataset_specs:
            # Check if this is a runspec file name rather than a dataset name
            found_runspec, runspec_data, _ = _find_runspec_by_name(dataset_spec, runspec_files)
            
            # Always add the dataset/runspec name for display
            display_names.append(dataset_spec)
            
            if found_runspec:
                # Add each dataset from the runspec for calculation
                processed_datasets.extend(runspec_data.keys())
            else:
                # If not a runspec file, treat as a regular dataset name
                processed_datasets.append(dataset_spec)
        
        if not processed_datasets:
            logger.warning(f"[calculate_aggregates] No valid datasets found for metric '{metric_name}'")
            continue
        
        # Step 3: Calculate aggregates for each model using the metric keys
        model_agg_scores = {}
        
        for model_type in model_types:
            model_scores = {}
            
            # For each metric key, collect values across all datasets
            for metric_key in metric_keys:
                values = []
                dataset_sizes = []
                
                # Process each dataset
                for dataset_name in processed_datasets:
                    try:
                        # Check if this dataset and model combination exists
                        if dataset_name in all_scores and model_type in all_scores[dataset_name]:
                            # Direct access to metrics from all_scores
                            metrics_dict = all_scores[dataset_name][model_type]
                            dataset_size = 1  # Default size if not specified
                            
                            # Check if this specific metric exists
                            if metric_key in metrics_dict:
                                value = metrics_dict[metric_key]
                                if isinstance(value, (int, float)):
                                    values.append(value)
                                    dataset_sizes.append(dataset_size)
                        else:
                            logger.warning(f"[calculate_aggregates] Dataset '{dataset_name}' not found in all_scores")
                    except KeyError as e:
                        logger.warning(f"[calculate_aggregates] Error accessing data for {model_type} in {dataset_name}: {str(e)}")
                
                # Calculate weighted average for this metric key
                if values:
                    if sum(dataset_sizes) > 0:
                        weighted_avg = sum(v * w for v, w in zip(values, dataset_sizes)) / sum(dataset_sizes)
                        model_scores[metric_key] = weighted_avg
                    else:
                        # Fallback to simple mean if weights are all zero
                        model_scores[metric_key] = statistics.mean(values)
            
            # Add scores for this model
            if model_scores:
                model_agg_scores[model_type] = model_scores
            else:
                logger.warning(f"[calculate_aggregates] No scores to aggregate for {model_type} in '{metric_name}'")
        
        # Add aggregate scores to the results
        if model_agg_scores:
            # Create a key with metric name and original runspec/dataset names
            display_names_str = ", ".join(display_names)
            aggregate_key = f"{metric_name} - {display_names_str}"
            aggregate_scores[aggregate_key] = model_agg_scores
    
    # Add aggregate scores to all_scores
    if aggregate_scores:
        all_scores["aggregates"] = aggregate_scores