import os
import yaml
import sys
from pathlib import Path
import logging
from utils.custom_logging import configure
from postprocessors.base import Postprocessor
from preprocessors.base import Preprocessor
import json
import asyncio
from datasets import load_dataset
from tqdm import tqdm
import importlib
import argparse
import statistics
# Central logging setup
from models.model import Model
from metrics.metrics import Metrics
from postprocessors.base import Postprocessor
from utils.constants import metric_map, metric_output
from utils.util import validate_config, _find_runspec_by_name, _find_dataset_in_runspecs, find_runspec_files


# Create logger at module level
logger = logging.getLogger(__name__)
# Removed duplicate imports that were moved to the top

class Engine:
    """Evaluate one or many models over the same dataset concurrently."""
    def __init__(self, models: list[Model], dataset: list[dict], metric: Metrics, postprocessor, dataset_name: str, task_type: str = None, temperature_overrides: list[dict] = None):
        logger.info(f"[Engine.__init__] Initializing Engine with {len(models)} model(s), dataset size: {len(dataset)}, metric: {metric.name}")
        self.models = models
        self.dataset = dataset
        self.metric = metric
        self.postprocessor = postprocessor
        # Keep track of dataset name so we can create per-dataset log files
        self.dataset_name = dataset_name
        # Store task_type for temperature setting
        self.task_type = task_type
        # Store temperature overrides
        self.temperature_overrides = temperature_overrides or []
        
        # Group models by their model attribute for sharding
        self.model_groups = {}
        for model in models:
            model_type = model.model  # The model attribute we're sharding on
            if model_type not in self.model_groups:
                self.model_groups[model_type] = []
            self.model_groups[model_type].append(model)

    # ---------------- internal helpers ----------------x
    # infer by batch size, calling generate text with retry for each sample
    async def _infer_single_model(self, model: Model, samples=None) -> list:
        samples = samples if samples is not None else self.dataset  # Use provided samples or full dataset
        logger.info(f"[Engine._infer_single_model] Running model: {model.name()} on dataset of size {len(samples)}")
        
        # Set temperature based on the task_type if provided
        task_type = self.task_type
        
        # Check for temperature override for this specific model and task combination
        override_temp = _get_temperature_override(model.name(), task_type, self.temperature_overrides)
        
        if override_temp is not None:
            # Use the override temperature directly
            logger.info(f"[Engine._infer_single_model] Using override temperature {override_temp} for model {model.name()} and task {task_type}")
            model.temperature = override_temp
            model.req_resp_hndlr.temperature = override_temp
        else:
            # Use the standard task-based temperature setting
            model.set_temp(task_type)
        sem = asyncio.Semaphore(model.batch_size)  # Use per-model batch size

        async def _call(idx: int, sample: dict):
            async with sem:
                resp = await model._generate_text_with_retry(sample, {"chunk_size": model.chunk_size,
                                                                      "metric": self.metric.name})
                return idx, resp

        # Create tasks paired with their original index
        tasks = [_call(i, ex) for i, ex in enumerate(samples)]
        results: list[str | None] = [None] * len(tasks)
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Inference ({model.name()})"):
            idx, resp = await coro
            results[idx] = resp
        logger.info(f"[Engine._infer_single_model] Model {model.name()} finished inference.")
        return results

    # Infer all models concurrently
    async def _infer_all(self):
        logger.info(f"[Engine._infer_all] Starting inference for all models: {[m.name() for m in self.models]}")
        results = {}

        # Process each unique model type (for sharding)
        for model_type, models in self.model_groups.items():
            if len(models) > 1:  # Multiple instances of the same model type - need sharding
                logger.info(
                    f"[Engine._infer_all] Sharding dataset for {len(models)} instances of model type '{model_type}'")
                # Divide dataset among model instances
                shard_size = len(self.dataset) // len(models)
                tasks = {}

                # Track the mapping of original indices to shard indices for recombination
                index_mappings = {}

                # Distribute samples and create tasks
                for i, model in enumerate(models):
                    start_idx = i * shard_size
                    # Last model gets any remaining samples
                    end_idx = (i + 1) * shard_size if i < len(models) - 1 else len(self.dataset)
                    shard = self.dataset[start_idx:end_idx]

                    # Keep track of original indices
                    index_mappings[model.name()] = list(range(start_idx, end_idx))

                    tasks[model.name()] = asyncio.create_task(self._infer_single_model(model, shard))
                    logger.info(
                        f"[Engine._infer_all] Model {model.name()} assigned {len(shard)} samples (indices {start_idx}-{end_idx - 1})")

                # Wait for all sharded tasks to complete
                shard_results = {name: await t for name, t in tasks.items()}

                # Combine results under model_type as the key
                combined_results = [None] * len(self.dataset)

                # Use index mappings to put results back in correct order
                for model_name, model_results in shard_results.items():
                    original_indices = index_mappings[model_name]
                    for shard_idx, orig_idx in enumerate(original_indices):
                        if shard_idx < len(model_results):
                            combined_results[orig_idx] = model_results[shard_idx]

                # Use the model_type as the key for combined results
                results[model_type] = combined_results
                logger.info(f"[Engine._infer_all] Combined results for {len(models)} instances of '{model_type}'")
            else:
                # Single instance, normal processing
                model = models[0]
                model_name = model.name()
                results[model_name] = await self._infer_single_model(model)

        logger.info(f"[Engine._infer_all] All models finished inference.")
        return results

    def run(self):
        logger.info("[Engine.run] Starting evaluation run.")
        model_responses_by_model = asyncio.run(self._infer_all())
        logger.info(f"[Engine.run] Predictions complete. Calculating scores...")
        scores = {}
        
        # Pass raw model responses directly to the postprocessor - it will handle different response types internally
        process_result = self.postprocessor.process(dataset=self.dataset, predictions=model_responses_by_model,
                                                    metric=self.metric.name)

        # Extract values from the dictionary returned by the postprocessor
        model_targets = process_result["model_targets"]
        predictions = process_result["processed_predictions"]
        instructions = process_result.get("instructions", None)
        ids = process_result.get("ids", [])
        lengths = process_result.get("lengths", [])
        if (self.metric.name == 'comet'):
            source_sentences = process_result['source_sentences']

        for model_name, outs in predictions.items():
            # Reset the metric's record_level_scores before each model evaluation
            self.metric.reset()
            
            # Let the metric handle per-record logging internally
            # Pass the full ModelResponse objects to the metric
            model_responses = model_responses_by_model.get(model_name, [])
            if ids and lengths:
                model_score = self.metric(outs, model_targets, ids, lengths, instructions=instructions, 
                                         dataset_name=self.dataset_name, model_name=model_name, 
                                         model_responses=model_responses)

            # Add extra source_sentences argument to compute COMET for translation tasks
            elif (self.metric.name == 'comet'):
                model_score = self.metric(outs, model_targets, source_sentences, instructions=instructions, 
                                         dataset_name=self.dataset_name, model_name=model_name, 
                                         model_responses=model_responses)
            else:
                model_score = self.metric(outs, model_targets, instructions=instructions, 
                                         dataset_name=self.dataset_name, model_name=model_name, 
                                         model_responses=model_responses)
            scores[model_name] = model_score
        logger.info(f"[Engine.run] Evaluation complete. Returning scores.")
        logger.info(f"[Engine.run] Scores: {scores}")
        # Return scores directly without nesting under metric name
        return scores


def get_class_from_module(module_prefix, module_name) -> Preprocessor | Postprocessor:
    try:
        # Convert class name (CamelCase) to filename (snake)
        # Get pre or post processor
        module_filename = ''.join(['_' + c.lower() if c.isupper() else c for c in module_name]).lstrip('_')
        module = importlib.import_module(f"{module_prefix}.{module_filename}")
        return getattr(module, module_name)
    except Exception as e:
        logger.warning(f"Could not import {module_name} from {module_prefix}: {e}")
        return None


def _load_callhome_dataset(repo, preprocessor_name, num_samples, properties):
    """Load and process a CallHome dataset using the specified preprocessor."""
    repo = Path(repo).resolve()
    logger.info(f"[_load_callhome_dataset] Loading CallHome local dataset from {repo}")
    # Dynamically load the preprocessor
    PreprocessorClass = get_class_from_module('preprocessors', preprocessor_name)
    if PreprocessorClass is None:
        error_msg = f"Could not load preprocessor {preprocessor_name}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    dataset = PreprocessorClass().process(repo, num_samples=num_samples, properties=properties)
    dataset_size = len(dataset) if dataset else 0
    return dataset, dataset_size


def _load_dataset(repo=None, num_samples=None, user_prompt_add_ons: list[str] = [], 
                  system_prompts: list[str] = [], length_filter=None, metric=None, split=None, dataset_info=None):
    """Load and preprocess a dataset from a local or remote path."""
    # Extract parameters from dataset_info
    preprocessor_name = dataset_info.get("preprocessor", "GeneralPreprocessor") if dataset_info else "GeneralPreprocessor"
    subset = dataset_info.get("subset", None) if dataset_info else None
    
    logger.info(f"[_load_dataset] Loading dataset {repo} with preprocessor {preprocessor_name}")

    # Set up properties that will be passed to any preprocessor
    properties = {"metric": metric}
    if user_prompt_add_ons:
        properties["user_prompt_add_ons"] = user_prompt_add_ons
    if system_prompts:
        properties["system_prompts"] = system_prompts
    if length_filter:
        logger.info(f"[_load_dataset] Applying length filter: {length_filter}")
        properties["length_filter"] = tuple(length_filter)  # Convert list to tuple
    if dataset_info:
        properties["dataset_info"] = dataset_info

    # Special handling for local CallHome dataset
    if preprocessor_name.startswith("Callhome"):
        return _load_callhome_dataset(repo, preprocessor_name, num_samples, properties)

    # For HuggingFace datasets
    if repo and (repo.startswith("/") or repo.startswith(".//")):
        repo = Path(repo).resolve()
        logger.info(f"[_load_dataset] Loading local dataset from {repo}")

    logger.info(f"[_load_dataset] Loading HuggingFace dataset repo: {repo}")
    # Determine the preferred split to load directly (more efficient)
    if split is not None:
        logger.info(f"[_load_dataset] Using user-specified split: {split}")
        if isinstance(split, str):
            preferred_splits = [split]
        else:
            preferred_splits = list(split)
    else:
        preferred_splits = ["test", "data", "train"]

    # Try to load a specific split directly
    dset = None
    # Try the preferred splits in order
    token = os.getenv("HF_TOKEN")
    logger.info(f"[_load_dataset] Using token: {token}")
    for split_name in preferred_splits:
        try:
            args = {"path": repo, "split": split_name, "trust_remote_code": True}
            if subset:
                args["name"] = subset
                logger.info(f"[_load_dataset] Attempting to load subset: {subset}, split: {split_name}")
            else:
                logger.info(f"[_load_dataset] Attempting to load split: {split_name}")
            if token:
                args["token"] = token
            dset = load_dataset(**args)
            logger.info(f"[_load_dataset] Successfully loaded split: {split_name}")
            break
        except Exception as e:
            logger.info(f"[_load_dataset] Split {split_name} not available: {e}")

    # Raise an error if no valid split was found
    if dset is None:
        logger.info(f"[_load_dataset] Attempting to load no split")
        try:
            args = {"path": repo, "trust_remote_code": True}
            if subset:
                args["name"] = subset
                logger.info(f"[_load_dataset] Attempting to load subset: {subset}")
            else:
                logger.info(f"[_load_dataset] Attempting to load dataset without subset")
            if token:
                args["token"] = token
            dset = load_dataset(**args)
            logger.info(f"[_load_dataset] Successfully loaded dataset without specific split")
        except Exception as e:
            logger.info(f"[_load_dataset] Failed to load dataset: {str(e)}")
            error_msg = f"[_load_dataset] No valid dataset found in {repo}"
            logger.error(error_msg)
            raise ValueError(e)

    logger.info(f"[_load_dataset] Dataset loaded  |  Size before trunc: {len(dset)}")

    if num_samples is not None:
        logger.info(f"[_load_dataset] Truncating dataset to first {num_samples} samples.")
        dset = dset[:num_samples]
    else:
        dset = dset[:len(dset)]
    logger.info(f"[_load_dataset] Preprocessing dataset using {preprocessor_name}...")
    PreprocessorClass = get_class_from_module('preprocessors', preprocessor_name)
    if PreprocessorClass is None:
        error_msg = f"Could not load preprocessor {preprocessor_name}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    processed = PreprocessorClass().process(dset, num_samples, properties)
    dataset_size = len(processed)
    logger.info(f"[_load_dataset] Dataset loaded and processed. Size: {dataset_size}")
    
    # Return both the processed dataset and its size
    return processed, dataset_size


def _get_temperature_override(model_name: str, task_type: str, temperature_overrides: list[dict]) -> float | None:
    """Check if there's a temperature override for this model and task combination.
    
    Args:
        model_name: The name of the model
        task_type: The type of task being performed
        temperature_overrides: List of override dictionaries from config.yaml
        
    Returns:
        The override temperature if found, None otherwise
    """
    if not temperature_overrides:
        return None
        
    for override in temperature_overrides:
        # Get the temperature value if present
        temp = override.get("temperature")
        if temp is None:
            continue
            
        # Check if this override applies to our model/task
        override_model = override.get("model")
        override_task = override.get("task")
        
        # Case 1: Model+Task specific override
        if override_model == model_name and override_task == task_type:
            return float(temp)
            
        # Case 2: Model-only override
        if override_model == model_name and not override_task:
            return float(temp)
            
        # Case 3: Task-only override
        if not override_model and override_task == task_type:
            return float(temp)
    
    return None

def _load_models(cfg_list: list[dict]) -> list[Model]:
    logger.info(f"[_load_models] Instantiating models from config: {cfg_list}")
    models = []
    for cfg in cfg_list:
        model_name = cfg["info"].get("name")
        logger.info(f"[_load_models] Instantiating model for {model_name}")
        model_obj = Model(cfg["info"], 0.7)
        models.append(model_obj)
    if not models:
        logger.info("[_load_models] ERROR: No valid models found in configuration.")
        raise ValueError("No valid models found in configuration.")
    for model in models:
        logger.info(f"Loaded {model.name()}")
    logger.info(f"[_load_models] Successfully instantiated {len(models)} model(s).")
    return models


def _load_metric(name: str, language: str = "en", judge_concurrency: int | None = None, judge_model: str | None = None):
    logger.info(
        f"[_load_metric] Loading metric: {name} (judge_concurrency={judge_concurrency}, judge_model={judge_model})")

    if name not in metric_map:
        raise ValueError(f"Unknown metric: {name}. Available metrics: {list(metric_map.keys())}")

    module_name, class_name = metric_map[name]

    try:
        # Dynamically import the module and class
        module = __import__(module_name, fromlist=[class_name])
        MetricClass = getattr(module, class_name)

        # Handle metric-specific initialization parameters
        if "wer" in name.lower():
            metric = MetricClass(language=language)
        elif "judge" in name.lower():
            metric = MetricClass(max_concurrency=judge_concurrency, model=judge_model)
        else:
            # Default initialization for other metrics
            metric = MetricClass()

        logger.info(f"[_load_metric] Metric loaded: {metric.name}")
        return metric
    except (ImportError, AttributeError) as e:
        logger.error(f"[_load_metric] Failed to load metric {name}: {e}")
        raise ValueError(f"Failed to load metric {name}: {e}")



#TO-DO: need to implement command line override, add common configs, group by task type
#main that runs
def setup_logging(log_config=None, log_file=None):
    """
    Set up logging with configuration from config file or specified log file
    
    Args:
        log_config: Dictionary containing logging configuration (optional)
        log_file: Path to log file (optional, overrides log_config if provided)
    """
    # Get log file path and level from config if provided
    if log_config is not None and isinstance(log_config, dict):
        file_path = log_file or log_config.get("log_file", "default.log")
        log_level = log_config.get("level", "INFO")
    else:
        file_path = log_file or "default.log"
        log_level = "INFO"
    
    # Configure logging using the custom_logging module
    configure(file_path)
    
    # Set root logger level
    logging_level = getattr(logging, log_level.upper()) if hasattr(logging, log_level.upper()) else logging.INFO
    logging.getLogger().setLevel(logging_level)
    
    # Set httpx logger to WARNING level to reduce noise
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    logger.info(f"Logging setup complete - using file: {file_path}, level: {log_level}")

def _calculate_aggregates(aggregates, all_scores, models):
    """
    Process aggregate metrics by calculating means across multiple datasets for a specific metric.
    
    Args:
        aggregates: List of aggregate configurations from the config file in format [metric_name, [dataset1, dataset2, ...]]
        all_scores: Dictionary of scores keyed by dataset_metric pairs
        models: List of model instances used for evaluation
    """
    logger.info("[calculate_aggregates] Processing aggregate metrics...")
    aggregate_scores = {}
    models_list = [model.name() for model in models]
    
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
        if metric_name not in metric_output:
            logger.warning(f"[calculate_aggregates] Metric '{metric_name}' not found in metric_output dict")
            continue
        
        metric_keys = metric_output[metric_name]
        
        # Step 2: Process each dataset/runspec entry
        processed_datasets = []  # For actual calculations
        display_names = []  # For display purposes (runspecs or dataset names)
        
        for dataset_spec in dataset_specs:
            # Check if this is a runspec file name rather than a dataset name
            found_runspec, runspec_data, _ = _find_runspec_by_name(dataset_spec, runspec_files)
            
            # Always add the dataset/runspec name for display
            display_names.append(dataset_spec)
            
            if found_runspec:
                logger.info(f"[calculate_aggregates] Found runspec file for '{dataset_spec}'")
                # Add each dataset from the runspec for calculation
                processed_datasets.extend(runspec_data.keys())
            else:
                logger.info(f"[calculate_aggregates] Found dataset file for '{dataset_spec}'")
                # If not a runspec file, treat as a regular dataset name
                processed_datasets.append(dataset_spec)
        
        if not processed_datasets:
            logger.warning(f"[calculate_aggregates] No valid datasets found for metric '{metric_name}'")
            continue
        
        # Step 3: Calculate aggregates for each model using the metric keys
        model_agg_scores = {}
        
        for model_name in models_list:
            model_scores = {}
            
            # For each metric key, collect values across all datasets
            for metric_key in metric_keys:
                values = []
                dataset_sizes = []
                
                # Process each dataset
                for dataset_name in processed_datasets:
                    key = f"{dataset_name}_{metric_name}"
                    try:
                        # Access scores for this dataset and metric
                        score_data = all_scores[key]
                        dataset_size = score_data.get("dataset_size", 1)  # Get dataset size, default to 1
                        model_dict = score_data["result"][model_name]
                        
                        # Check if the specific metric key exists in this dataset's results
                        if metric_key in model_dict:
                            value = model_dict[metric_key]
                            if isinstance(value, (int, float)):
                                values.append(value)
                                dataset_sizes.append(dataset_size)
                    except KeyError as e:
                        logger.warning(f"[calculate_aggregates] Error accessing data for {model_name} in {key}: {str(e)}")
                
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
                model_agg_scores[model_name] = model_scores
            else:
                logger.warning(f"[calculate_aggregates] No scores to aggregate for {model_name} in '{metric_name}'")
        
        # Add aggregate scores to the results
        if model_agg_scores:
            # Create a key with metric name and original runspec/dataset names
            display_names_str = ", ".join(display_names)
            aggregate_key = f"{metric_name} - {display_names_str}"
            aggregate_scores[aggregate_key] = model_agg_scores
            logger.info(f"[calculate_aggregates] Created aggregate '{aggregate_key}' with {len(processed_datasets)} datasets")
    
    # Add aggregate scores to all_scores
    if aggregate_scores:
        logger.info(f"[calculate_aggregates] Final aggregate scores: {json.dumps(aggregate_scores, indent=2)}")
        all_scores["aggregates"] = aggregate_scores

def _process_dataset_and_evaluate(dataset_name, dataset_info, metric_name, cfg, models, all_scores):
    """
    Process a dataset and run evaluation on it.
    
    Args:
        dataset_name: Name of the dataset to process
        dataset_info: Dictionary containing dataset information
        metric_name: Name of the metric to use
        cfg: Configuration dictionary
        models: List of model instances
        all_scores: Dictionary to store evaluation results
        
    Returns:
        bool: True if evaluation was performed, False if dataset was skipped
    """
    # Get needed settings directly from cfg
    accented_filter = cfg.get("accented", None)
    language_filter = cfg.get("language", None)
    num_samples = cfg.get("num_samples", None)
    user_prompt_add_ons = cfg.get("user_prompt_add_ons", [])
    system_prompts = cfg.get("system_prompts", [])
    length_filter = cfg.get("length_filter", None)
    judge_concurrency = cfg.get("judge_concurrency", 1)
    judge_model = cfg.get("judge_model", None)
    
    # Check if we need to filter out accented datasets
    if accented_filter is False and dataset_info.get("accented", False) is True:
        logger.info(f"[_process_dataset] Skipping dataset '{dataset_name}' because it is accented and accented filter is False")
        return False
        
    # Check if we need to filter by language
    if language_filter is not None:
        dataset_language = dataset_info.get("language", "").lower()
        if dataset_language and language_filter.lower() != dataset_language:
            logger.info(f"[_process_dataset] Skipping dataset '{dataset_name}' because its language '{dataset_language}' doesn't match filter '{language_filter}'")
            return False
    
    logger.info(f"[_process_dataset] Loading dataset '{dataset_name}' with metric '{metric_name}' ...")
    
    # Extract dataset parameters
    repo = dataset_info.get("hf_repo", None)
    split = None
    if not repo:
        repo = dataset_info.get("path", None)
    subset = dataset_info.get("subset", "")
    language = dataset_info.get("language", "en")
    preprocessor_name = dataset_info["preprocessor"]
    postprocessor_name = dataset_info["postprocessor"]

    if cfg.get("split", None) is not None:
        split = cfg.get("split")

    if dataset_info.get("split", None) is not None:
        split = dataset_info["split"]

    
    # Load dataset, metric, and postprocessor
    dataset, dataset_size = _load_dataset(repo, num_samples=num_samples, user_prompt_add_ons=user_prompt_add_ons, 
                                    system_prompts=system_prompts, length_filter=length_filter, metric=metric_name, split=split, dataset_info=dataset_info)
    metric = _load_metric(metric_name, language=language, judge_concurrency=judge_concurrency, judge_model=judge_model)
    
    # Dynamically import postprocessor class
    PostprocessorClass = get_class_from_module('postprocessors', postprocessor_name)
    if PostprocessorClass is None:
        logger.warning(f"Could not load postprocessor {postprocessor_name}, using default GeneralPostprocessor")
        # Try to load the default postprocessor
        PostprocessorClass = get_class_from_module('postprocessors', 'GeneralPostprocessor')
    postprocessor = PostprocessorClass()
    
    logger.info("[_process_dataset] Initializing Engine and running evaluation...")
    result = Engine(models, dataset, metric, postprocessor, dataset_name).run()
    key = f"{dataset_name}_{metric_name}"
    # Store both the result and dataset size together
    all_scores[key] = {"result": result, "dataset_size": dataset_size}
    
    return True


def main(cfg_path='config.yaml'):
    # Load config without validation first to get logging settings
    with open(cfg_path) as f:
        raw_cfg = yaml.safe_load(f)
    
    # Set up logging using the enhanced setup_logging function
    # This encapsulates all the logging configuration in one call
    setup_logging(log_config=raw_cfg.get("logging"))
    
    logger.info(f"[main] Loading config from {cfg_path}")
    
    # Validate the configuration file
    try:
        logger.info(f"[main] Validating config file: {cfg_path}")
        cfg = validate_config(cfg_path)
        logger.info(f"[main] Config file validation successful")
    except ValueError as e:
        logger.error(f"[main] Config validation error: {e}")
        raise
    
    # Load runspec files using the utility function
    runspec_files = find_runspec_files()

    # Load models
    logger.info(f"[main] Loading models...")
    models = _load_models(cfg.get("models", []))
    logger.info(f"[main] Loaded {len(models)} model(s).")

    if len(models) == 0:
        raise ValueError(f"No models found in {cfg_path}")

    # Get dataset-metric pairs from config.yaml
    dataset_metric_pairs = []
    for pair in cfg.get("dataset_metric", []):
        # Validate the pair format - should be a list with two elements
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"Invalid dataset_metric pair: {pair}. Must be a list with two elements [dataset, metric]")
        
        dataset_name, metric_name = pair
        dataset_metric_pairs.append((dataset_name, metric_name))

    logger.info(f"[main] Dataset-metric pairs from config: {dataset_metric_pairs}")

    # Store all scores in a flat dict with keys in format: 'dataset_name_metric_name'
    all_scores = {}

    # Process each dataset-metric pair
    for dname, metric_name in dataset_metric_pairs:
        logger.info(f"[main] Processing dataset '{dname}' with metric '{metric_name}' ...")

        # Step 1: Look for a matching runspec file
        found_runspec, selected_datasets, _ = _find_runspec_by_name(dname, runspec_files)
        
        # Step 2: If no matching runspec file by name, search within all runspec files for the specific dataset
        if not found_runspec:
            found_runspec, selected_datasets, _ = _find_dataset_in_runspecs(dname, runspec_files)
            
            if not found_runspec:
                logger.info(f"[main] Dataset not found, skipping: {dname}")
                continue
        
        # Process each selected dataset(if whole runspec could be multiple per dname/metric pair)
        for dataset_name, dataset_info in selected_datasets.items():
            # Process this dataset and evaluate
            _process_dataset_and_evaluate(dataset_name, dataset_info, metric_name, cfg, models, all_scores)
    
    logger.info("[main] Evaluation scores:")
    logger.info(json.dumps(all_scores, indent=2))
    
    # Process aggregate metrics if present in config
    aggregates = cfg.get("aggregate", [])
    if aggregates:
        logger.info("[main] Processing aggregate metrics...")
        _calculate_aggregates(aggregates, all_scores, models)
    
    return all_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run audio evaluation benchmark')
    parser.add_argument('--config', '-c', default='config.yaml',
                        help='Path to configuration file (default: config.yaml)')
    args = parser.parse_args()

    # Pass the config path to main
    all_scores = main(cfg_path=args.config)
    logger.info("[main] Evaluation complete.")
