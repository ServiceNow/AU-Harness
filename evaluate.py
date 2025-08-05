"""Module for evaluating language models on various benchmarks and datasets.

This module provides functionality to evaluate one or multiple models on different datasets
using various metrics, with support for concurrent evaluation and aggregation of results.
"""
import argparse
import asyncio
import importlib
import json
import logging
import math
import os
import statistics
import time
from pathlib import Path

import nest_asyncio
import yaml
from datasets import load_dataset
from tqdm import tqdm

from metrics.llm_judge import _BaseLLMJudge
from models.model import Model
from postprocessors.base import Postprocessor
from preprocessors.base import Preprocessor
from utils.constants import metric_map, metric_output
from utils.custom_logging import configure
from utils.request_manager import CentralRequestController, EngineRequestManager
from utils.util import (
    validate_config, _find_runspec_by_name,
    _find_dataset_in_runspecs, find_runspec_files
)

# Apply nest_asyncio to allow nested event loops in Azure OpenAI client calls
nest_asyncio.apply()

# Module-level logger - will be configured when setup_logging is called
logger = logging.getLogger(__name__)

class Engine:
    """Evaluate one or many models over the same dataset concurrently."""
    def __init__(self, models: list[Model], dataset_name: str, dataset_info: dict, metric_name: str,
                 filters: dict, task_type: str = None, temperature_overrides: list[dict] = None,
                 engine_id: str = None, request_manager = None, judge_properties: dict = None):

        # _load_dataset function
        repo = dataset_info.get("hf_repo", None)
        if not repo:
            repo = dataset_info.get("path", None)

        split = dataset_info.get("split", None)
        num_samples = filters.get("num_samples", None)
        user_prompt_add_ons = filters.get("user_prompt_add_ons", [])
        system_prompts = filters.get("system_prompts", [])
        length_filter = filters.get("length_filter", None)

        self.dataset, _ = _load_dataset(
            repo, num_samples=num_samples, user_prompt_add_ons=user_prompt_add_ons,
            system_prompts=system_prompts, length_filter=length_filter,
            metric=metric_name, split=split, dataset_info=dataset_info
        )

        # _load_metric function
        language = dataset_info.get("language", "en")
        self.metric = _load_metric(metric_name, language=language, judge_settings=judge_properties)

        # Load postprocessor
        postprocessor_name = dataset_info.get("postprocessor", "GeneralPostprocessor")
        postprocessor_class = get_class_from_module('postprocessors', postprocessor_name)
        if postprocessor_class is None:
            logger.warning(
            "Could not load postprocessor %s, using default GeneralPostprocessor",
            postprocessor_name
        )
            postprocessor_class = get_class_from_module('postprocessors', 'GeneralPostprocessor')
        self.postprocessor = postprocessor_class()

        logger.info(
            "[Engine.__init__] Initializing Engine with %d model(s), dataset size: %d, metric: %s",
            len(models), len(self.dataset), self.metric.name
        )

        self.models = models
        self.dataset_name = dataset_name
        self.engine_id = engine_id
        self.request_manager = request_manager
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
    async def _infer_single_model(self, model: Model, samples=None) -> list[str]:
        # Use provided samples or full dataset
        samples = samples if samples is not None else self.dataset
        task_type = self.task_type

        # Check for temperature override for this specific model and task combination
        override_temp = _get_temperature_override(
            model.name(), task_type, self.temperature_overrides
        )
        if override_temp is not None:
            # Use the override temperature directly
            logger.info(
                "[Engine._infer_single_model] Using override temperature %s "
                "for model %s and task %s",
                override_temp, model.name(), task_type
            )
            model.temperature = override_temp
            model.req_resp_hndlr.temperature = override_temp
        else:
            # Use the standard task-based temperature setting
            model.set_temp(task_type)
        # Get model type for request management
        model_type = model.model  # The actual model type (e.g., "gpt-4o-mini-audio-preview")
        # Generate a unique model instance ID
        model_instance_id = f"{model.name()}_{id(model)}"

        # Create an adjustable semaphore based on granted tokens
        token_sem = asyncio.Semaphore(0)  # Start with 0 tokens

        # Keep track of pending and completed samples
        pending_samples = list(range(len(samples)))  # Indices of samples waiting for tokens
        completed_samples = set()  # Indices of samples that have completed processing

        # Continuously ask for tokens (with backoff if none available) based on pending samples
        # Have dynamic wait times by dataset size - this "layering" of Engine priority gives models in the same Engine similar priority, so they don't wait on each other
        async def token_manager():
            request_count = 0
            # Calculate wait times based on dataset size
            dataset_size = len(samples)

            # Calculate a scale factor from 0 to 1 based on dataset size
            scale_factor = min(1.0, max(0.0, math.log10(dataset_size + 10) / 4.0))

            # Scale the no-token wait time between 0.5s and 2s
            no_token_wait = scale_factor * 2.0

            # Double the wait time when tokens are granted
            token_wait = no_token_wait * 2.0

            while len(pending_samples) > 0:
                request_count += 1
                # Request as many tokens as we need for pending samples
                request_amount = min(model.batch_size, len(pending_samples))

                if request_amount > 0:
                    granted = await self.request_manager.request_tokens(
                        model_type, model_instance_id, request_amount)

                    if granted > 0:
                        # Remove samples from pending list based on granted tokens
                        pending_samples[:] = pending_samples[granted:]
                        # Release semaphore permits for each granted token
                        for _ in range(granted):
                            token_sem.release()
                        # Wait based on dataset size when tokens were granted
                        await asyncio.sleep(token_wait)
                    else:
                        # Backoff when no tokens were granted, based on dataset size
                        # Apply a small multiplier for repeated failures, but cap it
                        backoff_multiplier = min(3.0, 1.0 + (request_count / 10))
                        await asyncio.sleep(no_token_wait * backoff_multiplier)
                else:
                    break

        asyncio.create_task(token_manager())

        # Wait for token to be available, run task, and add to completed
        async def _call_with_token_mgmt(idx: int, sample: dict):
            # Acquire a token
            await token_sem.acquire()
            try:
                # Process the sample
                resp = await model.generate_text_with_retry(sample,
                                                          {"chunk_size": model.chunk_size,
                                                           "metric": self.metric.name})
                result = resp

                # Add to completed set
                completed_samples.add(idx)

                # Return token to model's pool
                await self.request_manager.return_tokens(model_type, model_instance_id, 1)

                return idx, result
            except (RuntimeError, ValueError, ConnectionError, TimeoutError) as e:
                # Make sure to return token on error
                logger.error(
                    "[Engine._infer_single_model] Error processing sample %d in %s: %s",
                    idx, self.dataset_name, e
                )
                completed_samples.add(idx)
                await self.request_manager.return_tokens(model_type, model_instance_id, 1)
                return idx, ""

        # Create tasks paired with their original index
        tasks = [_call_with_token_mgmt(i, ex) for i, ex in enumerate(samples)]

        # Process results in order of completion
        results: list[str | None] = [None] * len(tasks)
        inference_desc = f"Inference ({self.dataset_name} | {model.name()} | {self.metric.name})"
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=inference_desc):
            idx, resp = await coro
            results[idx] = resp

        return results

    # Infer all models concurrently
    async def _infer_all(self):
        results = {}
        all_tasks = []
        task_info = {}

        # Prepare all tasks for concurrent execution
        for model_type, models in self.model_groups.items():
            if len(models) > 1:  # Multiple instances of the same model type - need sharding
                logger.info(
                    "[Engine._infer_all] Sharding dataset for %d instances of model type '%s'",
                    len(models), model_type
                )
                # Divide dataset among model instances
                shard_size = len(self.dataset) // len(models)

                # Track the mapping of original indices to shard indices for recombination
                index_mappings = {}
                sharded_tasks = {}

                # Distribute samples and create tasks
                for i, model in enumerate(models):
                    start_idx = i * shard_size
                    # Last model gets any remaining samples
                    end_idx = (i + 1) * shard_size if i < len(models) - 1 else len(self.dataset)
                    shard = self.dataset[start_idx:end_idx]

                    # Keep track of original indices
                    index_mappings[model.name()] = list(range(start_idx, end_idx))

                    task = asyncio.create_task(self._infer_single_model(model, shard))
                    sharded_tasks[model.name()] = task
                    all_tasks.append(task)

                # Store info for later reconstruction
                task_info[model_type] = {
                    "is_sharded": True,
                    "tasks": sharded_tasks,
                    "index_mappings": index_mappings
                }
            else:
                # Single instance, normal processing
                model = models[0]
                model_name = model.name()
                task = asyncio.create_task(self._infer_single_model(model))
                all_tasks.append(task)
                task_info[model_name] = {
                    "is_sharded": False,
                    "task": task
                }
        # Wait for all tasks to complete concurrently
        await asyncio.gather(*all_tasks)
        # Process results according to task type
        for key, info in task_info.items():
            if info["is_sharded"]:
                # Reconstruct sharded results
                shard_results = {name: task.result() for name, task in info["tasks"].items()}

                # Combine results under model_type as the key
                combined_results = [None] * len(self.dataset)

                # Use index mappings to put results back in correct order
                for model_name, model_results in shard_results.items():
                    original_indices = info["index_mappings"][model_name]
                    for shard_idx, orig_idx in enumerate(original_indices):
                        if shard_idx < len(model_results):
                            combined_results[orig_idx] = model_results[shard_idx]

                # Use the model_type as the key for combined results
                results[key] = combined_results
            else:
                # Single instance result
                results[key] = info["task"].result()

        return results

    async def run(self):
        """Run the evaluation engine and return scores."""
        logger.info(
            "[Engine.run] Starting evaluation run for %s with metric %s.",
            self.dataset_name, self.metric.name
        )
        raw_predictions = await self._infer_all()
        logger.info(
            "[Engine.run] Predictions complete for %s. Calculating scores...",
            self.dataset_name
        )
        scores = {}
        # Pass the metric name to the postprocessor
        process_result = self.postprocessor.process(
            dataset=self.dataset, predictions=raw_predictions, metric=self.metric.name
        )
        # Extract values from the dictionary returned by the postprocessor
        model_targets = process_result["model_targets"]
        predictions = process_result["processed_predictions"]
        instructions = process_result.get("instructions", None)
        ids = process_result.get("ids", [])
        lengths = process_result.get("lengths", [])

        # Determine if this is an LLM-judge metric
        is_llm_judge = isinstance(self.metric, _BaseLLMJudge)

        # Get the metric name from the current metric instance
        metric_name = self.metric.name

        # Get judge_properties from the metric if it's a LLM judge
        judge_settings = getattr(self.metric, '_judge_properties', None) if is_llm_judge else None

        # Get language attribute from the metric if available or default to 'en'
        language = getattr(self.metric, 'language', 'en')

        # Create metric instances for each model using _load_metric
        # _load_metric will handle deciding which parameters to use based on the metric type
        metric_instances = {
            model_name: _load_metric(metric_name, language=language, judge_settings=judge_settings)
            for model_name in predictions.keys()
        }

        async def score_model_with_tokens(model_name, outs):
            metric = metric_instances[model_name]
            model_responses = raw_predictions.get(model_name, [])
            # Check if this is an LLM judge
            if is_llm_judge:
                # For LLM judges, set the request manager
                # The metric will handle token management internally with the Engine Manager
                metric.set_request_manager(self.request_manager)

                if ids and lengths:
                    result = await metric(
                        outs, model_targets, ids, lengths,
                        instructions=instructions, dataset_name=self.dataset_name,
                        model_name=model_name, model_responses=model_responses
                    )
                else:
                    result = await metric(
                        outs, model_targets,
                        instructions=instructions, dataset_name=self.dataset_name,
                        model_name=model_name, model_responses=model_responses
                    )
                return model_name, result
            else:
                # For regular metrics, just run them directly (no token management needed)
                if ids and lengths:
                    result = await asyncio.to_thread(
                        metric, outs, model_targets, ids, lengths,
                        instructions=instructions, dataset_name=self.dataset_name,
                        model_name=model_name, model_responses=model_responses
                    )
                else:
                    result = await asyncio.to_thread(
                        metric, outs, model_targets,
                        instructions=instructions, dataset_name=self.dataset_name,
                        model_name=model_name, model_responses=model_responses
                    )
                return model_name, result

        # Run all model scoring concurrently
        tasks = [
            score_model_with_tokens(model_name, outs)
            for model_name, outs in predictions.items()
        ]
        results = await asyncio.gather(*tasks)
        for model_name, model_score in results:
            scores[model_name] = model_score
        # Return scores directly without nesting under metric name
        logger.info("[Engine.run] Evaluation complete. Returning scores.")
        return scores


def get_class_from_module(module_prefix, module_name) -> Preprocessor | Postprocessor:
    """Dynamically load a class from a module based on naming conventions."""
    try:
        # Get pre or post processor
        module_filename = ''.join(
            ['_' + c.lower() if c.isupper() else c for c in module_name]
        ).lstrip('_')
        module = importlib.import_module(f"{module_prefix}.{module_filename}")
        return getattr(module, module_name)
    except (ImportError, AttributeError, ModuleNotFoundError) as e:
        logger.warning(
            "Could not import %s from %s: %s",
            module_name, module_prefix, e
        )
        return None


def _load_callhome_dataset(repo, preprocessor_name, num_samples, properties):
    """Load and process a CallHome dataset using the specified preprocessor."""
    repo = Path(repo).resolve()
    # Dynamically load the preprocessor
    preprocessor_class = get_class_from_module('preprocessors', preprocessor_name)
    if preprocessor_class is None:
        error_msg = f"Could not load preprocessor {preprocessor_name}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    dataset = preprocessor_class().process(repo, num_samples=num_samples, properties=properties)
    dataset_size = len(dataset) if dataset else 0
    return dataset, dataset_size


def _load_dataset(
    repo=None, num_samples=None, user_prompt_add_ons=None,
    system_prompts=None, length_filter=None, metric=None, split=None, dataset_info=None
):
    """Load and preprocess a dataset from a local or remote path."""
    # Extract parameters from dataset_info
    preprocessor_name = (
        dataset_info.get("preprocessor", "GeneralPreprocessor")
        if dataset_info else "GeneralPreprocessor"
    )
    subset = dataset_info.get("subset", None) if dataset_info else None

    # Set up properties that will be passed to any preprocessor
    properties = {"metric": metric}
    if user_prompt_add_ons is None:
        user_prompt_add_ons = []
    if system_prompts is None:
        system_prompts = []
    if user_prompt_add_ons:
        properties["user_prompt_add_ons"] = user_prompt_add_ons
    if system_prompts:
        properties["system_prompts"] = system_prompts
    if length_filter:
        properties["length_filter"] = tuple(length_filter)  # Convert list to tuple
    if dataset_info:
        properties["dataset_info"] = dataset_info

    # Special handling for local CallHome dataset
    if preprocessor_name.startswith("Callhome"):
        return _load_callhome_dataset(repo, preprocessor_name, num_samples, properties)

    # For HuggingFace datasets
    if repo and (repo.startswith("/") or repo.startswith(".//")):
        repo = Path(repo).resolve()

    if split is not None:
        if isinstance(split, str):
            preferred_splits = [split]
        else:
            preferred_splits = list(split)
    else:
        preferred_splits = ["test", "data", "train"]

    # Try to load a specific split directly
    dset = None
    # Try the preferred splits in order
    token=os.getenv("HF_TOKEN")
    for split_name in preferred_splits:
        try:
            dset_args = {"path": repo, "split": split_name, "trust_remote_code": True}
            if subset:
                dset_args["name"] = subset
            if token:
                dset_args["token"] = token
            dset = load_dataset(**dset_args)
            break
        except (ImportError, ValueError, OSError, FileNotFoundError):
            pass

    # Raise an error if no valid split was found
    if dset is None:
        try:
            dset_args = {"path": repo, "trust_remote_code": True}
            if subset:
                dset_args["name"] = subset
            if token:
                dset_args["token"] = token
            dset = load_dataset(**dset_args)
        except Exception as e:
            error_msg = f"[_load_dataset] No valid dataset found in {repo}"
            logger.error(error_msg)
            raise ValueError(e) from e


    if num_samples is not None:
        dset = dset[:num_samples]
    else:
        dset = dset[:len(dset)]
    preprocessor_class = get_class_from_module('preprocessors', preprocessor_name)
    if preprocessor_class is None:
        error_msg = f"Could not load preprocessor {preprocessor_name}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    processed = preprocessor_class().process(dset, num_samples, properties)
    dataset_size = len(processed)

    # Return both the processed dataset and its size
    return processed, dataset_size


def _get_temperature_override(
    model_name: str, task_type: str, temperature_overrides: list[dict]
) -> float | None:
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

def load_models(
    cfg_list: list[dict], judge_properties: dict = None
) -> tuple[list[Model], CentralRequestController]:
    """
    Load models and initialize the central request controller.

    Args:
        cfg_list: Configuration list for models
        judge_properties: Optional judge properties

    Returns:
        Tuple of (models list, central request controller)
    """
    models = []

    # Initialize the central request controller
    central_request_controller = CentralRequestController()

    # Register all models with the controller
    for cfg in cfg_list:
        model_type = cfg["info"].get("model")
        batch_size = cfg["info"].get("batch_size", 1)

        model_obj = Model(cfg["info"])
        models.append(model_obj)

        # Register model type with the controller
        if model_type:
            central_request_controller.register_model_type(model_type, batch_size)

    # Register judge model if available
    if judge_properties:
        judge_model = judge_properties.get("judge_model")
        judge_concurrency = judge_properties.get("judge_concurrency", 1)

        if judge_model:
            central_request_controller.register_model_type(judge_model, judge_concurrency)

    if not models:
        logger.error("[_load_models] No valid models found in configuration.")
        raise ValueError("No valid models found in configuration.")

    return models, central_request_controller



# Metric Loader
def _load_metric(name: str, language: str = "en", judge_settings: dict = None):

    if name not in metric_map:
        raise ValueError(f"Unknown metric: {name}. Available metrics: {list(metric_map.keys())}")

    module_name, class_name = metric_map[name]

    try:
        # Dynamically import the module and class
        module = __import__(module_name, fromlist=[class_name])
        metric_class = getattr(module, class_name)

        # Handle metric-specific initialization parameters
        if "wer" in name.lower():
            metric = metric_class(language=language)
        elif "judge" in name.lower():
            # Extract judge settings or use empty dict if None
            judge_settings = judge_settings or {}

            # Pass all judge settings as judge_properties
            metric = metric_class(judge_properties=judge_settings)
        else:
            # Default initialization for other metrics
            metric = metric_class()

        return metric
    except (ImportError, AttributeError) as e:
        logger.error("[_load_metric] Failed to load metric %s: %s", name, e)
        raise ValueError(f"Failed to load metric {name}: {e}") from e

#main that runs
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
    with open(cfg_path, encoding='utf-8') as f:
        raw_cfg = yaml.safe_load(f)
    log_file = raw_cfg.get("logging", {}).get("log_file", "default.log")
    setup_logging(log_file)

    logger.info("[read_config] Loading config from %s", cfg_path)

    # Validate the configuration file
    try:
        cfg = validate_config(cfg_path)
        logger.info("[read_config] Config file validation successful")
    except ValueError as e:
        logger.error("[read_config] Config validation error: %s", e)
        raise

    # Convert judge_properties list of one-item dicts to a simple dict
    judge_properties = {}
    for item in cfg.get("judge_properties", []):
        if isinstance(item, dict):
            judge_properties.update(item)

    # Convert filters list of one-item dicts to a simple dict
    filters = {}
    for item in cfg.get("filters", []):
        if isinstance(item, dict):
            filters.update(item)

    temperature_overrides = cfg.get("temperature_overrides", None)

    return cfg, judge_properties, filters, temperature_overrides

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
            raise ValueError(
                f"Invalid dataset_metric pair: {pair}. "
                "Must be a list with two elements [dataset, metric]"
            )

        dataset_name, metric_name = pair
        dataset_metric_pairs.append((dataset_name, metric_name))

    # Expand each dataset-metric pair by finding runspecs
    expanded_pairs = []

    for dname, metric_name in dataset_metric_pairs:
        # Step 1: Look for a matching runspec file
        found_runspec, selected_datasets, matching_runspec_file = _find_runspec_by_name(
            dname, runspec_files
        )

        # Step 2: If no matching runspec file by name, search within all runspec files
        # for the specific dataset
        if not found_runspec:
            found_runspec, selected_datasets, matching_runspec_file = _find_dataset_in_runspecs(
                dname, runspec_files
            )

            if not found_runspec:
                logger.info("[expand_dataset_metric_pairs] Dataset not found, skipping: %s", dname)
                continue

        # Extract task_type from the matching runspec file name (stem)
        task_type = matching_runspec_file.stem if matching_runspec_file else None

        # Process each selected dataset (if whole runspec could be multiple per dname/metric pair)
        for dataset_name, dataset_info in selected_datasets.items():
            expanded_pairs.append((dataset_name, metric_name, dataset_info, task_type))

    return expanded_pairs

def _calculate_aggregates(aggregates, final_scores, models):
    """
    Process aggregate metrics by calculating means across multiple datasets for a specific metric.

    Args:
        aggregates: List of aggregate configurations from the config file in format
                   [metric_name, [dataset1, dataset2, ...]]
        final_scores: Dictionary of scores keyed by dataset_metric pairs
        models: List of model instances used for evaluation
    """
    logger.info("[calculate_aggregates] Processing aggregate metrics...")

    aggregate_scores = {}

    # Get unique model types
    model_types = set()
    for model in models:
        model_type = model.model  # The model type (e.g., "gpt-4o-mini-audio-preview")
        if model_type:
            model_types.add(model_type)

    # Load all runspec files using the utility function
    runspec_files = find_runspec_files()

    for agg_item in aggregates:
        # Skip invalid aggregates
        if not isinstance(agg_item, (list, tuple)) or len(agg_item) != 2:
            logger.warning("[calculate_aggregates] Invalid aggregate format: %s", agg_item)
            continue

        metric_name, dataset_specs = agg_item
        if not isinstance(dataset_specs, list) or not dataset_specs:
            logger.warning(
                "[calculate_aggregates] Invalid dataset specs list in aggregate for metric '%s'",
                metric_name
            )
            continue

        # Step 1: Look up metric keys from constants.py
        if metric_name not in metric_output:
            logger.warning(
                "[calculate_aggregates] Metric '%s' not found in metric_output dict",
                metric_name
            )
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
                # Add each dataset from the runspec for calculation
                processed_datasets.extend(runspec_data.keys())
            else:
                # If not a runspec file, treat as a regular dataset name
                processed_datasets.append(dataset_spec)

        if not processed_datasets:
            logger.warning(
                "[calculate_aggregates] No valid datasets found for metric '%s'",
                metric_name
            )
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
                        if (dataset_name in final_scores and
                            model_type in final_scores[dataset_name]):
                            # Direct access to metrics from final_scores
                            metrics_dict = final_scores[dataset_name][model_type]
                            dataset_size = 1  # Default size if not specified

                            # Check if this specific metric exists
                            if metric_key in metrics_dict:
                                value = metrics_dict[metric_key]
                                if isinstance(value, (int, float)):
                                    values.append(value)
                                    dataset_sizes.append(dataset_size)
                        else:
                            logger.warning(
                                "[calculate_aggregates] Dataset '%s' not found in final_scores",
                                dataset_name
                            )
                    except KeyError as e:
                        logger.warning(
                            "[calculate_aggregates] Error accessing data for %s in %s: %s",
                            model_type, dataset_name, str(e)
                        )

                # Calculate weighted average for this metric key
                if values:
                    if sum(dataset_sizes) > 0:
                        weighted_avg = (
                            sum(v * w for v, w in zip(values, dataset_sizes)) / sum(dataset_sizes)
                        )
                        model_scores[metric_key] = weighted_avg
                    else:
                        # Fallback to simple mean if weights are all zero
                        model_scores[metric_key] = statistics.mean(values)

            # Add scores for this model
            if model_scores:
                model_agg_scores[model_type] = model_scores
            else:
                logger.warning(
                    "[calculate_aggregates] No scores to aggregate for %s in '%s'",
                    model_type, metric_name
                )

        # Add aggregate scores to the results
        if model_agg_scores:
            # Create a key with metric name and original runspec/dataset names
            display_names_str = ", ".join(display_names)
            aggregate_key = f"{metric_name} - {display_names_str}"
            aggregate_scores[aggregate_key] = model_agg_scores

    # Add aggregate scores to final_scores
    if aggregate_scores:
        final_scores["aggregates"] = aggregate_scores

def create_engines(
    dataset_name, dataset_info, task_type, metric_name, filters, models,
    temperature_overrides, judge_properties, central_request_controller
):
    """
    Process a dataset and run evaluation on it.

    Args:
        dataset_name: Name of the dataset to process
        dataset_info: Dictionary containing dataset information
        task_type: Type of task for temperature settings
        metric_name: Name of the metric to use
        filters: Dictionary containing filter settings
        models: List of model instances
        temperature_overrides: List of temperature override configurations
        judge_properties: Dictionary of judge properties
        central_request_controller: The central request controller instance

    Returns:
        tuple: (Engine instance, dataset_name)
    """
    # Get needed settings directly from filters
    accented_filter = filters.get("accented", None)
    language_filter = filters.get("language", None)

    # Check if we need to filter out accented datasets
    if accented_filter is False and dataset_info.get("accented", False) is True:
        logger.info(
            "[_process_dataset] Skipping dataset '%s' because it is accented "
            "and accented filter is False",
            dataset_name
        )
        return False

    # Check if we need to filter by language
    if language_filter is not None:
        dataset_language = dataset_info.get("language", "").lower()
        if dataset_language and language_filter.lower() != dataset_language:
            logger.info(
                "[_process_dataset] Skipping dataset '%s' because its language '%s' "
                "doesn't match filter '%s'",
                dataset_name, dataset_language, language_filter
            )
            return False

    logger.info(
        "[_process_dataset] Creating engine for dataset '%s' with metric '%s' ...",
        dataset_name, metric_name
    )

    # Create engine ID and request manager
    engine_id = f"{dataset_name}_{metric_name}_{int(time.time())}"
    engine_request_manager = EngineRequestManager(engine_id, central_request_controller)

    # Create Engine - it will handle dataset/metric/postprocessor loading internally
    result = Engine(
        models=models, dataset_name=dataset_name, dataset_info=dataset_info,
        metric_name=metric_name, filters=filters, task_type=task_type,
        temperature_overrides=temperature_overrides, engine_id=engine_id,
        request_manager=engine_request_manager, judge_properties=judge_properties
    )

    return result, dataset_name

async def run_all_engines(all_engines):
    """
    Run all engines concurrently and collect results.

    Args:
        all_engines: List of (engine, dataset_name) tuples

    Returns:
        Dictionary of all scores
    """
    logger.info("[run_all_engines] Running %d engines concurrently...", len(all_engines))

    # Store all scores in a flat dict
    engine_scores = {}

    # Create tasks for each engine
    tasks = []
    for engine, dataset_name in all_engines:
        tasks.append(engine.run())

    # Run all tasks concurrently and gather results
    results = await asyncio.gather(*tasks)

    # Store results in all_scores dictionary
    for (engine, dataset_name), result in zip(all_engines, results):
        if dataset_name not in engine_scores:
            engine_scores[dataset_name] = {}

        # Result is in format: {metric_name: {model_name: scores}}
        for metric_name, model_scores in result.items():
            if metric_name not in engine_scores[dataset_name]:
                engine_scores[dataset_name][metric_name] = {}

            # Add model scores to the right metric bucket
            for model_name, scores in model_scores.items():
                engine_scores[dataset_name][metric_name][model_name] = scores
    logger.info("[run_all_engines] Evaluation complete. Final results:")
    logger.info(json.dumps(engine_scores, indent=2))
    return engine_scores



def main(cfg_path='config.yaml'):
    """Main function to run the evaluation benchmark.

    Args:
        cfg_path: Path to configuration file

    Returns:
        Dictionary containing evaluation scores
    """
    # 1. Read config and process configuration dictionaries
    cfg, judge_properties, filters, temperature_overrides = read_config(cfg_path)

    # 2. Load models and initialize central request controller
    models, central_request_controller = load_models(cfg.get("models", []), judge_properties)

    # 3. Expand dataset-metric pairs using runspecs
    expanded_pairs = expand_dataset_metric_pairs(cfg)

    # 4. Create engines for each expanded dataset-metric pair
    all_engines = []
    for dataset_name, metric_name, dataset_info, task_type in expanded_pairs:
        engine, dataset_name = create_engines(
            dataset_name=dataset_name,
            dataset_info=dataset_info,
            task_type=task_type,
            metric_name=metric_name,
            filters=filters,
            models=models,
            temperature_overrides=temperature_overrides,
            judge_properties=judge_properties,
            central_request_controller=central_request_controller
        )
        all_engines.append((engine, dataset_name))

    # 5. Run all engines concurrently
    scores = asyncio.run(run_all_engines(all_engines))

    # 6. Log final results and process aggregates
    aggregates = cfg.get("aggregate", [])
    if aggregates:
        _calculate_aggregates(aggregates, scores, models)

    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run audio evaluation benchmark')
    parser.add_argument('--config', '-c', default='config.yaml',
                        help='Path to configuration file (default: config.yaml)')
    args = parser.parse_args()

    # Pass the config path to main
    all_scores = main(cfg_path=args.config)
