from utils.custom_logging import configure
configure()
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Apply nest_asyncio to allow nested event loops in Azure OpenAI client calls
import nest_asyncio
nest_asyncio.apply()
from postprocessors.base import Postprocessor
from preprocessors.base import Preprocessor
import os
import asyncio
import importlib
import argparse
import json
import yaml
from datasets import load_dataset
from pathlib import Path
import time
from tqdm import tqdm
import logging
from utils.request_manager import CentralRequestController, EngineRequestManager
from models.model import Model
from metrics.metrics import Metrics
from postprocessors.base import Postprocessor
from utils.constants import metric_map
from metrics.llm_judge import _BaseLLMJudge

class Engine:
    """Evaluate one or many models over the same dataset concurrently."""
    def __init__(self, models: list[Model], dataset: list[dict], metric: Metrics, postprocessor, dataset_name: str, engine_id: str, request_manager, available_judge_calls: int = None):
        logger.info(f"[Engine.__init__] Initializing Engine with {len(models)} model(s), dataset size: {len(dataset)}, metric: {metric.name}")
        self.models = models
        self.dataset = dataset
        self.metric = metric
        self.postprocessor = postprocessor
        self.dataset_name = dataset_name
        self.available_judge_calls = available_judge_calls  # For LLM-judge concurrency splitting
        self.engine_id = engine_id
        self.request_manager = request_manager
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
        samples = samples if samples is not None else self.dataset  # Use provided samples or full dataset
        
        # Get model type for request management
        model_type = model.model  # The actual model type (e.g., "gpt-4o-mini-audio-preview")
        # Generate a unique model instance ID
        model_instance_id = f"{model.name()}_{id(model)}"
        
        # Create an adjustable semaphore based on granted tokens
        token_sem = asyncio.Semaphore(0)  # Start with 0 tokens
        
        # Keep track of pending and completed samples
        pending_samples = list(range(len(samples)))  # Indices of samples waiting for tokens
        completed_samples = set()  # Indices of samples that have completed processing
        
        # Continuously ask for tokens(with backoff if none available) based on pending samples
        async def token_manager():
            request_count = 0
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
                        # Short wait if we got tokens
                        await asyncio.sleep(0.5)
                    else:
                        # Backoff if we didn't get any tokens
                        await asyncio.sleep(0.1 * min(request_count, 10))
                else:
                    break
        
        asyncio.create_task(token_manager())
            
        # Wait for token to be available, run task, and add to completed
        async def _call_with_token_mgmt(idx: int, sample: dict):
            # Acquire a token
            await token_sem.acquire()
            try:
                # Process the sample
                resp = await model._generate_text_with_retry(sample, 
                                                          {"chunk_size": model.chunk_size, 
                                                           "metric": self.metric.name})
                result = resp.llm_response if resp else ""
                
                # Add to completed set
                completed_samples.add(idx)
                
                # Return token to model's pool
                await self.request_manager.return_tokens(model_type, model_instance_id, 1)
                                
                return idx, result
            except Exception as e:
                # Make sure to return token on error
                logger.error(f"[Engine._infer_single_model] Error processing sample {idx} in {self.dataset_name}: {e}")
                completed_samples.add(idx)
                await self.request_manager.return_tokens(model_type, model_instance_id, 1)
                return idx, ""
        
        # Create tasks paired with their original index
        tasks = [_call_with_token_mgmt(i, ex) for i, ex in enumerate(samples)]
        
        # Process results in order of completion
        results: list[str | None] = [None] * len(tasks)
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), 
                          desc=f"Inference ({self.dataset_name} | {model.name()} | {self.metric.name})"):
            idx, text = await coro
            results[idx] = text
            
        return results

    # Infer all models concurrently
    async def _infer_all(self):
        results = {}
        all_tasks = []
        task_info = {}

        # Prepare all tasks for concurrent execution
        for model_type, models in self.model_groups.items():
            if len(models) > 1:  # Multiple instances of the same model type - need sharding
                logger.info(f"[Engine._infer_all] Sharding dataset for {len(models)} instances of model type '{model_type}'")
                # Divide dataset among model instances
                shard_size = len(self.dataset) // len(models)
                
                # Track the mapping of original indices to shard indices for recombination
                index_mappings = {}
                sharded_tasks = {}
                
                # Distribute samples and create tasks
                for i, model in enumerate(models):
                    start_idx = i * shard_size
                    # Last model gets any remaining samples
                    end_idx = (i+1) * shard_size if i < len(models)-1 else len(self.dataset)
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
        logger.info(f"[Engine.run] Starting evaluation run for {self.dataset_name} with metric {self.metric.name}.")
        predictions = await self._infer_all()
        logger.info(f"[Engine.run] Predictions complete for {self.dataset_name}. Calculating scores...")
        scores = {}
        # Pass the metric name to the postprocessor
        process_result = self.postprocessor.process(dataset=self.dataset, predictions=predictions, metric=self.metric.name)
        # Extract values from the dictionary returned by the postprocessor
        model_targets = process_result["model_targets"]
        predictions = process_result["processed_predictions"]
        instructions = process_result.get("instructions", None)
        ids = process_result.get("ids", [])
        lengths = process_result.get("lengths", [])

        # Determine if this is an LLM-judge metric
        is_llm_judge = isinstance(self.metric, _BaseLLMJudge)
        # For LLM-judge, split concurrency
        if is_llm_judge and self.available_judge_calls:
            num_models = len(predictions)
            per_model_conc = max(1, self.available_judge_calls // num_models)
            # Get the judge model from the original metric
            judge_model = getattr(self.metric, '_model', None)
            
            # Instantiate a separate metric for each model with correct concurrency and preserving the judge model
            metric_instances = {
                model_name: type(self.metric)(max_concurrency=per_model_conc, model=judge_model)
                for model_name in predictions.keys()
            }
        else:
            metric_instances = {model_name: self.metric for model_name in predictions.keys()}
        
        async def score_model_with_tokens(model_name, outs):
            metric = metric_instances[model_name]
            
            # Check if this is an LLM judge
            if is_llm_judge:
                # For LLM judges, set the request manager
                # The metric will handle token management internally with the Engine Manager
                metric.set_request_manager(self.request_manager)
                
                if ids and lengths:
                    result = await metric(outs, model_targets, ids, lengths,
                                        instructions=instructions, dataset_name=self.dataset_name, model_name=model_name)
                else:
                    result = await metric(outs, model_targets,
                                        instructions=instructions, dataset_name=self.dataset_name, model_name=model_name)
                return model_name, result
            else:
                # For regular metrics, just run them directly (no token management needed)
                if ids and lengths:
                    result = await asyncio.to_thread(
                        metric, outs, model_targets, ids, lengths,
                        instructions=instructions, dataset_name=self.dataset_name, model_name=model_name
                    )
                else:
                    result = await asyncio.to_thread(
                        metric, outs, model_targets,
                        instructions=instructions, dataset_name=self.dataset_name, model_name=model_name
                    )
                return model_name, result

        # Run all model scoring concurrently
        tasks = [score_model_with_tokens(model_name, outs) for model_name, outs in predictions.items()]
        results = await asyncio.gather(*tasks)
        for model_name, model_score in results:
            scores[model_name] = model_score
        logger.info(f"[Engine.run] Evaluation complete. Returning scores.")
        logger.info(f"[Engine.run] Scores: {scores}")
        return {self.metric.name: scores}




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
    return dataset

def _load_dataset(repo=None, subset=None, num_samples=None, preprocessor_name="GeneralPreprocessor", user_prompt_add_ons: list[str] = [], system_prompts: list[str] = [], length_filter=None, metric=None, split=None, dataset_info=None, modality=None):
    """Load and preprocess a dataset from a local or remote path."""
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
    if modality:
        properties["modality"] = modality
    
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
            args = {"path": repo, "split": split_name, "trust_remote_code": True}
            if subset:
                args["name"] = subset
            if token:
                args["token"] = token
            dset = load_dataset(**args)
            break
        except Exception as e:
            pass
    
    # Raise an error if no valid split was found
    if dset is None:
        try:
            args = {"path": repo, "trust_remote_code": True}
            if subset:
                args["name"] = subset
            if token:
                args["token"] = token
            dset = load_dataset(**args)
        except Exception as e:
            error_msg = f"[_load_dataset] No valid dataset found in {repo}"
            logger.error(error_msg)
            raise ValueError(e)
    
    logger.info(f"[_load_dataset] Dataset loaded  |  Size before trunc: {len(dset)}")


    if num_samples is not None:
        logger.info(f"[_load_dataset] Truncating dataset to first {num_samples} samples.")
        dset = dset[:num_samples]
    else:
        dset = dset[:len(dset)]
    PreprocessorClass = get_class_from_module('preprocessors', preprocessor_name)
    if PreprocessorClass is None:
        error_msg = f"Could not load preprocessor {preprocessor_name}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    processed = PreprocessorClass().process(dset, num_samples, properties)
    logger.info(f"[_load_dataset] Dataset loaded and processed. Size: {len(processed)}")
    return processed



# Model Loader
def _load_models(cfg_list: list[dict]) -> list[Model]:
    models = []
    for cfg in cfg_list:
        model_obj = Model(cfg["info"])
        models.append(model_obj)
    if not models:
        raise ValueError("No valid models found in configuration.")
    for model in models:
        logger.info(f"Loaded {model.name()}")
    return models



# Metric Loader
def _load_metric(name: str, language: str = "en", judge_concurrency: int | None = None, judge_model: str | None = None):

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



def main(cfg_path='config.yaml'):
    logger.info(f"[main] Loading config from {cfg_path}")
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
        
    central_request_controller = CentralRequestController()
    logger.info("[main] Initialized CentralRequestController")
    
    # Register all models with the controller
    if "models" in cfg:
        for model_cfg in cfg.get("models", []):
            model_type = model_cfg["info"].get("model") 
            batch_size = model_cfg["info"].get("batch_size", 1)
            if model_type:
                logger.info(f"[main] Registering model type '{model_type}' with batch size {batch_size}")
                central_request_controller.register_model_type(model_type, batch_size)
    
    # Register judge model for LLM-judge evaluations
    judge_model = cfg.get("judge_model")
    judge_concurrency = cfg.get("judge_concurrency", 1)  
    if judge_model:
        logger.info(f"[main] Registering judge model '{judge_model}' with concurrency {judge_concurrency}")
        central_request_controller.register_model_type(judge_model, judge_concurrency)
    
    # Load runspec files from the runspecs directory
    runspecs_dir = Path("runspecs")
    
    # Get list of all runspec files in the runspecs directory
    runspec_files = list(runspecs_dir.glob("*.json"))
    
    # Load metric and model settings
    judge_concurrency = cfg.get("judge_concurrency", 1)
    judge_model = cfg.get("judge_model", None)
    user_prompt_add_ons = cfg.get("user_prompt_add_ons", [])
    system_prompts = cfg.get("system_prompts", [])
    length_filter = cfg.get("length_filter", None)
    num_samples = cfg.get("num_samples", None)
    
    # Load models
    models = _load_models(cfg.get("models", []))
    
    if len(models) == 0:
        raise ValueError(f"No models found in {cfg_path}")
    
    # Get dataset-metric pairs from config.yaml
    dataset_metric_pairs = []
    for pair_str in cfg.get("dataset_metric", []):
        # Remove parentheses and split by comma
        pair_str = pair_str.strip().strip("()").strip()
        items = [x.strip() for x in pair_str.split(",")]
        if len(items) != 2:
            raise ValueError(f"Invalid dataset_metric pair: {pair_str}. Must be in format '(dataset, metric)'")
        dataset_name, metric_name = items
        dataset_metric_pairs.append((dataset_name, metric_name))
        
    # Store all scores in a flat dict with keys in format: 'dataset_name_metric_name'
    all_scores = {}
    flattened_dataset_metric_pairs = []  # Will hold individual dataset-metric pairs after expansion
    
    # Helper function to check if a dataset passes filters
    def dataset_passes_filters(dataset_name, dataset_info, cfg):
        # Check for accented filter setting
        accented_filter = cfg.get("accented", None)
        if accented_filter is not None and accented_filter is False and dataset_info.get("accented", False) is True:
            logger.info(f"[main] Skipping dataset '{dataset_name}' because it is accented and accented filter is False")
            return False
            
        # Check for language filter setting
        language_filter = cfg.get("language", None)
        if language_filter is not None:
            dataset_language = dataset_info.get("language", "").lower()
            if dataset_language and language_filter.lower() != dataset_language:
                logger.info(f"[main] Skipping dataset '{dataset_name}' because its language '{dataset_language}' doesn't match filter '{language_filter}'")
                return False
                
        return True
    
    # Flatten dataset-metric pairs from config/runspecs
    for dataset_name, metric_name in dataset_metric_pairs:
        
        # Look for a matching runspec file
        found_runspec = False
        
        for runspec_file in runspec_files:
            runspec_name = runspec_file.stem
            
            # First, check if dataset name exactly matches the runspec file name
            if dataset_name == runspec_name:
                found_runspec = True
                
                # Load the runspec file
                with open(runspec_file, 'r') as f:
                    runspec_db = json.load(f)
                
                # Expand all datasets in this runspec
                for expanded_dataset_name, dataset_info in runspec_db.items():
                    # Add this dataset-metric pair to our flattened list if it passes filters
                    if dataset_passes_filters(expanded_dataset_name, dataset_info, cfg):
                        flattened_dataset_metric_pairs.append((expanded_dataset_name, metric_name, dataset_info))
                        logger.info(f"[main] Added dataset-metric pair: ({expanded_dataset_name}, {metric_name})")
                        
                break
        
        # If no matching runspec file by name, search within all runspec files for the specific dataset
        if not found_runspec:            
            # Search through all runspec files to find the dataset
            for runspec_file in runspec_files:
                with open(runspec_file, 'r') as f:
                    runspec_db = json.load(f)
                
                if dataset_name in runspec_db:
                    dataset_info = runspec_db[dataset_name]
                    
                    # Add this dataset-metric pair to our flattened list if it passes filters
                    if dataset_passes_filters(dataset_name, dataset_info, cfg):
                        flattened_dataset_metric_pairs.append((dataset_name, metric_name, dataset_info))
                        logger.info(f"[main] Added dataset-metric pair: ({dataset_name}, {metric_name})")
                        found_runspec = True
                        break
            
            if not found_runspec:
                logger.warning(f"[main] Dataset '{dataset_name}' not found in any runspec file")

    # Report the final count of dataset-metric pairs
    
    all_engines = []  # Track all engines for concurrent execution
    
    # Build engines for each dataset-metric pair
    for dataset_name, metric_name, dataset_info in flattened_dataset_metric_pairs:
        logger.info(f"[main] Creating engine for dataset '{dataset_name}' with metric '{metric_name}'")
        
        # Load dataset information
        repo = dataset_info.get("hf_repo", None)
        if not repo:
            repo = dataset_info.get("path", None)
            
        subset = dataset_info.get("subset", "")
        language = dataset_info.get("language", "en")
        preprocessor_name = dataset_info["preprocessor"]
        postprocessor_name = dataset_info["postprocessor"]
        modality = dataset_info.get("modality", "audio")
        
        # Determine the split
        split = None
        if cfg.get("split", None) is not None:
            split = cfg.get("split")
        if dataset_info.get("split", None) is not None:
            split = dataset_info["split"]

        # Load and preprocess the dataset
        dataset = _load_dataset(
            repo, subset=subset, 
            num_samples=num_samples, 
            preprocessor_name=preprocessor_name, 
            user_prompt_add_ons=user_prompt_add_ons, 
            system_prompts=system_prompts, 
            length_filter=length_filter, 
            metric=metric_name, 
            split=split, 
            dataset_info=dataset_info,
            modality=modality
        )
        metric = _load_metric(metric_name, language=language, judge_concurrency=judge_concurrency, judge_model=judge_model)
        
        # Dynamically import postprocessor class
        PostprocessorClass = get_class_from_module('postprocessors', postprocessor_name)
        if PostprocessorClass is None:
            logger.warning(f"Could not load postprocessor {postprocessor_name}, using default GeneralPostprocessor")
            PostprocessorClass = get_class_from_module('postprocessors', 'GeneralPostprocessor')
        postprocessor = PostprocessorClass()
        
        # Set up engine request manager
        engine_id = f"{dataset_name}_{metric.name}_{int(time.time())}"
        logger.info(f"[main] Setting up EngineRequestManager for {engine_id}")
        engine_request_manager = EngineRequestManager(engine_id, central_request_controller)
        
        # Create Engine with request manager
        engine = Engine(
            models=models,
            dataset=dataset,
            metric=metric,
            postprocessor=postprocessor,
            dataset_name=dataset_name,
            available_judge_calls=judge_concurrency,
            engine_id=engine_id,
            request_manager=engine_request_manager
        )
        
        # Add the engine to our collection for concurrent execution
        all_engines.append((engine, dataset_name))
        logger.info(f"[main] Created engine for dataset: {dataset_name}")
        
    # Now run all engines concurrently
    logger.info(f"[main] Running {len(all_engines)} engines concurrently...")
    
    async def run_all_engines():
        # Create tasks for each engine
        tasks = []
        for engine, dataset_name in all_engines:
            logger.info(f"[main] Creating task for engine: {dataset_name}")
            tasks.append(engine.run())
        
        # Run all tasks concurrently and gather results
        results = await asyncio.gather(*tasks)
        
        # Store results in all_scores dictionary
        for (engine, dataset_name), result in zip(all_engines, results):
            if dataset_name not in all_scores:
                all_scores[dataset_name] = {}
                
            # Result is in format: {metric_name: {model_name: scores}}
            for metric_name, model_scores in result.items():
                if metric_name not in all_scores[dataset_name]:
                    all_scores[dataset_name][metric_name] = {}
                    
                # Add model scores to the right metric bucket
                for model_name, scores in model_scores.items():
                    all_scores[dataset_name][metric_name][model_name] = scores
                    
            logger.info(f"[main] Processed results for dataset: {dataset_name}")
            
        return all_scores
    
    # Run the async function
    all_scores = asyncio.run(run_all_engines())
    
    # Log the final results
    logger.info(f"[main] Evaluation complete. Final results:")
    logger.info(json.dumps(all_scores, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run audio evaluation benchmark')
    parser.add_argument('--config', '-c', default='config.yaml', 
                        help='Path to configuration file (default: config.yaml)')
    args = parser.parse_args()
    
    # Pass the config path to main
    main(cfg_path=args.config)
