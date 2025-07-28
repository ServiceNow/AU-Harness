from utils.custom_logging import configure
configure()
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
from postprocessors.base import Postprocessor
from preprocessors.base import Preprocessor
import os
import json
from pathlib import Path
import asyncio
from datasets import load_dataset
import yaml
from tqdm import tqdm
import importlib
import argparse
# Central logging setup
from models.model import Model
from metrics.metrics import Metrics
from postprocessors.base import Postprocessor
from utils.constants import metric_map
from metrics.llm_judge import _BaseLLMJudge

class Engine:
    """Evaluate one or many models over the same dataset concurrently."""
    def __init__(self, models: list[Model], dataset: list[dict], metric: Metrics, postprocessor, dataset_name: str, available_judge_calls: int = None):
        logger.info(f"[Engine.__init__] Initializing Engine with {len(models)} model(s), dataset size: {len(dataset)}, metric: {metric.name}")
        self.models = models
        self.dataset = dataset
        self.metric = metric
        self.postprocessor = postprocessor
        self.dataset_name = dataset_name
        self.available_judge_calls = available_judge_calls  # For LLM-judge concurrency splitting
        # Group models by their model attribute for sharding
        self.model_groups = {}
        for model in models:
            model_type = model.model  # The model attribute we're sharding on
            if model_type not in self.model_groups:
                self.model_groups[model_type] = []
            self.model_groups[model_type].append(model)
        # Log model grouping information
        for model_type, group_models in self.model_groups.items():
            if len(group_models) > 1:
                logger.info(f"[Engine.__init__] Model type '{model_type}' has {len(group_models)} instances - will shard dataset")


    # ---------------- internal helpers ----------------x
    #infer by batch size, calling generate text with retry for each sample
    async def _infer_single_model(self, model: Model, samples=None) -> list[str]:
        samples = samples if samples is not None else self.dataset  # Use provided samples or full dataset
        logger.info(f"[Engine._infer_single_model] Running model: {model.name()} on dataset of size {len(samples)}")
        sem = asyncio.Semaphore(model.batch_size)  # Use per-model batch size
        async def _call(idx: int, sample: dict):
            async with sem:
                resp = await model._generate_text_with_retry(sample, {"chunk_size": model.chunk_size, "metric": self.metric.name})
                return idx, (resp.llm_response if resp else "")
        # Create tasks paired with their original index
        tasks = [_call(i, ex) for i, ex in enumerate(samples)]
        results: list[str | None] = [None] * len(tasks)
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Inference ({model.name()})"):
            idx, text = await coro
            results[idx] = text
        logger.info(f"[Engine._infer_single_model] Model {model.name()} finished inference.")
        return results

    # Infer all models concurrently
    async def _infer_all(self):
        logger.info(f"[Engine._infer_all] Starting inference for all models: {[m.name() for m in self.models]}")
        results = {}
        
        # Process each unique model type (for sharding)
        for model_type, models in self.model_groups.items():
            if len(models) > 1:  # Multiple instances of the same model type - need sharding
                logger.info(f"[Engine._infer_all] Sharding dataset for {len(models)} instances of model type '{model_type}'")
                # Divide dataset among model instances
                shard_size = len(self.dataset) // len(models)
                tasks = {}
                
                # Track the mapping of original indices to shard indices for recombination
                index_mappings = {}
                
                # Distribute samples and create tasks
                for i, model in enumerate(models):
                    start_idx = i * shard_size
                    # Last model gets any remaining samples
                    end_idx = (i+1) * shard_size if i < len(models)-1 else len(self.dataset)
                    shard = self.dataset[start_idx:end_idx]
                    
                    # Keep track of original indices
                    index_mappings[model.name()] = list(range(start_idx, end_idx))
                    
                    tasks[model.name()] = asyncio.create_task(self._infer_single_model(model, shard))
                    logger.info(f"[Engine._infer_all] Model {model.name()} assigned {len(shard)} samples (indices {start_idx}-{end_idx-1})")
                
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

    async def run(self):
        logger.info("[Engine.run] Starting evaluation run.")
        predictions = await self._infer_all()
        logger.info(f"[Engine.run] Predictions complete. Calculating scores...")
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
            # Instantiate a separate metric for each model with correct concurrency
            metric_instances = {
                model_name: type(self.metric)(max_concurrency=per_model_conc)
                for model_name in predictions.keys()
            }
        else:
            metric_instances = {model_name: self.metric for model_name in predictions.keys()}

        async def score_model(model_name, outs):
            metric = metric_instances[model_name]
            if ids and lengths:
                return model_name, await asyncio.to_thread(metric, outs, model_targets, ids, lengths, instructions=instructions, dataset_name=self.dataset_name, model_name=model_name)
            else:
                return model_name, await asyncio.to_thread(metric, outs, model_targets, instructions=instructions, dataset_name=self.dataset_name, model_name=model_name)

        # Run all model scoring concurrently
        tasks = [score_model(model_name, outs) for model_name, outs in predictions.items()]
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
    token=os.getenv("HF_TOKEN")
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
    logger.info(f"[_load_dataset] Dataset loaded and processed. Size: {len(processed)}")
    return processed




def _load_models(cfg_list: list[dict]) -> list[Model]:
    logger.info(f"[_load_models] Instantiating models from config: {cfg_list}")
    models = []
    for cfg in cfg_list:
        model_name = cfg["info"].get("name")
        logger.info(f"[_load_models] Instantiating model for {model_name}")
        model_obj = Model(cfg["info"])
        models.append(model_obj)
    if not models:
        logger.info("[_load_models] ERROR: No valid models found in configuration.")
        raise ValueError("No valid models found in configuration.")
    for model in models:
        logger.info(f"Loaded {model.name()}")
    logger.info(f"[_load_models] Successfully instantiated {len(models)} model(s).")
    return models




def _load_metric(name: str, language: str = "en", judge_concurrency: int | None = None, judge_model: str | None = None):
    logger.info(f"[_load_metric] Loading metric: {name} (judge_concurrency={judge_concurrency}, judge_model={judge_model})")

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
def main(cfg_path='config.yaml'):
    logger.info(f"[main] Loading config from {cfg_path}")
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Load runspec files from the runspecs directory
    runspecs_dir = Path("runspecs")
    
    # Get list of all runspec files in the runspecs directory
    runspec_files = list(runspecs_dir.glob("*.json"))
    logger.info(f"[main] Found {len(runspec_files)} runspec files")
    
    # Load metric and model settings
    judge_concurrency = cfg.get("judge_concurrency", 1)
    judge_model = cfg.get("judge_model", None)
    user_prompt_add_ons = cfg.get("user_prompt_add_ons", [])
    system_prompts = cfg.get("system_prompts", [])
    length_filter = cfg.get("length_filter", None)
    num_samples = cfg.get("num_samples", None)
    
    # Load models
    logger.info(f"[main] Loading models...")
    models = _load_models(cfg.get("models", []))
    logger.info(f"[main] Loaded {len(models)} model(s).")
    
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
    
    logger.info(f"[main] Dataset-metric pairs from config: {dataset_metric_pairs}")
    
    # Store all scores in a flat dict with keys in format: 'dataset_name_metric_name'
    all_scores = {}
    
    # Process each dataset-metric pair
    for dname, metric_name in dataset_metric_pairs:
        logger.info(f"[main] Processing dataset '{dname}' with metric '{metric_name}' ...")
        
        # Step 1: Look for a matching runspec file
        found_runspec = False
        selected_datasets = {}
        
        # Try to match the dataset name with one of the runspec files
        for runspec_file in runspec_files:
            runspec_name = runspec_file.stem
            
            # First, check if dataset name exactly matches the runspec file name
            if dname == runspec_name:
                logger.info(f"[main] Found matching runspec file: {runspec_file}")
                found_runspec = True
                
                # Load the runspec file
                with open(runspec_file, 'r') as f:
                    runspec_db = json.load(f)
                
                # Use all datasets in this runspec
                selected_datasets = runspec_db
                logger.info(f"[main] Using all {len(runspec_db)} datasets from {runspec_file}")
                break
        
        # Step 2: If no matching runspec file by name, search within all runspec files for the specific dataset
        if not found_runspec:
            logger.info(f"[main] No matching runspec file for '{dname}'. Searching within individual runspec files...")
            
            # Search through all runspec files to find the dataset
            for runspec_file in runspec_files:
                with open(runspec_file, 'r') as f:
                    runspec_db = json.load(f)
                
                if dname in runspec_db:
                    logger.info(f"[main] Found dataset '{dname}' in {runspec_file}")
                    # Use only this specific dataset
                    selected_datasets = {dname: runspec_db[dname]}
                    found_runspec = True
                    break
            
            if not found_runspec:
                logger.warning(f"[main] Dataset '{dname}' not found in any runspec file")
                logger.info(f"[main] Dataset not found: {dname}")
                continue
        
        # Check for accented filter setting
        accented_filter = cfg.get("accented", None)
        if accented_filter is not None:
            logger.info(f"[main] Applying accented filter setting: {accented_filter}")
            
        # Check for language filter setting
        language_filter = cfg.get("language", None)
        if language_filter is not None:
            logger.info(f"[main] Applying language filter setting: {language_filter}")
        
        # Process each selected dataset
        for dataset_name, dataset_info in selected_datasets.items():
            # Check if we need to filter out accented datasets
            if accented_filter is False and dataset_info.get("accented", False) is True:
                logger.info(f"[main] Skipping dataset '{dataset_name}' because it is accented and accented filter is False")
                continue

            # Check if we need to filter by language
            if language_filter is not None:
                dataset_language = dataset_info.get("language", "").lower()
                if dataset_language and language_filter.lower() != dataset_language:
                    logger.info(f"[main] Skipping dataset '{dataset_name}' because its language '{dataset_language}' doesn't match filter '{language_filter}'")
                    continue
            
            logger.info(f"[main] Loading dataset '{dataset_name}' with metric '{metric_name}' ...")
            
            repo = dataset_info.get("hf_repo", None)
            split = None
            if not repo:
                repo = dataset_info.get("path", None)
            subset = dataset_info.get("subset", "")
            language = dataset_info.get("language", "en")
            preprocessor_name = dataset_info["preprocessor"]
            postprocessor_name = dataset_info["postprocessor"]
            modality=dataset_info.get("modality", "audio")

            if cfg.get("split", None) is not None:
                split = cfg.get("split")

            if dataset_info.get("split", None) is not None:
                split = dataset_info["split"]

            
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
                # Try to load the default postprocessor
                PostprocessorClass = get_class_from_module('postprocessors', 'GeneralPostprocessor')
            postprocessor = PostprocessorClass()
            
            logger.info("[main] Initializing Engine and running evaluation...")
            engine = Engine(models, dataset, metric, postprocessor, dataset_name, available_judge_calls=judge_concurrency)
            scores = asyncio.run(engine.run())
            all_scores[dataset_name] = scores
            logger.info(f"[main] Finished evaluation for dataset: {dataset_name}")
    logger.info(json.dumps(all_scores, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run audio evaluation benchmark')
    parser.add_argument('--config', '-c', default='config.yaml', 
                        help='Path to configuration file (default: config.yaml)')
    args = parser.parse_args()
    
    # Pass the config path to main
    main(cfg_path=args.config)
