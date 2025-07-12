import logger_setup
logger_setup.configure()
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

import re
import json
from pathlib import Path
import asyncio
from datasets import load_dataset
import yaml
from tqdm import tqdm
import importlib
# Central logging setup
from models.model import Model
from metrics.metrics import Metrics

class Engine:
    """Evaluate one or many models over the same dataset concurrently."""
    def __init__(self, models: list[Model], dataset: list[dict], metric: Metrics, postprocessor, dataset_name: str):
        logger.info(f"[Engine.__init__] Initializing Engine with {len(models)} model(s), dataset size: {len(dataset)}, metric: {metric.name}")
        self.models = models
        self.dataset = dataset
        self.metric = metric
        self.postprocessor = postprocessor
        # Keep track of dataset name so we can create per-dataset log files
        self.dataset_name = dataset_name
        
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
        async def _call(idx: int, sample):
            async with sem:
                resp = await model._generate_text_with_retry(sample, {"chunk_size": model.chunk_size, "metric": self.metric.name})
                return idx, (resp.llm_response if resp else "")
        # Create tasks paired with their original index
        tasks = [_call(i, ex) for i, ex in enumerate(samples)]
        results: list[str | None] = [None] * len(tasks)
        # Use tqdm for progress while filling results in the correct order
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Inference ({model.name()})"):
            idx, text = await coro
            results[idx] = text
        logger.info(f"[Engine._infer_single_model] Model {model.name()} finished inference.")
        return results

    #infer all models concurrently
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

    #main engine runner
    def run(self):
        logger.info("[Engine.run] Starting evaluation run.")
        predictions = asyncio.run(self._infer_all())
        logger.info(f"[Engine.run] Predictions complete. Calculating scores...")
        scores = {}
        
        # Pass the metric name to the postprocessor
        process_result = self.postprocessor.process(dataset=self.dataset, predictions=predictions, metric=self.metric.name)
        
        model_targets, predictions, ids, lengths = process_result

        for model_name, outs in predictions.items():
            # Let the metric handle per-record logging internally
            if ids and lengths:
                model_score = self.metric(outs, model_targets, ids, lengths, dataset_name=self.dataset_name, model_name=model_name)
            else:
                model_score = self.metric(outs, model_targets, dataset_name=self.dataset_name, model_name=model_name)
            scores[model_name] = model_score
        logger.info(f"[Engine.run] Evaluation complete. Returning scores.")
        logger.info(f"[Engine.run] Scores: {scores}")
        return {self.metric.name: scores}





#pre/post helper getter
def get_class_from_module(module_prefix, module_name):
    try:
        module = importlib.import_module(f"{module_prefix}.{module_name}")
        return getattr(module, module_name)
    except Exception as e:
        return None

#load an preprocess(dataset specific)
from preprocessors.CallHomePreprocessor import CallHomePreprocessor

def _load_dataset(repo=None, subset=None, num_samples=None, preprocessor_name="AudiobenchPreprocessor", user_prompt_add_ons: list[str] = [], length_filter=None, zip=False, dataset_config=None, metric=None):
    """Load and preprocess a dataset from a local or remote path."""
    logger.info(f"[_load_dataset] Loading dataset {repo} with preprocessor {preprocessor_name}")
    # We can load different formats of datasets
    if repo and repo.startswith("/") or repo.startswith("./"): # Local path
        repo = Path(repo).resolve()
        logger.info(f"[_load_dataset] Loading local dataset from {repo}")
    
    if preprocessor_name.startswith("CallHome"):
        # Special-case CallHome dataset
        # Load CallHomePreprocessor from local preprocessor code
        properties = {"metric": metric}
        if user_prompt_add_ons:
            properties["user_prompt_add_ons"] = user_prompt_add_ons
        if length_filter:
            logger.info(f"[_load_dataset] Applying length filter: {length_filter}")
            properties["length_filter"] = tuple(length_filter)  # Convert list to tuple
        logger.info(f"[_load_dataset] Loading CallHome from path: {repo}")
        dataset = CallHomePreprocessor().process(repo, num_samples=num_samples, properties=properties)
        return dataset

    logger.info(f"[_load_dataset] Loading HuggingFace dataset repo: {repo}")
    # If 'subset' or 'data_dir' is specified, pass as second arg or kwarg
    # ----- robust split handling -----
    def _select_split(ds):
        """Return desired split name or None if *ds* is already a Dataset."""
        if isinstance(ds, dict):  # HuggingFace DatasetDict
            for cand in ("test", "data", "train"):
                if cand in ds:
                    return cand
        return None  # single-split Dataset

    if subset:
        logger.info(f"[_load_dataset] Loading subset: {subset}")
        ds_builder = load_dataset(repo, subset)
    else:
        ds_builder = load_dataset(repo)

    chosen_split = _select_split(ds_builder)
    dset = ds_builder if chosen_split is None else ds_builder[chosen_split]
    logger.info(f"[_load_dataset] Using split: {chosen_split or 'single-dataset'}  |  Size before trunc: {len(dset)}")
        #logger.info(f"[_load_dataset] Dataset loaded: {dset}")
    if num_samples is not None:
        logger.info(f"[_load_dataset] Truncating dataset to first {num_samples} samples.")
        dset = dset[:num_samples]
    #Added to convert to dict
    else:
        dset = dset[:len(dset)]
    #logger.info(f"[_load_dataset] Dataset loaded after truncation: {dset}")
    logger.info(f"[_load_dataset] Preprocessing dataset using {preprocessor_name}...")
    PreprocessorClass = get_class_from_module('preprocessors', preprocessor_name)
    properties = {"metric": metric}
    if user_prompt_add_ons:
        properties["user_prompt_add_ons"] = user_prompt_add_ons
    if length_filter:
        logger.info(f"[_load_dataset] Applying length filter: {length_filter}")
        properties["length_filter"] = tuple(length_filter)
    processed = PreprocessorClass().process(dset, properties)
    logger.info(f"[_load_dataset] Dataset loaded and processed. Size: {len(processed)}")
    return processed

#load models from config
def _load_models(cfg_list: list[dict]) -> list[Model]:
    logger.info(f"[_load_models] Instantiating models from config: {cfg_list}")
    models = []
    for cfg in cfg_list:
        # chunk_size is now supported as an info attribute for each model (default 30s if not specified)
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

#load metric from name
def _load_metric(name: str, language: str = "en", judge_concurrency: int | None = None, judge_model: str | None = None):
    logger.info(f"[_load_metric] Loading metric: {name} (judge_concurrency={judge_concurrency}, judge_model={judge_model})")
    if name == "word_error_rate":
        from metrics.word_error_rate_metrics import WERMetrics
        metric = WERMetrics(language=language)
    elif name == "bleu":
        from metrics.bleu_metrics import BleuMetrics
        metric = BleuMetrics()
    elif name == "llm_judge_binary":
        from metrics.llm_judge import BinaryLLMJudgeMetric
        metric = BinaryLLMJudgeMetric(max_concurrency=judge_concurrency, model=judge_model)
    elif name == "llm_judge_detailed":
        from metrics.llm_judge import DetailedLLMJudgeMetric
        metric = DetailedLLMJudgeMetric(max_concurrency=judge_concurrency, model=judge_model)
    elif name == "llm_judge_callhome":
        from metrics.llm_judge import CallHomeLLMJudgeMetric
        metric = CallHomeLLMJudgeMetric(max_concurrency=judge_concurrency, model=judge_model)
    elif name == "meteor":
        from metrics.meteor_score import MeteorScore
        metric = MeteorScore()
    else:
        raise ValueError(f"Unknown metric: {name}")
    logger.info(f"[_load_metric] Metric loaded: {metric.name}")
    return metric

#hardcoded cfg path - need to implement command line override, add common configs, group by task type
#main that runs
def main(cfg_path='config.yaml'):
    logger.info("[main] Starting main function.")
    cfg_path = Path(cfg_path)
    logger.info(f"[main] Loading config file: {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text())
    logger.info(f"[main] Config loaded: {cfg}")

    # Reconfigure log file if provided
    log_file_path = cfg.get("log_file")
    if log_file_path:
        logger_setup.configure(log_file_path)
        logger.info(f"[main] Reconfigured logging to file: {log_file_path}")
    batch_size = cfg.get("batch_size", 4)
    num_samples = cfg.get("num_samples", None)
    judge_concurrency = cfg.get("judge_concurrency", None)
    judge_model = cfg.get("judge_model", None)
    api_version = cfg.get("api_version", None)
    user_prompt_add_ons: list[str] = cfg.get("user_prompt_add_ons", []) or []
    length_filter = cfg.get("length_filter", None)  # Get the length_filter from config
    
    # --- Build (dataset, metric) pairs ---
    raw_pairs_seq = cfg["dataset_metric"]
    if not isinstance(raw_pairs_seq, list):
        raise ValueError("'dataset_metric' must be a YAML list, each element like '(dataset_name, metric_name)'.")
    dataset_metric_pairs: list[tuple[str, str]] = []
    for raw in raw_pairs_seq:
        if not isinstance(raw, str):
            raise ValueError("Each element in 'dataset_metric' must be a plain string of the form '(dataset, metric)'.")
        raw = raw.strip()
        if not (raw.startswith("(") and raw.endswith(")")):
            raise ValueError(f"Invalid tuple format: {raw}. Expected '(dataset, metric)'.")
        inner = raw[1:-1]
        parts = [p.strip() for p in inner.split(",")]
        if len(parts) != 2 or not all(parts):
            raise ValueError(f"Invalid tuple contents: {raw}. Expect exactly two comma-separated values.")
        dataset_metric_pairs.append((parts[0], parts[1]))
    logger.info("[main] Loading models...")
    models = _load_models(cfg["models"])
    all_scores = {}
    cfg_path = Path(__file__).with_name("audiobench_datasets.json")
    db = json.loads(cfg_path.read_text())
    for dname, metric_name in dataset_metric_pairs:
        logger.info(f"[main] Loading dataset '{dname}' with metric '{metric_name}' ...")
        if dname not in db:
            raise ValueError(f"Dataset '{dname}' not found in {cfg_path}.")
        repo = db[dname].get("hf_repo", None)
        if not repo:
            repo = db[dname].get("path", None)
        subset = db[dname].get("subset", "")
        language = db[dname].get("language", "en")
        preprocessor_name = db[dname]["preprocessor"]
        postprocessor_name = db[dname]["postprocessor"]
        
        dataset = _load_dataset(repo, subset=subset, num_samples=num_samples, preprocessor_name=preprocessor_name, 
                             user_prompt_add_ons=user_prompt_add_ons, length_filter=length_filter, metric=metric_name)
        metric = _load_metric(metric_name, language=language, judge_concurrency=judge_concurrency, judge_model=judge_model)
        # Dynamically import postprocessor class
        PostprocessorClass = get_class_from_module('postprocessors', postprocessor_name)
        if PostprocessorClass is None:
            PostprocessorClass = AudiobenchPostprocessor
        postprocessor = PostprocessorClass()
        logger.info("[main] Initializing Engine and running evaluation...")
        result = Engine(models, dataset, metric, postprocessor, dname).run()
        all_scores[dname] = result
    logger.info("[main] Evaluation scores:")
    logger.info(json.dumps(all_scores, indent=2))

if __name__ == "__main__":
    main()
