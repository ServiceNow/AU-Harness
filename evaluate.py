import logger_setup
logger_setup.configure()
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

import os
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


    # ---------------- internal helpers ----------------x
    #infer by batch size, calling generate text with retry for each sample
    async def _infer_single_model(self, model: Model) -> list[str]:
        logger.info(f"[Engine._infer_single_model] Running model: {model.name()} on dataset of size {len(self.dataset)}")
        sem = asyncio.Semaphore(model.batch_size)  # Use per-model batch size
        async def _call(idx: int, sample):
            async with sem:
                resp = await model._generate_text_with_retry(sample, {"chunk_size": model.chunk_size, "metric": self.metric.name})
                return idx, (resp.llm_response if resp else "")
        # Create tasks paired with their original indexx
        tasks = [_call(i, ex) for i, ex in enumerate(self.dataset)]
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
        tasks = {m.name(): asyncio.create_task(self._infer_single_model(m)) for m in self.models}
        results = {name: await t for name, t in tasks.items()}
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
    # Determine the preferred split to load directly (more efficient)
    preferred_splits = ["test", "data", "train"]
    
    # Try to load a specific split directly
    dset = None
    # Try the preferred splits in order
    token=os.getenv("HF_TOKEN")
    for split_name in preferred_splits:
        try:
            if subset:
                logger.info(f"[_load_dataset] Attempting to load subset: {subset}, split: {split_name}")
                if token:
                    dset = load_dataset(repo, subset, split=split_name, trust_remote_code=True, token=token)
                else:
                    dset = load_dataset(repo, subset, split=split_name, trust_remote_code=True)
            else:
                logger.info(f"[_load_dataset] Attempting to load split: {split_name}")
                if token:
                    dset = load_dataset(repo, split=split_name, trust_remote_code=True, token=token)
                else:
                    dset = load_dataset(repo, split=split_name, trust_remote_code=True)
            logger.info(f"[_load_dataset] Successfully loaded split: {split_name}")
            break
        except Exception as e:
            logger.info(f"[_load_dataset] Split {split_name} not available: {str(e)}")
    
    # Raise an error if no valid split was found
    if dset is None:
        logger.info(f"[_load_dataset] Attempting to load no split")
        try:
            if subset:
                dset = load_dataset(repo, subset, trust_remote_code=True)
            else:
                dset = load_dataset(repo, trust_remote_code=True)
        except Exception as e:
            error_msg = f"[_load_dataset] No valid dataset found in {repo}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    logger.info(f"[_load_dataset] Dataset loaded  |  Size before trunc: {len(dset)}")


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
    # Process each dataset-metric pair
    expanded_pairs = []
    # Check for accented filter setting
    accented_filter = cfg.get("accented", None)
    logger.info(f"[main] Accented filter setting: {accented_filter}")
    
    for dname, metric_name in dataset_metric_pairs:
        logger.info(f"[main] Processing dataset '{dname}' with metric '{metric_name}' ...")
        
        # Check if the dataset name directly exists in the database
        if dname in db:
            logger.info(f"[main] Found direct match for dataset '{dname}'")
            # Check if we need to filter out accented datasets
            if accented_filter is False and db[dname].get("accented", False) is True:
                logger.info(f"[main] Skipping dataset '{dname}' because it is accented and accented filter is False")
                continue
            expanded_pairs.append((dname, metric_name))
        else:
            # Check if dname might be a task category instead
            logger.info(f"[main] Dataset '{dname}' not found directly. Checking if it's a task category...")
            found_matches = False
            
            # Search for datasets that have the task category
            for dataset_name, dataset_info in db.items():
                if "tasks" in dataset_info and dname in dataset_info["tasks"]:
                    # Check if we need to filter out accented datasets
                    if accented_filter is False and dataset_info.get("accented", False) is True:
                        logger.info(f"[main] Skipping dataset '{dataset_name}' matching task '{dname}' because it is accented and accented filter is False")
                        continue
                    
                    logger.info(f"[main] Found dataset '{dataset_name}' matching task category '{dname}'")
                    expanded_pairs.append((dataset_name, metric_name))
                    found_matches = True
            
            if not found_matches:
                # If we get here, neither dataset nor task was found, or all matching datasets were filtered out
                raise ValueError(f"Neither dataset nor task category '{dname}' found in {cfg_path}, or all matching datasets were filtered out.")
    
    logger.info(f"[main] Expanded dataset-metric pairs: {expanded_pairs}")
    
    # Process each actual dataset
    for dname, metric_name in expanded_pairs:
        logger.info(f"[main] Loading dataset '{dname}' with metric '{metric_name}' ...")
        
        # These datasets must exist in the database since we've validated them above
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
