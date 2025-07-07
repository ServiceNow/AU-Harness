import json
from pathlib import Path
import asyncio
from datasets import load_dataset
import yaml
from tqdm import tqdm
import importlib
# Central logging setup
import logger_setup
logger_setup.configure()
import logging
logger = logging.getLogger(__name__)
from models.model import Model
from metrics.metrics import Metrics

class Engine:
    """Evaluate one or many models over the same dataset concurrently."""
    def __init__(self, models: list[Model], dataset: list[dict], metric: Metrics, postprocessor):
        logger.info(f"[Engine.__init__] Initializing Engine with {len(models)} model(s), dataset size: {len(dataset)}, metric: {metric.name}")
        self.models = models
        self.dataset = dataset
        self.metric = metric
        self.postprocessor = postprocessor


    # ---------------- internal helpers ----------------x
    #infer by batch size, calling generate text with retry for each sample
    async def _infer_single_model(self, model: Model) -> list[str]:
        logger.info(f"[Engine._infer_single_model] Running model: {model.name()} on dataset of size {len(self.dataset)}")
        sem = asyncio.Semaphore(model.batch_size)  # Use per-model batch size
        async def _call(idx: int, sample):
            async with sem:
                #logger.info(f"{sample.keys()}")
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
        #logger.info(f"[Engine._infer_all] Results: {results}")
        return results

    #main engine runner
    def run(self):
        logger.info("[Engine.run] Starting evaluation run.")
        predictions = asyncio.run(self._infer_all())
        logger.info(f"[Engine.run] Predictions complete. Calculating scores...")
        scores = {}
        model_targets = self.postprocessor.extract_model_targets(dataset=self.dataset)
        for model_name, outs in predictions.items():
            logger.info(f"[Engine.run] Scoring model: {model_name}")
            # Log first 5 prediction/target pairs for quick sanity-check
            n_log = min(10, len(outs), len(model_targets))
            logger.info(f"[Engine.run] Logging first {n_log} prediction-target pairs for sanity-check\n\n")
            for i in range(n_log):
                logger.info(
                    f"[Engine.run] Example {i}: Prediction = {outs[i]!r} | Target = {model_targets[i]!r}"
                )
            logger.info("\n")
            scores[model_name] = self.metric(outs, model_targets)
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

def _load_dataset(repo, subset=None, num_samples=None, preprocessor_name="AudiobenchPreprocessor", user_prompt_add_ons: list[str] = [], zip=False, dataset_config=None):
    # CallHome special download
    if (isinstance(repo, str) and repo.lower() == "callhome") or (dataset_config and dataset_config.get("name", "").lower() == "callhome"):
        logger.info("[_load_dataset] Detected CallHome dataset. Downloading media files...")
        CallHomePreprocessor().download_media(dataset=None, properties=dataset_config or {})

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
        logger.info(f"[_load_dataset] num_samples not provided; using full dataset length.")
        dset = dset[:len(dset)]
    #logger.info(f"[_load_dataset] Dataset loaded after truncation: {dset}")
    logger.info(f"[_load_dataset] Preprocessing dataset using {preprocessor_name}...")
    PreprocessorClass = get_class_from_module('preprocessors', preprocessor_name)
    processed = PreprocessorClass().process(dset, {"user_prompt_add_ons": user_prompt_add_ons})
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
    user_prompt_add_ons: list[str] = cfg.get("user_prompt_add_ons", []) or []
    
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
        repo = db[dname]["hf_repo"]
        subset = db[dname].get("subset", "")
        language = db[dname].get("language", "en")
        preprocessor_name = db[dname]["preprocessor"]
        postprocessor_name = db[dname]["postprocessor"]
        dataset = _load_dataset(repo, subset=subset, num_samples=num_samples, preprocessor_name=preprocessor_name, user_prompt_add_ons=user_prompt_add_ons)
        metric = _load_metric(metric_name, language=language, judge_concurrency=judge_concurrency, judge_model=judge_model)
        # Dynamically import postprocessor class
        PostprocessorClass = get_class_from_module('postprocessors', postprocessor_name)
        if PostprocessorClass is None:
            PostprocessorClass = AudiobenchPostprocessor
        postprocessor = PostprocessorClass()
        logger.info("[main] Initializing Engine and running evaluation...")
        result = Engine(models, dataset, metric, postprocessor).run()
        all_scores[dname] = result
    logger.info("[main] Evaluation scores:")
    logger.info(json.dumps(all_scores, indent=2))

if __name__ == "__main__":
    main()
