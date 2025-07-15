import json
from pathlib import Path
import asyncio
from datasets import load_dataset
import yaml
from tqdm import tqdm
import importlib
import tempfile
import requests
import zipfile
import os
import glob
# Central logging setup
import logger_setup
logger_setup.configure()
import logging
logger = logging.getLogger(__name__)
from models.model import Model
from metrics.metrics import Metrics
from postprocessors.base import Postprocessor
from utils.constants import DATASET_METADATA_FILE


class Engine:
    #Evaluate one or many models over the same dataset concurrently
    def __init__(self, models: list[Model], dataset: list[dict], metric: Metrics, postprocessor: Postprocessor):
        logger.info(f"[Engine.__init__] Initializing Engine with {len(models)} model(s), dataset size: {len(dataset)}, metric: {metric.name}")
        self.models = models
        self.dataset = dataset
        self.metric = metric
        self.postprocessor = postprocessor

    # Infer single model over dataset asynchronously
    async def _infer_single_model(self, model: Model) -> list[str]:
        logger.info(f"[Engine._infer_single_model] Running model: {model.name()} on dataset of size {len(self.dataset)}")
        sem = asyncio.Semaphore(model.batch_size)  # Use per-model batch size
        async def _call(idx: int, sample: dict):
            async with sem:
                resp = await model._generate_text_with_retry(sample, {"chunk_size": model.chunk_size, "metric": self.metric.name})
                return idx, (resp.llm_response if resp else "")
        # Create tasks paired with their original indexx
        tasks = [_call(i, ex) for i, ex in enumerate(self.dataset)]
        results: list[str | None] = [None] * len(tasks)
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Inference ({model.name()})"):
            idx, text = await coro
            results[idx] = text
        logger.info(f"[Engine._infer_single_model] Model {model.name()} finished inference.")
        return results

    # Infer all models concurrently
    async def _infer_all(self):
        logger.info(f"[Engine._infer_all] Starting inference for all models: {[m.name() for m in self.models]}")
        tasks = {m.name(): asyncio.create_task(self._infer_single_model(m)) for m in self.models}
        results = {name: await t for name, t in tasks.items()}
        logger.info(f"[Engine._infer_all] All models finished inference.")
        return results

    def run(self):
        logger.info("[Engine.run] Starting evaluation run.")
        predictions = asyncio.run(self._infer_all())
        logger.info(f"[Engine.run] Predictions complete. Calculating scores...")
        scores = {}
        model_targets = self.postprocessor.extract_model_targets(dataset=self.dataset)
        for model_name, outs in predictions.items():
            logger.info(f"[Engine.run] Scoring model: {model_name}")
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





def get_class_from_module(module_prefix, module_name):
    try:
        module = importlib.import_module(f"{module_prefix}.{module_name}")
        return getattr(module, module_name)
    except Exception as e:
        return None

def _load_dataset(repo, subset=None, num_samples=None, preprocessor_name="AudiobenchPreprocessor", user_prompt_add_ons: list[str] = None, zip=False):
    """
    Load a dataset from HuggingFace or a ZIP file.

    Args:
        repo: HuggingFace dataset repo or local path to ZIP file
        subset: Optional subset of the dataset to load
        num_samples: Optional number of samples to load
        preprocessor_name: Name of preprocessor to use(defaults to AudiobenchPreprocessor)
        user_prompt_add_ons: Optional list of user prompt add-ons
        zip: Optional flag to load from a ZIP file

    Returns:
        Dataset loaded from HuggingFace or ZIP file
    """
    user_prompt_add_ons = user_prompt_add_ons or []
    if zip:
        try:
            logger.info(f"[_load_dataset] ZIP flag set. Downloading archive from {repo}")
            tmpdir = tempfile.mkdtemp(prefix="ab_zip_")
            local_zip = os.path.join(tmpdir, "dataset.zip")
            # Stream download to avoid large memory usage
            with requests.get(repo, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(local_zip, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            logger.info(f"[_load_dataset] Extracting archive {local_zip} …")
            with zipfile.ZipFile(local_zip) as zf:
                zf.extractall(tmpdir)
            json_files = glob.glob(os.path.join(tmpdir, "**", "*.json"), recursive=True)
            if not json_files:
                raise RuntimeError("No JSON files discovered inside ZIP; cannot build dataset automatically.")
            logger.info(f"[_load_dataset] Found {len(json_files)} JSON file(s); building HF dataset from them…")
            dset = load_dataset("json", data_files=json_files, split="train")
            # Apply any sampling/truncation logic
            if num_samples is not None:
                logger.info(f"[_load_dataset] Truncating dataset to first {num_samples} samples.")
                dset = dset[:num_samples]
            else:
                dset = dset[:len(dset)]
            PreprocessorClass = get_class_from_module('preprocessors', preprocessor_name)
            processed = PreprocessorClass().process(dset, {"user_prompt_add_ons": user_prompt_add_ons})
            logger.info(f"[_load_dataset] Dataset loaded & processed from ZIP. Size: {len(processed)}")
            return processed
        except Exception as e:
            logger.exception(f"[_load_dataset] Failed to load dataset from ZIP: {e}")
            raise

    logger.info(f"[_load_dataset] Loading HuggingFace dataset repo: {repo}")
    def _select_split(ds):
        """
        
        Args:
            ds: Dataset to select split from
        
        Returns:
            str: Split name or None if *ds* is already a Dataset
        """
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
    if num_samples is not None:
        logger.info(f"[_load_dataset] Truncating dataset to first {num_samples} samples.")
        dset = dset[:num_samples]
    else:
        logger.info(f"[_load_dataset] num_samples not provided; using full dataset length.")
        dset = dset[:len(dset)]
    logger.info(f"[_load_dataset] Preprocessing dataset using {preprocessor_name}...")
    PreprocessorClass = get_class_from_module('preprocessors', preprocessor_name)
    processed = PreprocessorClass().process(dset, {"user_prompt_add_ons": user_prompt_add_ons})
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

#TO-DO: need to implement command line override, add common configs, group by task type
#main that runs
def main(cfg_path='config.yaml'):
    logger.info("[main] Starting main function.")
    cfg_path = Path(cfg_path)
    logger.info(f"[main] Loading config file: {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text())
    logger.info(f"[main] Config loaded: {cfg}")

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
    cfg_path = Path(__file__).with_name(DATASET_METADATA_FILE)
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
