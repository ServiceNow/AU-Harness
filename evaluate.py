import argparse
import json
from pathlib import Path
from typing import List
import asyncio
import importlib
import ast
from datasets import load_dataset
from preprocessors.audio_preprocessor import AudioBenchPreprocessor
import yaml
import logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    fh = logging.FileHandler("audiobench.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)
logger.propagate = True
from models.model import Model
from metrics.metrics import Metrics

class Engine:
    """Evaluate one or many models over the same dataset concurrently."""

    def __init__(self, models: list[Model], dataset: list[dict], metric: Metrics, batch_size: int = 4):
        logger.info(f"[Engine.__init__] Initializing Engine with {len(models)} model(s), dataset size: {len(dataset)}, metric: {metric.name}, batch_size: {batch_size}")
        self.models = models
        self.dataset = dataset
        self.metric = metric
        self.batch_size = batch_size

    # ---------------- internal helpers ----------------

    async def _infer_single_model(self, model: Model) -> list[str]:
        logger.info(f"[Engine._infer_single_model] Running model: {model.name()} on dataset of size {len(self.dataset)}")
        sem = asyncio.Semaphore(self.batch_size)

        async def _call(sample):
            async with sem:
                # Pass a copy to ensure thread safety
                resp = await model._generate_text_with_retry(sample, {}, {})
                return resp.llm_response if resp else ""

        tasks = [
            _call(ex) for ex in self.dataset
        ]

        results = await asyncio.gather(*tasks)
        logger.info(f"[Engine._infer_single_model] Model {model.name()} finished inference.")
        return results

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
        for model_name, outs in predictions.items():
            logger.info(f"[Engine.run] Scoring model: {model_name}")
            scores[model_name] = self.metric(outs, self.dataset)
        logger.info(f"[Engine.run] Evaluation complete. Returning scores.")
        return {self.metric.name: scores}


def _parse_args(argv: List[str] | None = None):
    logger.info("[_parse_args] Parsing command line arguments.")
    p = argparse.ArgumentParser(description="AudioBench-Minimal evaluator")
    p.add_argument("--cfg", type=Path, help="Path to YAML/JSON benchmark config", default=None)
    p.add_argument('--num-samples', type=int, default=None, help='Evaluate only the first N samples from the dataset')
    args = p.parse_args(argv)
    logger.info(f"[_parse_args] Parsed args: cfg={args.cfg}, num_samples={args.num_samples}")
    return args


def _load_dataset(name: str):
    logger.info(f"[_load_dataset] Loading dataset: {name}")
    cfg_path = Path(__file__).with_name("audiobench_datasets.json")
    db = json.loads(cfg_path.read_text())
    if name not in db:
        raise ValueError(f"Dataset '{name}' not found in {cfg_path}.")

    repo = db[name]["hf_repo"]
    logger.info(f"[_load_dataset] Loading HuggingFace dataset repo: {repo}")
    dset = load_dataset(repo, split="test" if "test" in load_dataset(repo).keys() else None)
    if args.num_samples is not None:
        logger.info(f"[_load_dataset] Truncating dataset to first {args.num_samples} samples.")
        #take first x samples
        dset = dset[:args.num_samples]
    # Convert HF Dataset to list[dict] and preprocess
    logger.info(f"[_load_dataset] Preprocessing dataset...")
    processed = AudioBenchPreprocessor().process(dset, {})
    logger.info(f"[_load_dataset] Dataset loaded and processed. Size: {len(processed)}")
    #logger.info("[_load_dataset] First record after preprocessing:", processed[0])
    return processed

def _load_models(cfg_list: list[dict]) -> list[Model]:
    logger.info(f"[_load_models] Instantiating models from config: {cfg_list}")
    models = []
    for cfg in cfg_list:
        module_path = f"models.{cfg['name']}"
        logger.info(f"[_load_models] Importing module: {module_path}")
        module = importlib.import_module(module_path)
        for attr in dir(module):
            obj = getattr(module, attr)
            if isinstance(obj, type) and issubclass(obj, Model) and obj is not Model:
                logger.info(f"[_load_models] Instantiated model class: {attr}")
                models.append(obj(cfg["info"]))
                break
    if not models:
        logger.info("[_load_models] ERROR: No valid models found in configuration.")
        raise ValueError("No valid models found in configuration.")
    logger.info(f"[_load_models] Successfully instantiated {len(models)} model(s).")
    return models

def _load_metric(name: str):
    logger.info(f"[_load_metric] Loading metric: {name}")
    if name == "word_error_rate":
        from metrics.word_error_rate_metrics import WERMetrics
        metric = WERMetrics()
    elif name == "bleu":
        from metrics.bleu_metrics import BLEUMetrics
        metric = BLEUMetrics()
    elif name == "llm_judge":
        from metrics.llm_judge import LLMJudgeMetric
        metric = LLMJudgeMetric()
    elif name == "meteor":
        from metrics.meteor_score import MeteorScore
        metric = MeteorScore()

    else:
        raise ValueError(f"Unknown metric: {name}")
    logger.info(f"[_load_metric] Metric loaded: {metric.name}")
    return metric

def main(argv: List[str] | None = None):
    logger.info("[main] Starting main function.")
    global args
    args = _parse_args(argv)
    if args.cfg is None:
        raise ValueError("--cfg must be provided")

    logger.info(f"[main] Loading config file: {args.cfg}")
    cfg = yaml.safe_load(args.cfg.read_text())
    logger.info(f"[main] Config loaded: {cfg}")

    logger.info("[main] Loading models...")
    models = _load_models(cfg["models"])

    batch_size = cfg.get("batch_size", 4)

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


    all_scores = {}
    for dname, metric_name in dataset_metric_pairs:
        logger.info(f"[main] Loading dataset '{dname}' with metric '{metric_name}' ...")
        dataset = _load_dataset(dname)
        metric = _load_metric(metric_name)

        if args.num_samples is not None:
            logger.info(f"[main] Truncating dataset to {args.num_samples} samples.")
            dataset = dataset[:args.num_samples]

        logger.info("[main] Initializing Engine and running evaluation...")
        result = Engine(models, dataset, metric, batch_size=batch_size).run()
        all_scores[dname] = result

    logger.info("[main] Evaluation scores:")
    logger.info(json.dumps(all_scores, indent=2))


if __name__ == "__main__":
    main()
