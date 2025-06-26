import json
from pathlib import Path
import asyncio
import importlib
from datasets import load_dataset
from postprocessors.audio_postprocessor import AudiobenchPostprocessor
from preprocessors.audio_preprocessor import AudioBenchPreprocessor
import yaml
# Central logging setup
import logger_setup
import logging
logger = logging.getLogger(__name__)
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
        logger.info(f"[Engine._infer_all] Results: {results}")
        return results

    def run(self):
        logger.info("[Engine.run] Starting evaluation run.")
        logger.info(f"Running on batch size {self.batch_size}")
        predictions = asyncio.run(self._infer_all())
        logger.info(f"prediction type: {type(predictions)}")
        logger.info(f"predictions: {predictions}")
        logger.info(f"[Engine.run] prediction length: {len(predictions)}")

        logger.info(f"[Engine.run] Predictions complete. Calculating scores...")
        scores = {}
        model_targets = AudiobenchPostprocessor().extract_model_targets(dataset = self.dataset)
        logger.info(f"[Engine.run] Model targets: {model_targets}")
        for model_name, outs in predictions.items():
            logger.info(f"[Engine.run] Scoring model: {model_name}")
            logger.info(f"[Engine.run] Outs type: {type(outs)}")
            logger.info(f"[Engine.run] Outs: {outs}")
            scores[model_name] = self.metric(outs, model_targets)
        logger.info(f"[Engine.run] Evaluation complete. Returning scoresz.")
        return {self.metric.name: scores}




def _load_dataset(name: str, num_samples=None):
    logger.info(f"[_load_dataset] Loading dataset: {name}")
    cfg_path = Path(__file__).with_name("audiobench_datasets.json")
    db = json.loads(cfg_path.read_text())
    if name not in db:
        raise ValueError(f"Dataset '{name}' not found in {cfg_path}.")

    repo = db[name]["hf_repo"]
    logger.info(f"[_load_dataset] Loading HuggingFace dataset repo: {repo}")
    dset = load_dataset(repo, split="test" if "test" in load_dataset(repo).keys() else None)
    if num_samples is not None:
        logger.info(f"[_load_dataset] Truncating dataset to first {num_samples} samples.")
        dset = dset[:num_samples]
    logger.info(f"[_load_dataset] Preprocessing dataset...")
    #logger.info(f"[_load_dataset] Dataset keys: {dset.keys()}")
    #logger.info(f"[_load_dataset] Dataset length: {len(dset)}")
    #logger.info(f"[_load_dataset] Dataset keys: {dset.keys()}")
    #for key in dset.keys():
    #    logger.info(f"Key: {key}, Value: {dset[key]}")
    #logger.info(f"dataset type: {type(dset)}")
    processed = AudioBenchPreprocessor().process(dset, {})
    
    logger.info(f"[_load_dataset] Dataset loaded and processed. Size: {len(processed)}")
    return processed, db[name].get("language", "en")


def _load_models(cfg_list: list[dict]) -> list[Model]:
    logger.info(f"[_load_models] Instantiating models from config: {cfg_list}")
    models = []
    for cfg in cfg_list:
        logger.info(f"[_load_models] Instantiating model: {cfg['name']}")
        if 'info' not in cfg:
            raise ValueError("Each model config must have an 'info' key.")
        model_info = dict(cfg['info'])  # Make a copy to avoid mutating the original
        model_info['name'] = cfg['name']  # Inject the top-level name into info
        if 'target' not in model_info:
            raise ValueError("Each model config info must have a 'target' key.")
        model_key = model_info['target']
        module_path = f"models.{model_key}"
        logger.info(f"[_load_models] Importing module: {module_path}")
        module = importlib.import_module(module_path)
        for attr in dir(module):
            obj = getattr(module, attr)
            if isinstance(obj, type) and issubclass(obj, Model) and obj is not Model:
                logger.info(f"[_load_models] Instantiated model class: {attr}")
                models.append(obj(model_info))
                break
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
        from metrics.bleu_metrics import BLEUMetrics
        metric = BLEUMetrics()
    elif name == "llm_judge_binary":
        from metrics.llm_judge import BinaryLLMJudgeMetric
        metric = BinaryLLMJudgeMetric(max_concurrency=judge_concurrency, model=judge_model)
    elif name == "llm_judge_detailed":
        from metrics.llm_judge import DetailedLLMJudgeMetric
        metric = DetailedLLMJudgeMetric(max_concurrency=judge_concurrency, model=judge_model)
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


def main(cfg_path='test_config.yaml'):
    logger.info("[main] Starting main function.")
    cfg_path = Path(cfg_path)
    logger.info(f"[main] Loading config file: {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text())
    logger.info(f"[main] Config loaded: {cfg}")

    batch_size = cfg.get("batch_size", 4)
    num_samples = cfg.get("num_samples", None)
    judge_concurrency = cfg.get("judge_concurrency", None)
    judge_model = cfg.get("judge_model", None)

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
    for dname, metric_name in dataset_metric_pairs:
        logger.info(f"[main] Loading dataset '{dname}' with metric '{metric_name}' ...")
        dataset, language = _load_dataset(dname, num_samples=num_samples)
        metric = _load_metric(metric_name, language=language, judge_concurrency=judge_concurrency, judge_model=judge_model)

        logger.info("[main] Initializing Engine and running evaluation...")
        result = Engine(models, dataset, metric, batch_size=batch_size).run()
        all_scores[dname] = result

    logger.info("[main] Evaluation scores:")
    logger.info(json.dumps(all_scores, indent=2))


if __name__ == "__main__":
    main()
