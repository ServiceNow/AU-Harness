import nltk

from nltk.translate.meteor_score import single_meteor_score

from metrics.metrics import Metrics
from utils import util


class MeteorScore(Metrics):
    def __call__(self, candidates, references, *, dataset_name: str | None = None, model_name: str | None = None):
        overall = self.compute_record_level_scores(candidates, references)
        if dataset_name and model_name:
            scores = overall.get(self.name, [])
            self._write_record_log(references, candidates, scores, dataset_name, model_name)
            self._append_final_score(overall, dataset_name, model_name)
        return overall

    def _append_final_score(self, overall, dataset_name, model_name):
        import json, re
        from pathlib import Path
        def _slug(s):
            return re.sub(r"[^A-Za-z0-9_]+", "_", s)
        log_dir = Path("run_logs")
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / f"{_slug(dataset_name)}_{_slug(self.name)}_{_slug(model_name)}.log"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"final_score": overall}, ensure_ascii=False) + "\n")

    def _write_record_log(self, refs, cands, scores, dataset_name, model_name):
        import json, re
        from pathlib import Path
        from itertools import zip_longest
        def _slug(s):
            return re.sub(r"[^A-Za-z0-9_]+", "_", s)
        log_dir = Path("run_logs")
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / f"{_slug(dataset_name)}_{_slug(self.name)}_{_slug(model_name)}.log"
        with open(log_path, "w", encoding="utf-8") as f:
            for ref, cand, sc in zip_longest(refs, cands, scores, fillvalue=None):
                entry = {"reference": ref, "candidate": cand}
                if sc is not None:
                    entry["score"] = sc
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        # Always write to shared run.log
        self._write_to_run_json(refs, cands, scores, dataset_name, model_name)
    """MeteorScore using nltk tokenizer."""

    def __init__(self):
        super().__init__()
        self.name = "meteor"
        self.scorer = single_meteor_score
        nltk.download("wordnet")
        nltk.download('punkt')
        nltk.download('punkt_tab')
        
    def _write_to_run_json(self, refs, cands, scores, dataset_name, model_name):
        """Write each sample's prediction to the shared run.log file."""
        import json
        from pathlib import Path
        from itertools import zip_longest
        
        run_path = Path(".") / "run.log"
        with open(run_path, "a", encoding="utf-8") as f:
            for ref, cand, sc in zip_longest(refs, cands, scores, fillvalue=None):
                entry = {
                    "dataset": dataset_name,
                    "metric": self.name,
                    "model": model_name,
                    "reference": ref,
                    "candidate": cand,
                }
                if sc is not None:
                    entry["score"] = sc
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def compute_record_level_scores(self, candidates: list, references: list) -> dict[str, list | None]:
        """Compute the scores that should be saved in the record level file.

        Args:
            candidates: Generated text from the model
            references: Reference text from the dataset

        Returns:
            Scores for each record. The keys should be the column names that will be saved in the record level file.
        """
        from tqdm import tqdm
        score_list = []
        for i in tqdm(range(len(candidates)), desc="METEOR"):
            # default preprocess is str.lower()
            # default stemmer is PorterStemmer()
            # default wordnet is nltk.corpus.wordnet
            score = self.scorer(references[i], candidates[i])
            score = util.smart_round(score)
            score_list.append(score)

        return {self.name: score_list}
