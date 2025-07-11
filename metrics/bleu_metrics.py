import json, re
from pathlib import Path
from sacrebleu.metrics import BLEU

from metrics.metrics import Metrics

class BleuMetrics(Metrics):
    def __call__(self, candidates, references, *, dataset_name: str | None = None, model_name: str | None = None):
        overall = self.get_score(candidates, references)
        if dataset_name and model_name:
            scores = self.record_level_scores.get(self.name, [])
            self._write_record_log(references, candidates, scores, dataset_name, model_name)
            self._append_final_score(overall, dataset_name, model_name)
        return overall

    def _append_final_score(self, overall, dataset_name, model_name):
        import json, re
        from pathlib import Path
        def _slug(s):
            return re.sub(r"[^A-Za-z0-9_]+", "_", s)
        log_path = Path(".") / f"{_slug(dataset_name)}_{_slug(self.name)}_{_slug(model_name)}.log"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"final_score": overall}, ensure_ascii=False) + "\n")
    """TODO: Need SME to add."""

    def __init__(self, max_ngram_order=4):
        super().__init__()
        self.scorer = None
        self.name = "BLEU"
        self.max_ngram_order = max_ngram_order

    def get_score(self, candidates, references):
        """This gives overall score of complete dataset.

        Args:
            candidates: generated text list
            references: reference text list

        Returns:
            {"BLEU":100}
        """
        self.scorer = BLEU(max_ngram_order=self.max_ngram_order)
        bs = self.scorer.corpus_score(candidates, [references])
        return {self.name: bs.score}

    # ---------------------------------------------------
    # Internal helper
    # ---------------------------------------------------
    def _write_record_log(self, refs, cands, scores, dataset_name, model_name):
        if not refs or not scores:
            return
        def _slug(s):
            return re.sub(r"[^A-Za-z0-9_]+", "_", s)
        log_path = Path(".") / f"{_slug(dataset_name)}_{_slug(self.name)}_{_slug(model_name)}.log"
        with open(log_path, "w", encoding="utf-8") as f:
            for ref, cand, sc in zip(refs, cands, scores):
                f.write(json.dumps({"reference": ref, "candidate": cand, "score": sc}, ensure_ascii=False) + "\n")

    def compute_record_level_scores(self, candidates: list, references: list) -> dict[str, list | None]:
        """Compute the scores that should be saved in the record level file.

        Args:
            candidates: Generated text from the model
            references: Reference text from the dataset

        Returns:
            Scores for each record. The keys should be the column names that will be saved in the record level file.
        """
        from tqdm import tqdm
        self.scorer = BLEU(effective_order=True, max_ngram_order=self.max_ngram_order)
        scores = [self.scorer.sentence_score(c, [r]).score for c, r in tqdm(zip(candidates, references), desc="BLEU", total=len(candidates))]
        return {self.name: scores}
