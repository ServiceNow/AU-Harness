from sacrebleu.metrics import BLEU

from metrics.metrics import Metrics

class BleuMetrics(Metrics):
    def __call__(self, candidates, references, dataset_name: str | None = None, model_name: str | None = None):
        return self.get_score(candidates, references)
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
