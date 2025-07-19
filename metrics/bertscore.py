
from bert_score import score
from metrics.metrics import Metrics
from utils import util

from metrics.word_error_rate_metrics import normalize_text


class BertScore(Metrics):
    def __call__(self, candidates, references, dataset_name: str | None = None, model_name: str | None = None):
        return self.compute_record_level_scores(candidates, references)
    """BertScore using bert-score (pypi)."""

    def __init__(self):
        super().__init__()
        self.name = "bertscore"
        self.scorer = score

    def compute_record_level_scores(self, candidates: list, references: list) -> dict[str, list | None]:
        """Compute the scores that should be saved in the record level file.

        Args:
            candidates: Generated text from the model
            references: Reference text from the dataset

        Returns:
            Scores for each record. The keys should be the column names that will be saved in the record level file.
        """

        # Optimizing for batch processing (more efficient with GPU)
        from tqdm import tqdm
        score_list = []
        for i in tqdm(range(len(candidates)), desc="BERTSCORE"):
            #=== Consistent normalization with WER processing === 
            reference, candidate = references[i], candidates[i]
            norm_reference = normalize_text(reference)
            norm_candidate = normalize_text(candidate) 

            precision,recall,f1 = self.scorer([norm_reference], [norm_candidate], model_type='bert-base-multilingual-cased')
            f1_score = f1.numpy().tolist()
            score_list.extend(f1_score)
        return {self.name: score_list}
