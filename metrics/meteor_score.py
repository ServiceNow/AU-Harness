"""METEOR score metrics implementation.

This module provides METEOR score calculation capabilities for evaluating
text generation quality with semantic similarity measures.
"""
import nltk
from nltk.translate.meteor_score import single_meteor_score
from tqdm import tqdm

from metrics.metrics import Metrics
from metrics.word_error_rate_metrics import normalize_text
from utils import util
from utils.custom_logging import write_record_log, append_final_score


class MeteorScore(Metrics):
    """METEOR score evaluation metric.
    
    Computes METEOR scores for text generation evaluation using semantic
    similarity measures including exact, stem, synonym, and paraphrase matches.
    """
    def __call__(self, candidates, references, instructions=None, *, dataset_name: str | None = None, model_name: str | None = None, model_responses=None):
        # Store instructions and model_responses for potential later use
        self.instructions = instructions
        self.model_responses = model_responses if model_responses else []

        # Get individual scores
        self.record_level_scores = self.compute_record_level_scores(candidates, references)

        # Calculate the mean score directly to avoid async issues
        scores = self.record_level_scores.get(self.name, [])
        valid_scores = [score for score in scores if score is not None]
        mean_score = util.smart_round(sum(valid_scores) / len(valid_scores)) if valid_scores else 0.0
        overall_score = {self.name: mean_score}

        if dataset_name and model_name:
            write_record_log(self, references, candidates, scores, dataset_name, model_name, 
                           instructions=self.instructions, model_responses=self.model_responses)
            append_final_score(self, overall_score, dataset_name, model_name, self.model_responses)
        
        # Return both individual scores and the aggregate score
        return {**self.record_level_scores, **overall_score}

    def __init__(self):
        super().__init__()
        self.name = "meteor"
        self.scorer = single_meteor_score
        self.instructions = None
        self.model_responses = []
        self.record_level_scores = {}
        nltk.download("wordnet", quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)

    def compute_record_level_scores(self, candidates: list, references: list) -> dict[str, list | None]:
        # Here we can use self.instructions if needed
        """Compute the scores that should be saved in the record level file.

        Args:
            candidates: Generated text from the model
            references: Reference text from the dataset

        Returns:
            Scores for each record. The keys should be the column names that will be saved in the record level file.
        """
        score_list = []
        for i in tqdm(range(len(candidates)), desc="METEOR"):
            # default preprocess is str.lower()
            # default stemmer is PorterStemmer()
            # default wordnet is nltk.corpus.wordnet

            # === Consistent normalization with WER processing ===
            reference, candidate = references[i], candidates[i]
            norm_reference = normalize_text(reference)
            norm_candidate = normalize_text(candidate)

            # Compute METEOR Score
            score = self.scorer(norm_reference.split(), norm_candidate.split())
            score = util.smart_round(score)
            score_list.append(score)

        return {self.name: score_list}
