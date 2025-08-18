"""BLEU score metrics implementation.

This module provides BLEU score calculation capabilities for evaluating
text generation quality using n-gram precision measures.
"""
from sacrebleu.metrics import BLEU
from tqdm import tqdm

from metrics.metrics import Metrics
from metrics.word_error_rate_metrics import normalize_text
from utils.custom_logging import write_record_log, append_final_score


class BleuMetrics(Metrics):
    """BLEU score evaluation metric.
    
    Computes BLEU scores for text generation evaluation using n-gram precision
    measures to assess translation quality and text similarity.
    """
    def __call__(self, candidates, references, instructions=None, *, dataset_name: str | None = None, model_name: str | None = None, model_responses=None):
        # Store instructions and model_responses for potential later use
        self.instructions = instructions
        self.model_responses = model_responses if model_responses else []
        # Use corpusBLEU for overall score
        overall = self.get_score(candidates, references)
        if dataset_name and model_name:
            scores = self.record_level_scores.get(self.name, [])
            # Use sentenceBLEU for record-level scores
            scores, normalized_candidates, normalized_references = self.compute_record_level_scores(candidates, references) 
            # write_record_log will also write to run.log internally
            write_record_log(self, normalized_references, normalized_candidates, scores, dataset_name, model_name,
                           instructions=self.instructions, model_responses=self.model_responses)
            # Directly call append_final_score
            append_final_score(self, overall, dataset_name, model_name, self.model_responses)
        return overall

    def __init__(self, max_ngram_order=4):
        super().__init__()
        self.scorer = None
        self.name = "bleu"
        self.max_ngram_order = max_ngram_order
        self.instructions = None
        self.model_responses = []

    def get_score(self, candidates, references):
        """This gives overall score of complete dataset.

        Args:
            candidates: generated text list
            references: reference text list

        Returns:
            {"BLEU":100}
        """
        self.scorer = BLEU(max_ngram_order=self.max_ngram_order)
        # === Consistent normalization with WER processing ===
        norm_references = [normalize_text(r) for r in references]
        norm_candidates = [normalize_text(c) for c in candidates]
        bs = self.scorer.corpus_score(norm_candidates, [norm_references])
        return {self.name: bs.score}

    # ---------------------------------------------------
    # Internal helper
    # ---------------------------------------------------
    def compute_record_level_scores(self, candidates: list, references: list) -> dict[str, list | None]:
        # Here we can use self.instructions if needed
        """Compute the scores that should be saved in the record level file.

        Args:
            candidates: Generated text from the model
            references: Reference text from the dataset

        Returns:
            Scores for each record. The keys should be the column names that will be saved in the record level file.
        """
        scores = []
        normalized_candidates, normalized_references = [], []
        self.scorer = BLEU(effective_order=True, max_ngram_order=self.max_ngram_order)
        for c, r in tqdm(zip(candidates, references), desc="BLEU", total=len(candidates)):
            # === Consistent normalization with WER processing ===
            norm_reference = normalize_text(r)
            norm_candidate = normalize_text(c)
            score = self.scorer.sentence_score(norm_candidate, [norm_reference])
            scores.append(score)
            normalized_candidates.append(norm_candidate)
            normalized_references.append(norm_reference)
        return {self.name: scores}, normalized_candidates, normalized_references
