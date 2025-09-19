"""GSM8k correctness metrics implementation.

Evaluates GSM8k correctness  using exact match comparison
for mathematical reasoning accuracy.
"""
import re
from tqdm import tqdm
from typing import Tuple

from metrics.metrics import Metrics
from utils import util
from utils.custom_logging import write_record_log, append_final_score


class Gsm8kExactMatch(Metrics):
    """GSM8k correctness evaluation metric.
    
    Computes exact match accuracy for GSM8k mathematical reasoning
    after preprocessing to remove commas, dollar signs, and periods.
    """
    def __call__(self, candidates, references, instructions=None, *, task_name: str | None = None, model_name: str | None = None, model_responses=None):
        """Evaluate GSM8k correctness and optionally log results.
        
        Args:
            candidates: Generated text list
            references: Reference text list  
            instructions: Optional instructions text
            task_name: Task identifier for logging
            model_name: Model identifier for logging
            model_responses: Optional model responses for logging
            
        Returns:
            Dictionary with overall accuracy score
        """
        self.instructions = instructions
        self.model_responses = model_responses if model_responses else []

        self.boxed_pattern = re.compile(r"\\boxed\{([^}]+)\}")
        self.cleanup_pattern = re.compile(r"[,$\\]")
        self.trailing_period_pattern = re.compile(r"\.$")
        self.number_pattern = re.compile(r"(\-?[0-9\.]+)")
        self.marker_pattern = re.compile(r"#### (\-?[0-9\.]+)")
        
        scores, normalized_candidates, normalized_references = self.compute_record_level_scores(candidates, references)        
        overall = self.get_score(candidates, references)
        
        if task_name and model_name:
            score_list = scores.get(self.name, [])
            write_record_log(self, normalized_references, normalized_candidates, score_list, task_name, model_name,
                           instructions=self.instructions, model_responses=self.model_responses)
            append_final_score(self, overall, task_name, model_name, self.model_responses)
        return overall

    def __init__(self):
        super().__init__()
        self.name = "gsm8k_exact_match"
        self.instructions = None
        self.model_responses = []

    def _clean_text(self, text: str) -> str:
        """Remove currency symbols, backslashes, and trailing periods."""
        text = self.cleanup_pattern.sub("", text)
        return self.trailing_period_pattern.sub("", text)

    def _extract_number(self, text: str) -> str:
        """Extract numerical value from cleaned text."""
        match = self.number_pattern.search(text)
        return match.group(1) if match else text.strip()

    def _preprocess_answer(self, text: str) -> str:
        """Extract and clean numerical answer from text."""
        if not isinstance(text, str):
            text = str(text)
        
        boxed_matches = list(self.boxed_pattern.finditer(text))
        if boxed_matches:
            boxed_content = boxed_matches[-1].group(1)
            cleaned_content = self._clean_text(boxed_content)
            return self._extract_number(cleaned_content)
        
        return text

    def _process_reference(self, text: str) -> str:
        """Extract numerical answer from reference text."""
        if not isinstance(text, str):
            text = str(text)
        
        text = self._clean_text(text)
        
        last_marker_pos = text.rfind("#### ")
        if last_marker_pos != -1:
            text = text[last_marker_pos:]
        
        match = self.marker_pattern.search(text)
        return match.group(1) if match else text.strip()

    def get_score(self, candidates: list, references: list) -> dict[str, float]:
        """Compute overall accuracy percentage.
        
        Args:
            candidates: Generated text from the model
            references:  Reference text from the dataset
            
        Returns:
            Dictionary with accuracy percentage under metric name
        """

        if not self.record_level_scores:
            self.record_level_scores, _, _ = self.compute_record_level_scores(candidates, references)

        scores = self.record_level_scores.get(self.name, [])
        accuracy = (sum(scores) / len(scores) * 100.0 if scores else 0.0)
                
        return {self.name: util.smart_round(accuracy, 2)}

    def compute_record_level_scores(self, candidates: list, references: list) -> Tuple[dict[str, list | None], list, list]:
        """Compute per-record scores.
        
        Args:
            candidates: Generated text from the model
            references: Reference text from the dataset
            
        Returns:
            Tuple of (scores dict, normalized candidates, normalized references)
        """
        scores = []
        normalized_candidates, normalized_references = [], []
        
        for c, r in tqdm(zip(candidates, references), desc="GSM8k Correctness", total=len(candidates)):
            norm_reference = self._process_reference(r)
            norm_candidate = self._preprocess_answer(c)
            scores.append(int(norm_candidate == norm_reference))
            normalized_candidates.append(norm_candidate)
            normalized_references.append(norm_reference)
        
        self.record_level_scores = {self.name: scores}
        return self.record_level_scores, normalized_candidates, normalized_references