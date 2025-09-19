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
        
        overall = self.get_score(candidates, references)
        
        if task_name and model_name:
            scores = self.record_level_scores.get(self.name, [])
            scores, normalized_candidates, normalized_references = self.compute_record_level_scores(candidates, references) 
            write_record_log(self, normalized_references, normalized_candidates, scores, task_name, model_name,
                           instructions=self.instructions, model_responses=self.model_responses)
            append_final_score(self, overall, task_name, model_name, self.model_responses)
        return overall

    def __init__(self):
        super().__init__()
        self.name = "gsm8k_exact_match"
        self.instructions = None
        self.model_responses = []

    def _preprocess_answer(self, text: str) -> str:
        """Extract and clean numerical answer from text.
        
        Looks for \\boxed{} patterns and extracts numerical values.
        """
        if not isinstance(text, str):
            text = str(text)
        
        boxed_matches = list(re.finditer(r"\\boxed\{([^}]+)\}", text))
        if boxed_matches:
            boxed_match = boxed_matches[-1]
            boxed_content = boxed_match.group(1)
            boxed_content = re.sub(r"[,$\\]", "", boxed_content)
            boxed_content = re.sub(r"\.$", "", boxed_content)
        
            match = re.search(r"(\-?[0-9\.]+)", boxed_content)
            return match.group(1) if match else boxed_content.strip()
        return text

    def _process_reference(self, text: str) -> str:
        """Extract numerical answer from reference text.
        
        Looks for #### markers and extracts following numerical values.
        """
        if not isinstance(text, str):
            text = str(text)
        
        text = re.sub(r"[,$\\]", "", text)
        
        last_marker_pos = text.rfind("#### ")
        if last_marker_pos != -1:
            text = text[last_marker_pos:]
        
        text = re.sub(r"\.$", "", text)
        
        match = re.search(r"#### (\-?[0-9\.]+)", text)
        
        return match.group(1) if match else text.strip()

    def get_score(self, candidates: list, references: list) -> dict[str, float]:
        """Compute overall accuracy percentage.
        
        Args:
            candidates: Generated text from the model
            references:  Reference text from the dataset
            
        Returns:
            Dictionary with accuracy percentage under metric name
        """
        norm_references = [self._process_reference(r) for r in references]
        norm_candidates = [self._preprocess_answer(c) for c in candidates]
        correct = sum(1 for c, r in zip(norm_candidates, norm_references) if c == r)
        total = len(candidates)
        accuracy = (correct / total) * 100.0 if total > 0 else 0.0
        
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
            # Preprocess both candidate and reference
            norm_reference = self._process_reference(r)
            norm_candidate = self._preprocess_answer(c)
            
            # Check exact match (1.0 for correct, 0.0 for incorrect)
            score = 1.0 if norm_candidate == norm_reference else 0.0
            scores.append(score)
            
            normalized_candidates.append(norm_candidate)
            normalized_references.append(norm_reference)
            
        return {self.name: scores}, normalized_candidates, normalized_references