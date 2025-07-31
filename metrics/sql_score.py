from metrics.metrics import Metrics
from typing import List, Tuple, Dict, Optional, Union
import os
import pandas as pd
import nltk
from utils.custom_logging import write_record_log, append_final_score
from metrics.text2sql.evaluation import evaluate

# Constants for file paths and data selection
SPIDER_DATA_DIR = "data/spider/"
SPIDER_DB_DIR = "data/spider/database"


class SqlScore(Metrics):
    def __init__(self):
        super().__init__()
        self.name = "text2sql_score"
        self.metric_ex = "Execution Accuracy"
        self.metric_em = "Exact Set Match"
        self.processed_text = "Post Processed Text"

        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")

        self.record_level_score = {}

    def __call__(
        self,
        candidates: List[str],
        references: List[Tuple[List[str], List[Dict[str, Optional[Union[str, int]]]]]],
        *,
        instructions: Optional[List[str]] = None,
        dataset_name: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> dict[str, dict[str, float] | float]:
        """
        Evaluate SQL execution accuracy and exact set match using Spider evaluation.

        Args:
            candidates (List[str]): Generated SQL strings.
            references (List[str]): Reference SQL strings.

        Returns:
            dict: Flattened dictionary with accuracy scores.
        """
        scores = evaluate(
            glist=references,
            plist=candidates,
            db_dir=SPIDER_DB_DIR,
            etype="all",
            table=os.path.join(SPIDER_DATA_DIR, "tables.jsonl"),
        )

        self.record_level_score = {
            self.processed_text: candidates,
            self.metric_ex: scores.get("per_record_ex", []),
            self.metric_em: scores.get("per_record_em", []),
        }

        # Write detailed record-level logs (if dataset_name and model_name provided)
        if dataset_name and model_name:
            append_final_score(self, scores, dataset_name, model_name)
            write_record_log(
                self, 
                refs=references, 
                cands=candidates, 
                scores=scores, 
                dataset_name=dataset_name, 
                model_name=model_name, 
                explanations=None, 
                instructions=instructions
            )

        return self._clean_scores(scores)

    def _clean_scores(self, scores: dict) -> dict:
        """
        Flatten the output scores.

        Args:
            scores: Dictionary containing evaluation scores.

        Returns:
            Flattened dictionary containing formatted scores.
        """
        flattened_scores = {}
        for level, ex, em in zip(
            scores["levels"], scores["exec_accuracy_score"], scores["exact_match_score"]
        ):
            if level == "all":
                level = "overall"
            flattened_scores[f"{level}_exec_accuracy"] = round(ex, 4)
            flattened_scores[f"{level}_exact_set_match"] = round(em, 4)
        return flattened_scores

    def get_all_score_df(
        self, ids: List[int], candidates: List[str], references: List[str]
    ) -> pd.DataFrame:
        if not self.record_level_score:
            _ = self.get_score(candidates, references)
        all_scores = self.record_level_score
        all_scores["id"] = ids
        return pd.DataFrame(all_scores)

    def compute_record_level_scores(
        self,
        candidates: List[str],
        references: List[str],
    ) -> List[float]:
        """
        Compute record-level execution accuracy scores.

        Args:
            candidates (List[str]): Generated SQL strings.
            references (List[str]): Reference SQL strings.

        Returns:
            List[float]: A list where each item is 1.0 if execution was correct, else 0.0.
        """
        scores = evaluate(
            glist=references,
            plist=candidates,
            db_dir=SPIDER_DB_DIR,
            etype="all",
            table=os.path.join(SPIDER_DATA_DIR, "tables.jsonl"),
        )
        return [float(x) for x in scores.get("per_record_ex", [])]

