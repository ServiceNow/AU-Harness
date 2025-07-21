from abc import ABC, abstractmethod
from operator import itemgetter

import pandas as pd
from pydantic import Field

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
from metrics.base_metric_metadata import MetricMetadata
from utils import util


class Metrics(ABC, MetricMetadata):
    """Standard Metrics Base Class."""
    
    record_level_scores: dict = Field(default_factory=dict, exclude=True, description="Record level scores")
    contexts: list[dict] = Field(default_factory=list, exclude=True, description="Contexts for the metric")
    params: dict = Field(default_factory=dict, exclude=True, description="Parameters for the metric")
    model_responses: list = Field(default_factory=list, exclude=True, description="Model responses from inference")
    
    def __init__(self, **data):
        super().__init__(**data)

    def set_params(self, params: dict):
        """Set parameters for the metric.

        Args:
            params: dictionary of parameters
        """
        self.params = params

    def set_contexts(self, contexts: list[dict]):
        """Add additional columns from the dataset which can be leveraged in compute_record_level_scores."""
        self.contexts = contexts

    def get_score(self, candidates, references) -> dict:
        """Get overall score.

        Args:
            candidates: generated text list
            references: reference text list

        Returns:
            vary based on implementation, should be a dict
        """
        assert self.name is not None
        assert len(candidates) == len(references)
        if not self.record_level_scores:
            self.record_level_scores = self.compute_record_level_scores(candidates, references)

        res = {}
        for name, score_list in self.record_level_scores.items():
            assert isinstance(score_list, list)
            score_list = [score for score in score_list if score is not None]
            score = util.smart_round(sum(score_list) / len(score_list)) if len(score_list) else 0.0
            res[name] = score
        return res

    def get_score_by_source(self, candidates: list[str], references: list[str], split_by_key: str) -> dict:
        """Get scores at drilled down level.

        Based on user input's 'split_by_key' parameter, it would subset candidates according to unique values of 'key'
        and then calculate metrics for each one of them separately.

        Args:
            candidates: List of Predictions
            references: List of Ground Truths
            split_by_key: Key by which metrics computation needs to be split.

        Returns:
            Dictionary where keys are all unique values in the column 'split_by_key' and 'overall' score and their values.
        """
        if len(self.contexts) > 0 and split_by_key in self.contexts[0]:
            all_scores = {}
            unique_sources = list({c[split_by_key] for c in self.contexts})
            for _, ds in enumerate(unique_sources):
                indices = util.get_context_indices_for_filter(split_by_key, ds, self.contexts)

                candidates_for_ds = itemgetter(*indices)(candidates)
                references_for_ds = itemgetter(*indices)(references)
                if len(indices) == 1:
                    candidates_for_ds = [candidates_for_ds]
                    references_for_ds = [references_for_ds]

                # sometimes we compare options, which can be number or float
                candidates_for_ds = [str(c).lower().strip() for c in candidates_for_ds]
                references_for_ds = [str(r).lower().strip() for r in references_for_ds]

                score = self.get_score(candidates_for_ds, references_for_ds)[self.name]
                all_scores[ds] = score

            all_scores["overall"] = self.get_score(candidates, references)[self.name]
        else:
            raise ValueError(
                f"Expects contexts to be initialized and containing '{split_by_key}' key within it but wasn't found."
            )
        return all_scores

    @abstractmethod
    def compute_record_level_scores(self, candidates: list, references: list) -> dict[str, list | None]:
        """Compute the scores that should be saved in the record level file.

        Args:
            candidates: Generated text from the model
            references: Reference text from the dataset

        Returns:
            Scores for each record. The keys should be the column names that will be saved in the record level file.
        """
        return {}

    def get_all_score_df(self, ids, candidates, references):
        """Get score for each record.

        Args:
            ids: ids for all the records in df
            candidates: generated text list
            references: reference text list

        Returns:
            returns a panda dataframe including id column
        """
        assert self.name is not None
        assert len(candidates) == len(references)
        df = pd.DataFrame()
        if not self.record_level_scores:
            self.record_level_scores = self.compute_record_level_scores(candidates, references)
        all_scores = self.record_level_scores
        if all_scores.values():
            all_scores["id"] = ids
            df = pd.DataFrame(all_scores)
        return df
