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
    
    def __init__(self, **data):
        super().__init__(**data)

    async def get_score(self, candidates, references, dataset_name=None, model_name=None) -> dict:
        """Get overall score.

        Args:
            candidates: generated text list
            references: reference text list
            dataset_name: optional dataset name for progress bar
            model_name: optional model name for progress bar

        Returns:
            vary based on implementation, should be a dict
        """
        assert self.name is not None
        assert len(candidates) == len(references)
        
        if not self.record_level_scores:
            self.record_level_scores = await self.compute_record_level_scores(candidates, references, dataset_name, model_name)

        res = {}
        for name, score_list in self.record_level_scores.items():
            assert isinstance(score_list, list)
            score_list = [score for score in score_list if score is not None]
            score = util.smart_round(sum(score_list) / len(score_list)) if len(score_list) else 0.0
            res[name] = score
        return res


    @abstractmethod
    def compute_record_level_scores(self, candidates: list, references: list) -> dict[str, list | None]:
        raise NotImplementedError("Subclasses must implement compute_record_level_scores")
