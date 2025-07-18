import nltk
import json
from pathlib import Path
from itertools import zip_longest
import json, re
from pathlib import Path
from nltk.translate.meteor_score import single_meteor_score
from metrics.metrics import Metrics
from utils import util
from tqdm import tqdm
from utils.logging import write_record_log, append_final_score

class MeteorScore(Metrics):
    def __call__(self, candidates, references, *, dataset_name: str | None = None, model_name: str | None = None):
        overall = self.compute_record_level_scores(candidates, references)
        if dataset_name and model_name:
            scores = overall.get(self.name, [])
            # write_record_log will also write to run.log internally
            write_record_log(self, references, candidates, scores, dataset_name, model_name)
            # Directly call append_final_score
            append_final_score(self, overall, dataset_name, model_name)
        return overall



    def __init__(self):
        super().__init__()
        self.name = "meteor"
        self.scorer = single_meteor_score
        nltk.download("wordnet")
        nltk.download('punkt')
        nltk.download('punkt_tab')
        

    def compute_record_level_scores(self, candidates: list, references: list) -> dict[str, list | None]:
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
            score = self.scorer(references[i], candidates[i])
            score = util.smart_round(score)
            score_list.append(score)

        return {self.name: score_list}
