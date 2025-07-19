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
from utils.custom_logging import write_record_log, append_final_scorefrom metrics.word_error_rate_metrics import normalize_text


class MeteorScore(Metrics):
    def __call__(self, candidates, references, instructions=None, *, dataset_name: str | None = None, model_name: str | None = None):
        # Store instructions for potential later use
        self.instructions = instructions
        overall = self.compute_record_level_scores(candidates, references)
        if dataset_name and model_name:
            scores = overall.get(self.name, [])
            # write_record_log will also write to run.log internally
            write_record_log(self, references, candidates, scores, dataset_name, model_name, instructions=self.instructions)
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
            
            #=== Consistent normalization with WER processing === 
            reference, candidate = references[i], candidates[i]
            norm_reference = normalize_text(reference)
            norm_candidate = normalize_text(candidate) 

            # Compute METEOR Score
            score = self.scorer(norm_reference.split(), norm_candidate.split())
            score = util.smart_round(score)
            score_list.append(score)

        return {self.name: score_list}
