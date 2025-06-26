import nltk
from nltk import word_tokenize
from nltk.translate.meteor_score import single_meteor_score

from metrics.metrics import Metrics
from utils import util


class MeteorScore(Metrics):
    def __call__(self, candidates, references):
        return self.compute_record_level_scores(candidates, references)
    """MeteorScore using nltk tokenizer."""

    def __init__(self):
        super().__init__()
        self.name = "meteor"
        self.scorer = single_meteor_score
        nltk.download("wordnet")

    def compute_record_level_scores(self, candidates: list, references: list) -> dict[str, list | None]:
        """Compute the scores that should be saved in the record level file.

        Args:
            candidates: Generated text from the model
            references: Reference text from the dataset

        Returns:
            Scores for each record. The keys should be the column names that will be saved in the record level file.
        """
        score_list = []
        for i in range(len(candidates)):
            # default preprocess is str.lower()
            # default stemmer is PorterStemmer()
            # default wordnet is nltk.corpus.wordnet
            score = self.scorer(word_tokenize(references[i]), word_tokenize(candidates[i]))
            score = util.smart_round(score)
            score_list.append(score)

        return {self.name: score_list}
