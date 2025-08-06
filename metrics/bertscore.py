"""BertScore metric implementation for text evaluation."""
from tqdm import tqdm
from bert_score import score

from metrics.metrics import Metrics
from metrics.word_error_rate_metrics import normalize_text


class BertScore(Metrics):
    def __call__(self, candidates, references, instructions=None, *, dataset_name: str | None = None, model_name: str | None = None, model_responses=None):
        # Store instructions and model_responses for potential later use
        self.instructions = instructions
        self.model_responses = model_responses if model_responses else []
        
        # Get individual scores
        self.record_level_scores = self.compute_record_level_scores(candidates, references)
        
        # Calculate the mean score directly to avoid async issues
        scores = self.record_level_scores.get(self.name, [])
        valid_scores = [score for score in scores if score is not None]
        mean_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
        overall_score = {self.name: mean_score}
        
        if dataset_name and model_name:
            # write_record_log will also write to run.log internally
            write_record_log(self, references, candidates, scores, dataset_name, model_name, 
                           instructions=self.instructions, model_responses=self.model_responses)
            # Directly call append_final_score with the aggregate score
            append_final_score(self, overall_score, dataset_name, model_name, self.model_responses)
        
        # Return both individual scores and the aggregate score
        return {**self.record_level_scores, **overall_score}

    def __init__(self):
        super().__init__()
        self.name = "bertscore"
        self.scorer = score
        self.instructions = None
        self.model_responses = None
        self.record_level_scores = None

    def __call__(self, candidates, references, instructions=None, *, dataset_name: str | None = None, model_name: str | None = None, model_responses=None):
        # Store instructions and model_responses for potential later use
        self.instructions = instructions
        self.model_responses = model_responses if model_responses else []

        # Get individual scores
        self.record_level_scores = self.compute_record_level_scores(candidates, references)

        # Calculate the mean score directly to avoid async issues
        scores = self.record_level_scores.get(self.name, [])
        valid_scores = [score for score in scores if score is not None]
        mean_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
        overall_score = {self.name: mean_score}

        if dataset_name and model_name:
            # write_record_log will also write to run.log internally
            write_record_log(self, references, candidates, scores, dataset_name, model_name,
                           instructions=self.instructions, model_responses=self.model_responses)
            # Directly call append_final_score with the aggregate score
            append_final_score(self, overall_score, dataset_name, model_name)

        # Return both individual scores and the aggregate score
        return {**self.record_level_scores, **overall_score}

    def compute_record_level_scores(self, candidates: list, references: list) -> dict[str, list | None]:
        """Compute the scores that should be saved in the record level file.

        Args:
            candidates: Generated text from the model
            references: Reference text from the dataset

        Returns:
            Scores for each record. The keys should be the column names that will be saved in the record level file.
        """

        score_list = []
        for i in tqdm(range(len(candidates)), desc="BERTSCORE"):
            # === Consistent normalization with WER processing ===
            reference, candidate = references[i], candidates[i]
            norm_reference = normalize_text(reference)
            norm_candidate = normalize_text(candidate)

            _, _, f1 = self.scorer([norm_reference], [norm_candidate],
                                   model_type='bert-base-multilingual-cased')
            f1_score = f1.numpy().tolist()
            score_list.extend(f1_score)
        return {self.name: score_list}
