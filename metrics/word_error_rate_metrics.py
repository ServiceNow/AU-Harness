import unicodedata
from collections import defaultdict
import re
from jiwer import (
    Compose,
    ReduceToListOfListOfChars,
    RemovePunctuation,
    RemoveWhiteSpace,
    Strip,
    ToLowerCase,
    process_words,
)
from tqdm import tqdm
from num2words import num2words
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
from metrics.base_metric_metadata import MetricMetadata
from metrics.metrics import Metrics
from metrics.wer.normalizers import JapaneseTextNormalizer
from metrics.wer.whisper_normalizer.english import EnglishTextNormalizer
from metrics.wer.whisper_normalizer.basic import BasicTextNormalizer
from utils.logging import write_record_log, append_final_score

from utils import constants

language_map = {
    "en": ["english", "en"],
    "ja": ["japanese", "ja"],
}
NORMALIZERS = {"en": EnglishTextNormalizer(), "ja": JapaneseTextNormalizer()}
DEFAULT_NORMALIZER = BasicTextNormalizer()
BASIC_TRANSFORMATIONS = Compose(
    [
        ToLowerCase(),
        RemovePunctuation(),
        Strip(),
    ]
)
# CER stands for Character Error Rate
CER_LANGUAGES = {"ja"}
CER_DEFAULTS = Compose(
    [
        RemoveWhiteSpace(),
        ReduceToListOfListOfChars(),
    ]
)


def convert_unicode_to_characters(text: str) -> str:
    """Convert unicode to composed form."""
    try:
        return unicodedata.normalize("NFC", text)
    except Exception as e:
        # Optionally log the error
        logger.warning(f"Unicode normalization failed: {e}. Returning original text.")
        return text


def convert_digits_to_words(text: str, language: str):
    if not language:
        return text
    """Convert numbers to words (e.g., "3" to "three")."""
    try:
        return re.sub(r"\d+", lambda m: num2words(int(m.group()), lang=language), text)
    except Exception as e:
        logger.info(f"Failed to convert digits to words for language {language} - continuing...")
        logger.warning(f"Non-fatal error: {e} - continuing...")
        return text


def normalize_text(text: str, language: str) -> str:
    """Normalize text based on language.

    Args:
        text: input text
        language: language code
    """
    language = language_map.get(language, language)
    normalizer = NORMALIZERS.get(language, DEFAULT_NORMALIZER)
    text = convert_unicode_to_characters(text)
    text = convert_digits_to_words(text, language)
    return BASIC_TRANSFORMATIONS([normalizer(text)])[0]


class WERMetrics(Metrics):
    def __call__(self, candidates, references, ids=None, lengths=None, *, dataset_name: str | None = None, model_name: str | None = None):
        overall = self.get_score(candidates, references, ids, lengths)
        if dataset_name and model_name:
            # WER record scores are stored under 'wer_per_row'
            scores = self.record_level_scores.get("wer_per_row", [])
            # write_record_log will also write to run.log internally
            write_record_log(self, references, candidates, scores, dataset_name, model_name)
            # Directly call append_final_score for the overall metric
            append_final_score(self, overall, dataset_name, model_name)
        return overall



    def __init__(self, language="en"):
        super().__init__()
        self.name = "word_error_rate"
        self.lower_better = True
        self.description = "The proportion of words that are incorrectly predicted, when compared to the reference text. The dataset is considered as one big conversation."
        self.language = language

    def compute_attributes(self, incorrect: list[int | float], total: list[int | float], attributes: list[str]) -> dict:
        """Compute the attributes (e.g., accent, gender) that should be saved in the record level file for analysis."""
        results = {}
        for attribute in attributes:
            current_attr = self.record_level_scores.get(attribute, [])
            incorrect_per_attr = defaultdict(int)
            total_per_attr = defaultdict(int)
            for _incorrect, _total, attr_value in zip(incorrect, total, current_attr):
                if attr_value:
                    incorrect_per_attr[attr_value] += _incorrect
                    total_per_attr[attr_value] += _total

            for attr in incorrect_per_attr:
                total_attr = total_per_attr.get(attr, 0)
                if total_attr:
                    results[f"wer_{attribute}_{attr}"] = incorrect_per_attr[attr] / total_attr
        return results

    def get_score(self, candidates, references, ids=None, lengths=None):
        """Get overall score.

        Args:
            candidates: generated text list
            references: reference text list
            ids: optional list of conversation IDs (first 4 letters)
            lengths: optional list of audio sample lengths in seconds

        Returns:
            Dict with WER metrics by overall, conversation, and length buckets
        """
        scores = self.compute_record_level_scores(candidates, references)

        # Compute the overall WER
        incorrect_chars = sum(scores["incorrect"])
        total_chars = sum(scores["total"])
        # Overall WER is the sum of incorrect divided by sum of total
        overall_wer = incorrect_chars / total_chars if total_chars > 0 else 0
        # Cap at 1.0
        overall_wer = min(overall_wer, 1.0)
        
        # We also track per-sample average for a more balanced view
        avg_sample_wer = sum(scores["wer_per_row"]) / len(scores["wer_per_row"]) if scores["wer_per_row"] else 0
        # Cap the WER at 1.0
        avg_sample_wer = min(avg_sample_wer, 1.0)

        # Initialize the result with both WER metrics
        result = {
            "average_sample_wer": avg_sample_wer,
            "overall_wer": overall_wer
        }

        if ids and len(ids) == len(scores["wer_per_row"]):
            conversation_wer = {}
            # Group WERs by conversation ID
            id_to_wers = defaultdict(list)
            id_to_incorrect = defaultdict(int)
            id_to_total = defaultdict(int)
            
            for i, conv_id in enumerate(ids):
                if i < len(scores["wer_per_row"]) and i < len(scores["incorrect"]) and i < len(scores["total"]):
                    id_to_wers[conv_id].append(scores["wer_per_row"][i])
                    id_to_incorrect[conv_id] += scores["incorrect"][i]
                    id_to_total[conv_id] += scores["total"][i]
            
            # Calculate average WER for each conversation ID
            for conv_id, wers in id_to_wers.items():
                # Using ratio of sums for conversation WER
                conv_wer = id_to_incorrect[conv_id] / id_to_total[conv_id] if id_to_total[conv_id] > 0 else 0
                # Cap at 1.0
                conv_wer = min(conv_wer, 1.0)
                conversation_wer[conv_id] = conv_wer
            
            result["conversation_wer"] = conversation_wer
        
        # If lengths are provided, calculate WER by length buckets
        if lengths and len(lengths) == len(scores["wer_per_row"]):
            # Define length buckets
            buckets = [(0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2), (2, 3), (3, float('inf'))]
            bucket_labels = ["0-0.5", "0.5-1", "1-1.5", "1.5-2", "2-3", "3+"]
            length_wer = {}
            
            # Group WERs by length bucket
            bucket_to_incorrect = {label: 0 for label in bucket_labels}
            bucket_to_total = {label: 0 for label in bucket_labels}
            
            for i, length in enumerate(lengths):
                if i < len(scores["wer_per_row"]) and i < len(scores["incorrect"]) and i < len(scores["total"]):
                    # Find which bucket this length belongs to
                    bucket_idx = next((j for j, (min_len, max_len) in enumerate(buckets) 
                                      if min_len <= length < max_len), len(buckets) - 1)
                    bucket_label = bucket_labels[bucket_idx]
                    
                    bucket_to_incorrect[bucket_label] += scores["incorrect"][i]
                    bucket_to_total[bucket_label] += scores["total"][i]
            
            # Calculate WER for each length bucket
            for bucket_label in bucket_labels:
                if bucket_to_total[bucket_label] > 0:
                    bucket_wer = bucket_to_incorrect[bucket_label] / bucket_to_total[bucket_label]
                    # Cap at 1.0
                    bucket_wer = min(bucket_wer, 1.0)
                    length_wer[bucket_label] = bucket_wer
                else:
                    length_wer[bucket_label] = 0.0
            
            result["length_wer"] = length_wer

        # Store the scores for later record level reporting
        # Important to use setdefault which is a no-op if the value already exists
        # As users can evaluate multiple models and call compute_record_level_scores multiple times
        self.record_level_scores.setdefault("wer_per_row", scores["wer_per_row"])
        self.record_level_scores.setdefault("incorrect", scores["incorrect"])
        self.record_level_scores.setdefault("total", scores["total"])
        return result

    def compute_record_level_scores(self, candidates: list, references: list):
        """Compute the scores that should be saved in the record level file.

        Args:
            candidates: Generated text from the model
            references: Reference text from the dataset

        Returns:
            Scores for each record. The keys should be the column names that will be saved in the record level file.
        """
        incorrect_scores = []
        total_scores = []
        scores = []
        references_clean = []
        candidates_clean = []

        for i, (reference, candidate) in enumerate(tqdm(zip(references, candidates), desc="word_error_rate", total=len(references))):
            lang_code = getattr(self, 'language', 'en')
            references_clean.append(normalize_text(reference, lang_code))
            candidates_clean.append(normalize_text(candidate, lang_code))
            if references_clean[-1].strip() == "":
                logger.warning(
                    f"After normalization, '{reference}' is empty. Considering all words in '{candidate}' as incorrect."
                )
                incorrect_scores.append(len(candidates_clean[-1].split()))
                total_scores.append(1)
            else:
                kwargs = (
                    {kwarg: CER_DEFAULTS for kwarg in ("truth_transform", "hypothesis_transform")}
                    if lang_code in CER_LANGUAGES
                    else {}
                )
                measures = process_words(references_clean[-1], candidates_clean[-1], **kwargs)

                # Newer jiwer returns a dataclass-like object with attributes
                substitutions = measures.substitutions
                deletions = measures.deletions
                insertions = measures.insertions
                hits = measures.hits

                incorrect_scores.append(substitutions + deletions + insertions)
                total_scores.append(substitutions + deletions + hits)
            wer = incorrect_scores[-1] / total_scores[-1]
            if wer > 1.0:
                wer = 1.0
            scores.append(wer)

        results = {
            "wer_per_row": scores,
            "candidates_clean": candidates_clean,
            "references_clean": references_clean,
            "incorrect": incorrect_scores,
            "total": total_scores,
        }
        accents = [record.get("accent") for record in self.contexts]
        gender = [record.get("gender") for record in self.contexts]
        if any(accents):
            results["accent"] = accents
        if any(gender):
            results["gender"] = gender
        return results

    def get_reporting_summary_score(self, overall_score: dict[str, float]) -> dict:
        """Gets the score to display in wandb. If a metric says lower-is-better, highlight with an ↓.

        Args:
            overall_score: The overall score that was computed for the metric
        Returns:
            The dictionary of columns and values to actually present in wandb
        """
        return overall_score

    def get_metadata(self) -> dict:
        """Return metadata info."""
        metadata = {
            "wer": MetricMetadata(
                name="wer",
                display_name=f"{constants.INVERTED_METRIC_INDICATOR} Word Error Rate",
                description=self.description,
                higher_is_better=False,
            )
        }
        for attribute in ("accent", "gender"):
            current_attr = set(self.record_level_scores.get(attribute, []))
            for attr_value in current_attr:
                if attr_value is not None:
                    metadata[f"wer_{attribute}_{attr_value}"] = MetricMetadata(
                        name=f"wer_{attribute}_{attr_value}",
                        display_name=f"{constants.INVERTED_METRIC_INDICATOR} Word Error Rate {attribute.title()} ({attr_value})",
                        description=self.description,
                        higher_is_better=False,
                    )
        return metadata
