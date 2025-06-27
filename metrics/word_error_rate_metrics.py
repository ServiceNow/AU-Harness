import re
from collections import defaultdict

from jiwer import (
    Compose,
    ReduceToListOfListOfChars,
    RemovePunctuation,
    RemoveWhiteSpace,
    Strip,
    ToLowerCase,
    process_words,
)
from num2words import num2words

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
from metrics.base_metric_metadata import MetricMetadata
from metrics.metrics import Metrics
from metrics.wer.normalizers import JapaneseTextNormalizer
from metrics.wer.whisper_normalizer.english import EnglishTextNormalizer
from metrics.wer.whisper_normalizer.basic import BasicTextNormalizer

from utils import constants
from utils.util import smart_round

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
    """Convert unicode (\u00e9) to characters (é)."""
    return text.encode("raw_unicode_escape").decode("unicode-escape")


def convert_digits_to_words(text: str, language: str):
    if language is "":
        return text
    """Convert numbers to words (e.g., "3" to "three")."""
    return re.sub(r"\d+", lambda m: num2words(int(m.group()), lang=language), text)


def normalize_text(text: str, language: str) -> str:
    """Normalize text based on language.

    Args:
        text: input text
        language: language code
    """
    normalizer = NORMALIZERS.get(language, DEFAULT_NORMALIZER)
    #logger.info(f"[normalize_text] Normalizing text: {text}")
    text = convert_unicode_to_characters(text)
    text = convert_digits_to_words(text, language)
    return BASIC_TRANSFORMATIONS([normalizer(text)])[0]


class WERMetrics(Metrics):
    def __call__(self, candidates, references):
        logger.info(f"[WERMetrics.__call__] Calculating WER for {len(candidates)} samples.")
        return self.get_score(candidates, references)
    """Word Error Rate metric class, used for transcription tasks."""

    def __init__(self, language="en"):
        super().__init__()
        self.name = "WER"
        self.display_name = "Word Error Rate"
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

    def get_score(self, candidates, references) -> dict:
        """Get overall score.

        Args:
            candidates: generated text list
            references: reference text list

        Returns:
            vary based on implementation, should be a dict
        """
        if not self.record_level_scores:
            self.record_level_scores = self.compute_record_level_scores(candidates, references)

        wer_per_row = self.record_level_scores["wer_per_row"]
        incorrect = self.record_level_scores["incorrect"]
        total = self.record_level_scores["total"]

        results = {
            "wer": sum(incorrect) / sum(total),
            "average_wer_per_row": smart_round(sum(wer_per_row) / len(wer_per_row)) if len(wer_per_row) else 0.0,
        }

        results.update(self.compute_attributes(incorrect, total, ["accent", "gender"]))
        return results

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

        for i, (reference, candidate) in enumerate(zip(references, candidates)):
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
            logger.info(f"For sample {i}: reference={reference} candidate={candidate}")
            scores.append(incorrect_scores[-1] / total_scores[-1])

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
        overall_score.pop("average_wer_per_row")
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
