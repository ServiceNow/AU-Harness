"""
    Acknowledgement: We adopt the WDER and cpWER and SpkCntMAE metrics from DiarizationLM work and integrate into our evaluation kit.
    Source: https://github.com/google/speaker-id/tree/master/DiarizationLM
"""

import dataclasses

import numpy as np
import word_levenshtein as levenshtein
from scipy import optimize
from tqdm import tqdm

from metrics.metrics import Metrics
from metrics.word_error_rate_metrics import normalize_text
from utils.custom_logging import write_record_log, append_final_score


@dataclasses.dataclass
class UtteranceMetrics:
    """Metrics for one utterance/ row."""

    wer_insert: int = 0
    wer_delete: int = 0
    wer_sub: int = 0
    wer_correct: int = 0
    wer_total: int = 0

    wder_sub: int = 0
    wder_correct: int = 0
    wder_total: int = 0

    cpwer_insert: int = 0
    cpwer_delete: int = 0
    cpwer_sub: int = 0
    cpwer_correct: int = 0
    cpwer_total: int = 0

    speaker_count_error: int = 0


def merge_cpwer(
        wer_metrics: list[UtteranceMetrics], cpwer_metrics: UtteranceMetrics
) -> None:
    """Compute cpWER metrics by merging a list of WER metrics."""
    for utt in wer_metrics:
        cpwer_metrics.cpwer_insert += utt.wer_insert
        cpwer_metrics.cpwer_delete += utt.wer_delete
        cpwer_metrics.cpwer_sub += utt.wer_sub
        cpwer_metrics.cpwer_correct += utt.wer_correct
        cpwer_metrics.cpwer_total += utt.wer_total


def compute_wer(
        hyp_text_list: list, ref_text_list: list
) -> tuple[UtteranceMetrics, list[tuple[int, int]]]:
    """Compute the word error rate (WER) of an utterance."""
    result = UtteranceMetrics()
    hyp_text = ' '.join(hyp_text_list)
    ref_text = ' '.join(ref_text_list)

    hyp_normalized = normalize_text(hyp_text)
    ref_normalized = normalize_text(ref_text)
    hyp_words = hyp_normalized.split(' ')
    ref_words = ref_normalized.split(' ')

    # Get the alignment.
    _, align = levenshtein.levenshtein_with_edits(ref_normalized, hyp_normalized)

    # Apply the alignment on ref speakers.
    for i, j in align:
        if i == -1:
            result.wer_insert += 1
        elif j == -1:
            result.wer_delete += 1
        else:
            if ref_words[i] == hyp_words[j]:
                result.wer_correct += 1
            else:
                result.wer_sub += 1

    result.wer_total = result.wer_correct + result.wer_sub + result.wer_delete
    assert result.wer_total == len(ref_words)
    return result, align


def compute_wder(ref_spk_list, hyp_spk_list, ref_words, hyp_words, align, result):
    """Compute WDER, cpWER and spkcnterr metrics by merging a list of WER metrics."""
    hyp_spk_list_align = []
    ref_spk_list_align = []

    for i, j in align:
        if i != -1 and j != -1:
            ref_spk_list_align.append(ref_spk_list[i])
            hyp_spk_list_align.append(hyp_spk_list[j])

    # Build cost matrix.
    max_spk = max(*ref_spk_list_align, *hyp_spk_list_align)
    cost_matrix = np.zeros((max_spk, max_spk), dtype=int)
    for aligned, original in zip(ref_spk_list_align, hyp_spk_list_align):
        cost_matrix[aligned - 1, original - 1] += 1

    # Solve alignment.
    row_index, col_index = optimize.linear_sum_assignment(
        cost_matrix, maximize=True
    )
    result.wder_correct = int(cost_matrix[row_index, col_index].sum())
    result.wder_total = len(ref_spk_list_align)
    result.wder_sub = result.wder_total - result.wder_correct

    #### Compute cpwer
    spk_pair_metrics = {}
    cost_matrix = np.zeros((max_spk, max_spk), dtype=int)
    for i in range(1, max_spk + 1):
        ref_words_for_spk = [
            ref_words[k] for k in range(len(ref_words)) if ref_spk_list[k] == i
        ]
        if not ref_words_for_spk:
            continue
        for j in range(1, max_spk + 1):
            hyp_words_for_spk = [
                hyp_words[k] for k in range(len(hyp_words)) if hyp_spk_list[k] == j
            ]
            if not hyp_words_for_spk:
                continue
            spk_pair_metrics[(i, j)], _ = compute_wer(
                hyp_text_list=hyp_words_for_spk,
                ref_text_list=ref_words_for_spk,
            )
            cost_matrix[i - 1, j - 1] = spk_pair_metrics[(i, j)].wer_correct

    # Solve alignment.
    row_index, col_index = optimize.linear_sum_assignment(
        cost_matrix, maximize=True
    )
    metrics_to_concat = []
    for r, c in zip(row_index, col_index):
        if (r + 1, c + 1) not in spk_pair_metrics:
            continue
        metrics_to_concat.append(spk_pair_metrics[(r + 1, c + 1)])

    merge_cpwer(metrics_to_concat, result)

    ########################################
    # Compute speaker count error.
    ########################################
    hyp_spk_count = len(set(hyp_spk_list))
    ref_spk_count = len(set(ref_spk_list))
    result.speaker_count_error = hyp_spk_count - ref_spk_count
    return result


def prepare_speaker_info(input_info, lang_code):
    """
        Extracting relevant speaker information. Reusable for both reference and hypothesis.
        
        Args:
            input_info: Input utterance, starting with speaker label (i.e. A: Hello world)
            lang_code: language code for different normalization processors (if needed)

        Returns:
            speaker: speaker ID information 
            word_level_speaker: sequence of speaker labels aligned with words in utterances
            cleaned_transcript: Transcript after normalization process (similar to WER processing)
    """
    split_input_info = input_info.split(':')
    speaker = split_input_info[0].strip()  # [A: hello world]
    cleaned_transcript = normalize_text(' '.join(split_input_info[1:]).strip(), lang_code)
    word_level_speaker = [speaker] * len(cleaned_transcript.split(' '))
    return speaker, word_level_speaker, cleaned_transcript


class DiarizationMetrics(Metrics):
    """
        Diarization Error Rate (DER) metric variations for LLMs. 
        Supporting
            1. WDER: Word-level Diarization Error Rate
            2. cpDER: Concatenated minium-permutation Word Error Rate
            3. SpkCntMAE: Speaker Count Mean Absolute Error 
    """

    def __call__(self, candidates, references, instructions=None, *, dataset_name: str | None = None,
                 model_name: str | None = None, model_responses=None):
        # Store instructions and model_responses for potential later use
        self.instructions = instructions
        self.model_responses = model_responses if model_responses else []
        overall = self.get_score(candidates, references)

        if dataset_name and model_name:
            # WDER & cpWER record scores are stored under 'wder_per_row' and 'cpwer_per_row'
            scores = self.record_level_scores.get("wder_per_row", [])

            write_record_log(self, references, candidates, scores, dataset_name, model_name,
                             instructions=self.instructions)
            # Directly call append_final_score
            append_final_score(self, overall, dataset_name, model_name)
        return overall

    def __init__(self, language="en"):
        super().__init__()
        self.name = "diarization_metrics"
        self.display_name = "Word Diarization Error Rate (WDER) and concatenated minium-permutation WER (cpWER) and Speaker Count MAE (SpkCntMAE)"
        self.description = "The proportion of incorrectly predicted speakers when compared to the reference speakers, on the fine-grained word level"
        self.language = language
        self.instructions = None
        self.model_responses = []

    def get_score(self, candidates, references, ids=None, lengths=None):
        """Get overall score.

        Args:
            candidates: generated text list
            references: reference text list
            ids: optional list of conversation IDs (first 4 letters)
            lengths: optional list of audio sample lengths in seconds

        Returns:
            Dict with diarization-relevant metrics.
        """
        scores = self.compute_record_level_scores(candidates, references)
        result = {}

        # Overall WDER
        wder_total = sum(scores['wder_total'])
        overall_wder = sum(scores['wder_sub']) / wder_total if wder_total > 0 else 0

        # Overall CPWER
        cpwer_total = sum(scores['cpwer_total'])
        overall_cpwer = (sum(scores['cpwer_sub']) + sum(scores['cpwer_insert'])) / cpwer_total if cpwer_total > 0 else 0

        avg_sample_wder = sum(scores['wder_per_row']) / len(scores['wder_per_row']) if scores['wder_per_row'] else 0
        avg_sample_wder = min(avg_sample_wder, 1.0)

        avg_sample_cpwer = sum(scores['cpwer_per_row']) / len(scores['cpwer_per_row']) if scores['cpwer_per_row'] else 0
        avg_sample_cpwer = min(avg_sample_cpwer, 1.0)

        # Speaker count MAE
        avg_speaker_count_absolute_error = sum(scores['spkcnterr']) / len(scores['spkcnterr']) if scores[
            'spkcnterr'] else 0
        avg_speaker_count_absolute_error = min(avg_speaker_count_absolute_error, 1.0)

        result = {
            "average_sample_wder": avg_sample_wder,
            "overall_wder": overall_wder,
            "average_sample_cpwer": avg_sample_cpwer,
            "overall_cpwer": overall_cpwer,
            "speaker_count_absolute_error": avg_speaker_count_absolute_error,
        }

        # # Store the scores for later record level reporting
        # # Important to use setdefault which is a no-op if the value already exists
        # # As users can evaluate multiple models and call compute_record_level_scores multiple times
        self.record_level_scores.setdefault("wder_per_row", scores["wder_per_row"])
        self.record_level_scores.setdefault("cpwer_per_row", scores["cpwer_per_row"])

        return result

    def compute_record_level_scores(self, candidates: list, references: list):
        """Compute the scores that should be saved in the record level file.

        Args:
            candidates: Generated text from the model (with speaker info). Ex: A: How are you doing\nB: Good. How are you?
            references: Reference text from the dataset (with speaker info)

        Returns:
            Scores for each record. The keys should be the column names that will be saved in the record level file.
        """

        cpwer_scores, wder_scores = [], []
        total_wder_sub, total_wder_total = [], []
        total_cpwer_sub, total_cpwer_insert, total_cpwer_total = [], [], []
        total_spkcnterr = []

        for i, (reference, candidate) in enumerate(
                tqdm(zip(references, candidates), desc="diarization_metrics", total=len(references))):
            lang_code = getattr(self, 'language', 'en')
            ref_by_lines = reference.split('\n')
            cand_by_lines = candidate.split('\n')

            ##########################################################
            # Extract speakers by turns from references and hypotheses
            ##########################################################
            num_min_iterations = min(len(ref_by_lines), len(cand_by_lines))
            ref_speakers, cand_speakers = [], []
            cleaned_ref_transcripts, cleaned_cand_transcripts = [], []
            word_level_ref_speakers, word_level_cand_speakers = [], []

            for i in range(num_min_iterations):
                cur_ref_speaker, cur_word_level_ref_speaker, normalized_ref_transcript = prepare_speaker_info(
                    ref_by_lines[i], lang_code)
                cur_cand_speaker, cur_word_level_cand_speaker, normalized_cand_transcript = prepare_speaker_info(
                    cand_by_lines[i], lang_code)

                ref_speakers.append(cur_ref_speaker)
                word_level_ref_speakers.extend(cur_word_level_ref_speaker)
                cleaned_ref_transcripts.append(normalized_ref_transcript)

                cand_speakers.append(cur_cand_speaker)
                word_level_cand_speakers.extend(cur_word_level_cand_speaker)
                cleaned_cand_transcripts.append(normalized_cand_transcript)

            if len(ref_by_lines) > num_min_iterations:
                for j in range(len(ref_by_lines) - num_min_iterations):
                    cur_ref_speaker, cur_word_level_ref_speaker, normalized_ref_transcript = prepare_speaker_info(
                        ref_by_lines[num_min_iterations + j], lang_code)
                    ref_speakers.append(cur_ref_speaker)
                    word_level_ref_speakers.extend(cur_word_level_ref_speaker)
                    cleaned_ref_transcripts.append(normalized_ref_transcript)

            else:
                for j in range(len(cand_by_lines) - num_min_iterations):
                    cur_cand_speaker, cur_word_level_cand_speaker, normalized_cand_transcript = prepare_speaker_info(
                        cand_by_lines[num_min_iterations + j], lang_code)
                    cand_speakers.append(cur_ref_speaker)
                    word_level_cand_speakers.extend(cur_word_level_cand_speaker)
                    cleaned_cand_transcripts.append(normalized_cand_transcript)

            try:
                assert len(ref_by_lines) == len(ref_speakers)
            except Exception as exc:
                raise ValueError(
                    "The ground truths are not labeled correctly. Reference speakers and reference transcripts are not aligned by turns.") from exc
            try:
                assert len(cand_by_lines) == len(cand_speakers)
            except Exception as exc:
                raise ValueError(
                    "The generated outputs are not formatted correctly. Hypothesis speakers and hypothesis transcripts are not aligned by turns.") from exc

            ##########################################################
            # Flattten the transcripts to the word-level
            # Aligning with word-level speaker-ids
            ##########################################################
            flattened_ref_transcripts = [x for s in cleaned_ref_transcripts for x in s.split(' ')]
            flattened_cand_transcripts = [x for s in cleaned_cand_transcripts for x in s.split(' ')]

            all_possible_speakers = list(set(word_level_ref_speakers).union(set(word_level_cand_speakers)))

            # spkr count needs to start from 1 to n for wder metrics to work
            numeric_word_level_ref_speakers = [all_possible_speakers.index(s) + 1 for s in word_level_ref_speakers]
            numeric_word_level_cand_speakers = [all_possible_speakers.index(s) + 1 for s in word_level_cand_speakers]

            try:
                assert len(flattened_ref_transcripts) == len(numeric_word_level_ref_speakers)
                assert len(flattened_cand_transcripts) == len(numeric_word_level_cand_speakers)
            except Exception as exc:
                raise ValueError("Either reference transcripts or candidate transcripts are not spaced correctly.") from exc
            result, align = compute_wer(flattened_cand_transcripts, flattened_ref_transcripts)

            result = compute_wder(numeric_word_level_ref_speakers, numeric_word_level_cand_speakers,
                                  flattened_ref_transcripts, flattened_cand_transcripts, align, result)
            per_row_wder = result.wder_sub / (result.wder_total + 1e-12)
            per_row_cpwer = (result.cpwer_sub + result.cpwer_insert) / (result.cpwer_total)

            cpwer_scores.append(per_row_cpwer)
            wder_scores.append(per_row_wder)

            total_wder_sub.append(result.wder_sub)
            total_wder_total.append(result.wder_total)

            total_cpwer_sub.append(result.cpwer_sub)
            total_cpwer_insert.append(result.cpwer_insert)
            total_cpwer_total.append(result.cpwer_total)
            total_spkcnterr.append(abs(result.speaker_count_error))

        results = {
            'cpwer_per_row': cpwer_scores,
            'wder_per_row': wder_scores,

            'wder_sub': total_wder_sub,
            'wder_total': total_wder_total,

            'cpwer_sub': total_cpwer_sub,
            'cpwer_insert': total_cpwer_insert,
            'cpwer_total': total_cpwer_total,

            'spkcnterr': total_spkcnterr,
        }

        return results
