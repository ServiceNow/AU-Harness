import re
import unicodedata
from collections import defaultdict

from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate
from pyannote.core import Annotation, Segment

from metrics.word_error_rate_metrics import normalize_text

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
from metrics.base_metric_metadata import MetricMetadata
from metrics.metrics import Metrics

from utils import constants


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
    speaker = split_input_info[0].strip() # [A: hello world]
    cleaned_transcript = normalize_text(' '.join(split_input_info[1:]).strip(), lang_code)
    word_level_speaker = [speaker] * len(cleaned_transcript.split(' '))
    return speaker, word_level_speaker, cleaned_transcript

def convert_speaker_list_to_diarization_tuples(turn_history):
    """
        Convert the list of speakers into tuples and Annotation format for diarization metric application.
        Ex: [A,A,B,B] into [(A,0,1), (B,2,3)] and {'A':[Segment(0,1)], 'B':[Segment(2,3)]}

        **NOTE**: For Diarization to work, we introduce margin to replicate the time segments (avoiding scenarios such as (1,1) 
        which can cause unexpected behaviors from the diarization metric function calls. 

        Args:
            turn_history: Speaker sequence of the given conversation/dialogue (i.e. [A,B,A,B])
        
        Returns:
            tuple_output: List of tuples where each tuple is defined as (speaker, start_idx, end_idx)
            annotation_examples: Annotation() object for pyannote.metrics. Example: Annotation[Segment(1.0,1.0)] = 'A'
    """

    if (len(turn_history) == 1):
        return [(turn_history[0],0.0,0.0)]
    start,end = 0,1
    current_speaker = turn_history[start]
    tuple_output = []
    annotation_examples = Annotation()
    margin=0.0001
    while (end < len(turn_history) - 1):
        if (turn_history[end] != turn_history[start]):
            # Record the changes
            tuple_output.append((current_speaker, float(start), float(end-1)))
            annotation_examples[Segment(float(start)+margin, float(end-1)+ margin * 10)] = current_speaker
            # Reset
            start = end 
            current_speaker = turn_history[start]
        end += 1
    if (turn_history[start] == turn_history[end]):
        tuple_output.append((current_speaker, float(start), float(end)))
        annotation_examples[Segment(float(start) + margin, float(end) + margin * 10)] = current_speaker

    else:
        tuple_output.append((current_speaker, float(start), float(end-1)))
        tuple_output.append((turn_history[end], float(end), float(end)))

        annotation_examples[Segment(float(start) + margin, float(end-1) + margin * 10)] = current_speaker
        annotation_examples[Segment(float(end) + margin, float(end) + margin * 10)] = turn_history[end]

    return tuple_output, annotation_examples


class DERMetrics(Metrics):
    """
        Diarization Error Rate (DER) metric variations. 
        Supporting
            1. DER: Turn-based Diarization Error Rate
            2. WDER: Word-level Diarization Error Rate
            3. JER: Jaccard Error Rate 
    """
    def __call__(self, candidates, references, ids=None, lengths=None, *, dataset_name: str | None = None, model_name: str | None = None):
        overall = self.get_score(candidates, references, ids, lengths)

        if dataset_name and model_name:
            # DER record scores are stored under 'der_per_row'
            scores = self.record_level_scores.get("der_per_row", [])
            self._write_record_log(references, candidates, scores, dataset_name, model_name)
            # Append overall metric at the end
            self._append_final_score(overall, dataset_name, model_name)
        return overall

    def _append_final_score(self, overall, dataset_name, model_name):
        import json, re
        from pathlib import Path
        def _slug(s):
            return re.sub(r"[^A-Za-z0-9_]+", "_", s)
        log_path = Path(".") / f"{_slug(dataset_name)}_{_slug(self.name)}_{_slug(model_name)}.log"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"final_score": overall}, ensure_ascii=False) + "\n")

    def _write_record_log(self, refs, cands, scores, dataset_name, model_name):
        import json, re
        from pathlib import Path
        from itertools import zip_longest
        def _slug(s):
            return re.sub(r"[^A-Za-z0-9_]+", "_", s)
        log_path = Path(".") / f"{_slug(dataset_name)}_{_slug(self.name)}_{_slug(model_name)}.log"
        with open(log_path, "w", encoding="utf-8") as f:
            for ref, cand, sc in zip_longest(refs, cands, scores, fillvalue=None):
                entry = {"reference": ref, "candidate": cand}
                if sc is not None:
                    entry["score"] = sc
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        # Always write to shared run.log
        self._write_to_run_json(refs, cands, scores, dataset_name, model_name)
        logger.info(f"Wrote record-level log to {log_path}")
        # Write to shared run.json
        #self._write_to_run_json(refs, cands, scores, dataset_name, model_name)

    def __init__(self, language="en"):
        super().__init__()
        self.name = "diarization_error_rate"
        self.display_name = "Diarization Error Rate"
        self.description = "The proportion of incorrectly predicted speakers when compared to the reference speakers. They can be turn_based or dataset_based. The dataset is considered as one big conversation."
        self.language = language

        self.der_metric = DiarizationErrorRate(collar=0.0)
        self.jaccard_metric = JaccardErrorRate()


    def _write_to_run_json(self, refs, cands, scores, dataset_name, model_name):
        """Write each sample's prediction to a shared run.log file."""
        import json
        from pathlib import Path
        from itertools import zip_longest
        
        run_path = Path(".") / "run.log"
        with open(run_path, "a", encoding="utf-8") as f:
            for ref, cand, sc in zip_longest(refs, cands, scores, fillvalue=None):
                entry = {
                    "dataset": dataset_name,
                    "metric": self.name,
                    "model": model_name,
                    "reference": ref,
                    "candidate": cand,
                }
                if sc is not None:
                    entry["score"] = sc
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    
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
            Dict with DER metrics by overall, conversation, and length buckets
        """
        scores = self.compute_record_level_scores(candidates, references)
        result={}
        # Compute the overall turn-based DER
        turn_based_der_incorrect = sum(scores['turn_based_der_incorrect'])
        turn_based_der_total = sum(scores['turn_based_der_total'])

        # Overall DER
        overall_der = turn_based_der_incorrect / turn_based_der_total if turn_based_der_total > 0 else 0
        overall_der = min(overall_der, 1.0)

        avg_sample_turn_based_der = sum(scores['der_per_row'])/ len(scores['der_per_row']) if scores['der_per_row'] else 0
        avg_sample_turn_based_der = min(avg_sample_turn_based_der, 1.0)

        # Compute overall WDER
        wder_incorrect = sum(scores['wder_incorrect'])
        wder_total = sum(scores['wder_total'])

        # Overall WDER
        overall_wder = wder_incorrect / wder_total if turn_based_der_total > 0 else 0
        overall_wder = min(overall_wder, 1.0)

        avg_sample_wder = sum(scores['wder_per_row'])/ len(scores['wder_per_row']) if scores['wder_per_row'] else 0
        avg_sample_wder = min(avg_sample_wder, 1.0)

        avg_sample_jer = sum(scores['jer_per_row'])/ len(scores['jer_per_row']) if scores['jer_per_row'] else 0
        avg_sample_jer = min(avg_sample_jer, 1.0)

        result ={
            "average_turn_based_der": avg_sample_turn_based_der,
            "overall_turn_based_der": overall_der,
            "average_sample_wder": avg_sample_wder,
            "overall_wder": overall_wder,
            "average_sample_jer": avg_sample_jer,
        }

        # # Store the scores for later record level reporting
        # # Important to use setdefault which is a no-op if the value already exists
        # # As users can evaluate multiple models and call compute_record_level_scores multiple times
        self.record_level_scores.setdefault("der_per_row", scores["der_per_row"])
        self.record_level_scores.setdefault("turn_based_der_incorrect", scores["turn_based_der_incorrect"])
        self.record_level_scores.setdefault("turn_based_der_total", scores["turn_based_der_total"])

        self.record_level_scores.setdefault("wder_per_row", scores["wder_per_row"])
        self.record_level_scores.setdefault("wder_incorrect", scores["wder_incorrect"])
        self.record_level_scores.setdefault("wder_total", scores["wder_total"])

        return result

    def compute_record_level_scores(self, candidates: list, references: list):
        """Compute the scores that should be saved in the record level file.

        Args:
            candidates: Generated text from the model
            references: Reference text from the dataset

        Returns:
            Scores for each record. The keys should be the column names that will be saved in the record level file.
        """
        from tqdm import tqdm
        der_scores, wder_scores, jer_scores = [], [], []
        references_clean, candidates_clean = [], []
        turn_based_der_incorrect, turn_based_der_total = [], []
        wder_incorrect, wder_total = [], []
        jer_incorrect, jer_total = [], []

        for i, (reference, candidate) in enumerate(tqdm(zip(references, candidates), desc="diarization_error_rate", total=len(references))):
            lang_code = getattr(self, 'language', 'en')
            ref_by_lines = reference.split('\n')
            cand_by_lines = candidate.split('\n')
           
            # Extract speakers by turns from references and hypotheses        
            num_min_iterations = min(len(ref_by_lines), len(cand_by_lines))
            ref_speakers, cand_speakers = [], []
            cleaned_ref_transcripts, cleaned_cand_transcripts=[],[]
            word_level_ref_speakers, word_level_cand_speakers=[],[]

            for i in range (num_min_iterations):
                cur_ref_speaker, cur_word_level_ref_speaker, normalized_ref_transcript = prepare_speaker_info(ref_by_lines[i], lang_code)
                cur_cand_speaker, cur_word_level_cand_speaker, normalized_cand_transcript = prepare_speaker_info(cand_by_lines[i], lang_code)

                ref_speakers.append(cur_ref_speaker)
                word_level_ref_speakers.extend(cur_word_level_ref_speaker)
                cleaned_ref_transcripts.append(normalized_ref_transcript)

                cand_speakers.append(cur_cand_speaker)
                word_level_cand_speakers.extend(cur_word_level_cand_speaker)
                cleaned_cand_transcripts.append(normalized_cand_transcript)

            if (len(ref_by_lines) > num_min_iterations):
                for j in range (len(ref_by_lines) - num_min_iterations):
                    cur_ref_speaker, cur_word_level_ref_speaker, normalized_ref_transcript = prepare_speaker_info(ref_by_lines[num_min_iterations + j], lang_code)
                    ref_speakers.append(cur_ref_speaker)
                    word_level_ref_speakers.extend(cur_word_level_ref_speaker)
                    cleaned_ref_transcripts.append(normalized_ref_transcript)

            else:
                for j in range (len(cand_by_lines) - num_min_iterations):
                    cur_cand_speaker, cur_word_level_cand_speaker, normalized_cand_transcript = prepare_speaker_info(cand_by_lines[num_min_iterations + j], lang_code)
                    cand_speakers.append(cur_ref_speaker)
                    word_level_cand_speakers.extend(cur_word_level_cand_speaker)
                    cleaned_cand_transcripts.append(normalized_cand_transcript)

            assert len(ref_by_lines) == len(ref_speakers)
            assert len (cand_by_lines) == len(cand_speakers)

            
            # Convert list of speakers to the correct format to compute metrics
            ref_speaker_tuples, ref_annotations = convert_speaker_list_to_diarization_tuples(ref_speakers)
            cand_speaker_tuples, cand_annotations = convert_speaker_list_to_diarization_tuples(cand_speakers)

            word_level_ref_speaker_tuples, word_level_ref_annotations = convert_speaker_list_to_diarization_tuples(word_level_ref_speakers)
            word_level_cand_speaker_tuples, word_level_cand_annotations = convert_speaker_list_to_diarization_tuples(word_level_cand_speakers)

            # Components: 'confusion', 'missed detection', 'total', 'correct', 'false alarm', 'diarization error rate'
            der_components = self.der_metric(ref_annotations, cand_annotations, detailed=True) 
            der = min(der_components['diarization error rate'], 1.0) # cap der to 1.0

            der_turn_total, der_turn_confusion, der_turn_missed_detection, der_turn_correct, der_turn_false_alarm = der_components['total'], der_components['confusion'], der_components['missed detection'], der_components['total'], der_components['false alarm']
            
            # Components: 'speaker error', 'speaker count', 'jaccard error rate'
            jer_components = self.jaccard_metric(ref_annotations, cand_annotations, detailed=True)
            jer = min(jer_components['jaccard error rate'],1.0)
            jer_speaker_error, jer_speaker_count = jer_components['speaker error'], jer_components['speaker count']

            wder_components = self.der_metric(word_level_ref_annotations, word_level_cand_annotations, detailed=True)
            wder = min(wder_components['diarization error rate'], 1.0)

            wder_sample_total, wder_confusion, wder_missed_detection, wder_correct, wder_false_alarm = wder_components['total'], wder_components['confusion'], wder_components['missed detection'], wder_components['total'], wder_components['false alarm']

            
            # Aggregating per-row errors/ totals for overall dataset metric calculation
            turn_based_der_incorrect.append(der_turn_confusion + der_turn_missed_detection + der_turn_false_alarm)
            turn_based_der_total.append(der_turn_total)

            wder_incorrect.append(wder_confusion + wder_missed_detection + wder_false_alarm)
            wder_total.append(wder_sample_total)

            candidates_clean.append(cleaned_cand_transcripts)
            references_clean.append(cleaned_ref_transcripts)

            der_scores.append(der)
            wder_scores.append(wder)
            jer_scores.append(jer)


        results = {
            "der_per_row": der_scores,
            "wder_per_row": wder_scores,
            "jer_per_row": jer_scores,

            "candidates_clean": candidates_clean,
            "references_clean": references_clean,
            "turn_based_der_incorrect": turn_based_der_incorrect,
            "turn_based_der_total": turn_based_der_total,
            "wder_incorrect": wder_incorrect,
            "wder_total": wder_total
        }

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
            "der": MetricMetadata(
                name="der",
                display_name=f"{constants.INVERTED_METRIC_INDICATOR} Diarization Error Rate",
                description=self.description,
                higher_is_better=False,
            )
        }
        for attribute in ("accent", "gender"):
            current_attr = set(self.record_level_scores.get(attribute, []))
            for attr_value in current_attr:
                if attr_value is not None:
                    metadata[f"der_{attribute}_{attr_value}"] = MetricMetadata(
                        name=f"der_{attribute}_{attr_value}",
                        display_name=f"{constants.INVERTED_METRIC_INDICATOR} Diarization Error Rate {attribute.title()} ({attr_value})",
                        description=self.description,
                        higher_is_better=False,
                    )
        return metadata
