# 📏 Metrics Overview

This document provides a summary and detailed explanation of all evaluation metrics used in the framework.

---

## 📊 Metric Overview Table

| Metric Name                | Description                                      | Reported metric values                                |
|---------------------------|--------------------------------------------------|-----------------------------------------------|
| `llm_judge_binary` (&uarr;)        | Binary LLM-based correctness judgment            | llm_judge_binary                 |
| `llm_judge_detailed` (&uarr;)      | Detailed scoring across multiple dimensions      | llm_judge_detailed            |
| `llm_judge_big_bench_audio` (&uarr;) | LLM-based BigBench audio evaluations           | llm_judge_big_bench_audio                                      |
| `bleu` (&uarr;)                    | N-gram overlap score                             | bleu                    |
| `bertscore` (&uarr;)               | Semantic similarity using BERT embeddings        | bertscore                       |
| `meteor` (&uarr;)                  | Alignment-based score with synonym handling      | meteor                    |
| `word_error_rate_metrics` (&darr;) | Measures ASR errors via insertions, deletions    | average_sample_wer, overall_wer                            |
| `diarization_metrics` (&darr;)     | LLM-Adaptive diarization-relevent metrics    | avg_sample_wder, overall_wder, avg_sample_cpwer, overall_cpwer, avg_speaker_count_absolute_error                           |
| `bfcl_match_score` (&uarr;)        | Structured logic form comparison                 | bfcl_match_score                          |
| `sql_score` (&uarr;)               | SQL correctness and execution match              | text2sql_score                                 |
| `instruction_following` (&uarr;)   | LLM-judged instruction following capability                 | instruction_following              |

---

## 📋 Metric Details

### `llm_judge_binary`
- **Type**: Binary classification
- **Description**: Judges whether a model output is correct or not using an LLM.
- **Used In**: Emotion/gender/accent recognition, intent classification, QA
- **Scoring**: `1` for correct, `0` for incorrect. Final score is mean accuracy.

---

### `llm_judge_detailed`
- **Type**: Multi-dimensional judgment
- **Description**: Uses an LLM to assess output quality based on attributes like fluency, relevance, and completeness.
- **Used In**: Summarization, scene understanding
- **Scoring**: Aggregated scores per attribute or overall composite rating.

---

### `llm_judge_big_bench_audio`
- **Type**: LLM-based QA judgment
- **Description**: Evaluates performance on BigBench-like audio QA tasks.
- **Used In**: Spoken query QA (sqqa)
- **Scoring**: Based on LLM-determined correctness or ranking.

---

### `bleu`
- **Type**: N-gram precision metric
- **Description**: Measures how many n-grams in the prediction match the reference.
- **Used In**: Translation, summarization
- **Scoring**: Score between `0` and `1`, higher is better.

---

### `bertscore`
- **Type**: Semantic similarity
- **Description**: Uses contextual BERT embeddings to match tokens semantically.
- **Used In**: Translation, generation
- **Scoring**: Outputs precision, recall, and F1 (between `0` and `1`).

---

### `meteor`
- **Type**: Alignment metric
- **Description**: Improves on BLEU by considering synonyms, stemming, and paraphrase.
- **Used In**: Translation, summarization
- **Scoring**: Score between `0` and `1`.

---

### `word_error_rate_metrics`
- **Type**: Speech recognition metric
- **Description**: Measures the correctness of the generated hypothesis vs reference transcript
- **Reported Value**: 
    - `average_sample_wer`: Averaging WER of each sample across the evaluated dataset samples
    - `overall_wer`: $$(\sum(D) + \sum(I) + \sum(S)) / total\_word\_counts$$
- **Used In**: asr, long_form_asr, code_switching_asr

---

### `diarization_error_rate_metrics`
- **Type**: LLM-Adaptive Diariziation-relevant metrics
- **Description**: Measure the diarization error rate for the generated LLM-generated hypothesis
- **Reported Value**:
    - `average_sample_wder`: Averaging WDER of each sample across the evaluated dataset samples
    - `overall_wder`: Overall Errors / Overall Total Counts
    - `avg_sample_cpwer`: Averaging cpWER of each sample across the evaluated dataset samples
    - `overall_cpwer`: Overall Errors / Overall Total Counts
    - `avg_speaker_count_absolute_error`:  Accuracy of the predicted number of speakers
- **Used In**: speaker_diarization, emotion_diarization

---

### `bfcl_match_score`
- **Type**: Function calling metric
- **Description**: Evaluate the function calling capabilities. 
- **Used In**: Audio Function calling 
- **Scoring**: Score between `0` and `1`.

---

### `sql_score`
- **Type**: Coding correctness metric
- **Description**: Evaluate the correctness of the generated SQL. 
- **Used In**: Audio SQL coding 
- **Scoring**: Score between `0` and `1`.

---

### `instruction_following`
- **Type**: Instruction following evaluation metric
- **Description**: Measuring the instruction following capabilities. 
- **Used In**: Audio Instruction Following
- **Scoring**: Score between `0` and `1`.
