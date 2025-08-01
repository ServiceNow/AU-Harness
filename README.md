<div align="center" style="margin-bottom: 1em;">

*Your ultimate audio evaluation framework*

</div>

## Overview

LALMEval is a powerful open-source framework for evaluating audio-based language models. Built for researchers and developers, LALMEval provides a comprehensive suite of tools to benchmark and compare the performance of various audio processing models across a wide range of tasks.

## Why LALMEval?

1. 🚀 **Blazing Fast**:
   - Any amount of models are run against any amount of datasets and metrics, each with their own separate Engines, allowing for parallelization of the entire evaluation pipeline
   - Model inference and evaluation is batched, with the only bottleneck being user-set batch size
   - Dataset Sharing is implemented for linearly scalable inference throughput

2. 🔧 **Immensely Customizable**:
   - Dataset and Samples can be customized by accents, language, length, and more
   - Models can be customized by temperature, request parameters, and batch size
   - Score reporting can be customized through the aggregation parameter

3. 📦 **Super Modular**:
   - Streamlined evaluation processes allow for better understanding of the codebase
   - Modularized functions allow for easy extension and customization

4. 🎯 **Wide Task Coverage**:
   - We support 18 unique tasks over 5 different categories
   - Over 50 unique datasets, with 420+ unique subsets
   - 9 different metrics for broader evaluation coverage

## Usage

LALMEval requires setting up a configuration file (`config.yaml`) to define your evaluation parameters. This file controls which models, datasets, and metrics are used in your evaluation.

To get started with LALMEval:

1. Clone this repository
2. Populate your `config.yaml` file based on the example provided in `sample_config.yaml` - the given 'config.yaml' already has the mandatory fields
3. Run the end-to-end evaluation with `bash evaluate.sh`

### Configuration Options

The `config.yaml` file supports the following customization options:

#### Dataset and Metrics
```yaml
dataset_metric:
  - "(wavcaps_qa_test, word_error_rate)"  # Evaluate by dataset-metric pair
  - "(emotion_recognition, llm_judge_binary)"  # Evaluate by task_type
  - "(spoken_language_reasoning, instruction_following)"  # Evaluate by task category
```

#### Sampling and Filtering
```yaml
num_samples: 100  # Number of samples to evaluate (remove for all)
judge_concurrency: 8  # Number of concurrent judge calls
judge_model: "gpt-4o-mini"  # Model used for judging
user_prompt_add_ons: ["asr_clean_output"]  # Additional prompting instructions
length_filter: [1.0, 3.0]  # Filter audio samples by length (seconds)
accented: false  # Filter for accented/non-accented speech
language: "english"  # Filter by language
```

#### Result Aggregation
```yaml
aggregate:
  - ["emotion performance", [["meld_emotion_test", "llm_judge_binary"]]]
  - ["gender performance", [["voxceleb_gender_test", "llm_judge_binary"], ["iemocap_gender_recognition", "llm_judge_binary"]]]
  - ["all binary judgements", [["all", "llm_judge_binary"]]]
```

#### Temperature Control
```yaml
temperature_overrides:
  # Model and task override
  - model: "gpt-4o-mini-audio-preview"
    task: "emotion_recognition"
    temperature: 0.9
  # Model only override
  - model: "gpt-4o-mini-audio-preview"
    temperature: 0.3
```

#### Model Configuration
```yaml
models:
  - info:
      name: "gpt-4o-mini-audio-preview"
      inference_type: "openai"  # openai, vllm, or audio transcription
      url: "${ENDPOINT_URL}"
      delay: 100
      retry_attempts: 8
      timeout: 30
      model: "gpt-4o-mini-audio-preview"
      auth_token: "${AUTH_TOKEN}"
      api_version: "${API_VERSION}"
      batch_size: 100
      chunk_size: 45  # Max audio length in seconds
```

### Adding Datasets

LALMEval supports adding custom datasets through runspec JSON files. These files define the dataset properties and how they should be processed.

#### Creating a Runspec File

Create a JSON file in the `runspecs` directory under the appropriate task category. Each dataset should be defined with the following properties:

```json
{
  "dataset_name": {
    "hf_repo": "HuggingFace/repository-name",
    "subset": "subset-name",  // Optional
    "language": "language-code",
    "preprocessor": "PreprocessorClass",
    "postprocessor": "PostprocessorClass",
    "accented": true|false  // Optional
  }
}
```

#### Fields

**Required:**
- `hf_repo`: The HuggingFace repository containing the dataset
- `language`: The language code (e.g., "en" for English)
- `preprocessor`: The preprocessor class to use for preparing the data
- `postprocessor`: The postprocessor class to use for formatting results

**Optional:**
- `subset`: The specific subset within the repository to use
- `accented`: Boolean indicating whether the dataset contains accented speech

#### Example

Here's an example runspec for spoken dialogue summarization datasets:

```json
{
  "mnsc_sds_part3_test": {
    "hf_repo": "AudioLLMs/Multitask-National-Speech-Corpus-v1-extend",
    "subset": "SDS-PART3-Test",
    "language": "en",
    "preprocessor": "GeneralPreprocessor",
    "postprocessor": "GeneralPostprocessor",
    "accented": true
  }
}
```

#### Using Your Dataset

After creating the runspec file, you can reference your dataset in the `config.yaml` file:

```yaml
dataset_metric:
  - "(your_dataset_name, word_error_rate)"  # Or any other appropriate metric
```

## License

LALMEval is licensed under the ___ License. See [LICENSE](LICENSE) for more information.

## Vulnerability Reporting

For security issues, please email ___