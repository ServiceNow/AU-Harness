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
2. Setup your environment:
```bash
python -m venv myEnv
source myEnv/bin/activate
pip install -r requirements.txt
```
3. Populate your `config.yaml` file based on the example provided in `sample_config.yaml` - the given 'config.yaml' already has the mandatory fields
4. Run the end-to-end evaluation:
```bash
bash evaluate.sh
```

To deploy your own models, look at /models/inference_boilerplate/ for more instructions.

### Configuration Options

The `config.yaml` file supports the following customization options:

#### Dataset and Metrics
```yaml
dataset_metric:
  - ["librispeech_test_other", "word_error_rate"] #evaluate by dataset
  - ["emotion_recognition", "llm_judge_binary"] # evaluate by task type
  - ["spoken_language_understanding", "llm_judge_binary"] # evaluate by task category
```

#### Sampling and Filtering
```yaml
filters:
  num_samples: 300 # optional - number of samples to run(remove for all)
  user_prompt_add_ons: ["asr_clean_output"] # optional - additional prompting in text instructions for each sample
  length_filter: [1.0, 30.0] # optional - filters for only audio samples in this length(seconds)
  accented: false # optional - filters for only audio samples in this length(seconds)
  language: "en" # optional - filters for only audio samples in this language - use language code
  system_prompts: ["audio_expert"] # optional - system prompts for each sample
```

#### Result Aggregation
```yaml
# Optional - allows for custom score aggregation at the end
# Follow the format of [x, [y1, y2]] where x is a valid metric, and each y is a valid, matching dataset 
aggregate:
  - ["llm_judge_binary", ["emotion_recognition"]]
  - ["llm_judge_detailed", ["alpaca_audio_test", "openhermes_instruction_test"]]
```

#### Temperature Control
```yaml
temperature_overrides:
  # Model and task override
  - model: "gpt-4o-mini-audio-preview"
    task: "emotion_recognition"
    temperature: 0.5
  # Model only override
  - model: "gpt-4o-mini-audio-preview"
    temperature: 0.7
  # Task only override
  - task: "accent_recognition"
    temperature: 0.5
```

#### Model Configuration
```yaml
models:
  - info:
      name: "gpt-4o-mini-audio-preview" # Mandatory
      inference_type: "openai"  # openai, vllm, or audio transcription
      url: ${ENDPOINT_URL} # Mandatory
      delay: 100 # Optional
      retry_attempts: 8 # Optional
      timeout: 30 # Optional
      model: "gpt-4o-mini-audio-preview" # Mandatory
      auth_token: ${AUTH_TOKEN} # Mandatory
      api_version: ${API_VERSION} # Mandatory
      batch_size: 200 # Mandatory
      chunk_size: 30  # Optional - Max audio length in seconds
```

#### Judge Configuration
```yaml
judge_properties:
  judge_concurrency: 300 # optional - default is 1
  judge_model: "gpt-4o-mini" # mandatory
  judge_type: "openai" # mandatory (vllm or openai)
  judge_api_version: ${API_VERSION} # optional(needed for openai)
  judge_api_endpoint: ${API_ENDPOINT} # mandatory
  judge_api_key: ${API_KEY} # mandatory
  judge_temperature: 0.1 # optional
  judge_prompt_model_override: "Qwen3-32b" # optional
```

### Adding Datasets

LALMEval supports adding custom datasets through runspec JSON files. These files define the dataset properties and how they should be processed.

#### Creating a Runspec File

Create a JSON file in the `runspecs` directory under the appropriate task category and type. Each dataset should be defined with the following properties:

```json
{
  "dataset_name": {
        "hf_repo": "huggingface_repo", // Mandatory
        "subset": "subset", // Optional (recommended)
        "split": "usa", // Optional (recommended)
        "language": "english", // Mandatory
        "modality": "audio", // Optional 
        "audio_column": "audio", // Optional
        "target_column": "reference", // Optional (recommended)
        "additional_instruction_column": null, // Optional 
        "textual_input_column": "prompt", ; // Optional
        "preprocessor": "PreprocessorClass", // Mandatory
        "postprocessor": "PostprocessorClass" //Mandatory
    },
}
```

#### Example

Here's an example runspec for spoken dialogue summarization datasets:

```json
"sd-qa_ind_s_audio": {
        "hf_repo": "hlt-lab/voicebench",
        "subset": "sd-qa",
        "split": "ind_s",
        "language": "english",
        "modality": "audio",
        "audio_column": "audio",
        "target_column": "reference",
        "additional_instruction_column": null,
        "textual_input_column": "prompt",
        "preprocessor": "GeneralPreprocessor",
        "postprocessor": "GeneralPostprocessor"
    },
```

#### Using Your Dataset

After creating the runspec file, you can reference your dataset in the `config.yaml` file:

```yaml
dataset_metric:
  - "[your_dataset_name, metric_name]" 
```

#### CallHome

To run the CallHome dataset for ASR and Speaker Diaraization, follow these steps:

##### Download the datasets

Go to https://talkbank.org/ca/access/CallHome/ and click on the language corpus you want

Click "Download Transcripts" to get a zip file of each transcript. Copy all the .cha files to the transcripts folder in /private_datasets/{specific_langauge_folder}/transcripts

Go to the Media folder (https://media.talkbank.org/ca/CallHome) and select the specific language corpus, then click on the 0wav/ folder for the language corpus. 
Open the dev console and paste this code

```javascript
document.querySelectorAll('a[href*="?f=save"]').forEach((link, i) => {
    setTimeout(() => link.click(), i * 1000);
});
```

Paste these wav files into the audio folder of the specific language path in private_datasets

Run the specific dataset or all datasets across word_error_rate(utterance by utterance), llm_judge_binary(holistic view of 30 second turn by turn transcription), or speaker diarization

```yaml
dataset_metric:
  - ["callhome_eng", "word_error_rate"] # OR
  - ["callhome, "speaker_diarization"]
```
## Kit Structure

### General Evaluation Flow

![LALMEval Architecture](EvaluationStructure.png)

The evaluation flow in LALMEval follows a highly concurrent architecture:

1. **Configuration & Initialization**: The system parses `config.yaml` to load models, datasets, metrics, and other evaluation parameters.

2. **Engine Assembly**: For each dataset-metric pair, an Engine is created containing:
   - A dataset
   - A preprocesser
   - The specified metric
   - An appropriate postprocessor
   - References to all specified models

3. **Concurrent Execution**: 
   - All Engines run simultaneously
   - Within each Engine, model inference occurs concurrently across all models
   - After inference completes, the postprocessor transforms model outputs
   - Evaluation is performed concurrently, with record-level scores logged throughout

4. **Results Aggregation**: The main process awaits completion of all Engines before compiling and reporting final performance metrics.

This architecture enables efficient scaling with multiple models and datasets while maintaining organized evaluation workflows.

### Runspec Structure

LALMEval organizes datasets and tasks hierarchically in the `runspecs` directory:

```
runspecs/
├── spoken_language_understanding/  
├── speech_recognition/             
├── paralinguistics/                
├── spoken_language_reasoning/      
└── callhome/                       
```

Each task category directory contains JSON files that define specific tasks or datasets within that category. For example:

- `paralinguistics/` includes `emotion_recognition.json`, `accent_recognition`, etc.

When configuring evaluations in `config.yaml`, you can reference:
- Specific datasets by name (e.g., `"librispeech_test_other"`)
- Entire task types by referencing the JSON file stem (e.g., `"accent_recognition"`)
- Whole task categories by directory name (e.g., `"spoken_language_understanding"`)

This flexible structure allows for targeted or broad evaluations depending on your needs.

## License

LALMEval is licensed under the Apache 2.0 License.
