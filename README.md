<div align="center" style="margin-bottom: 1em;">

**HEAR-Kit: Holistic Evaluation of AudioLLM Responses**

*Comprehensive • Fast • Reproducible*

<img src="assets/images/ver-1-bg.png" alt="HEAR-Kit Brand Logo" width="200">

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Version](https://img.shields.io/badge/version-0.1.0-orange.svg)](https://github.com/ServiceNow/HEAR-Kit)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/ServiceNow/HEAR-Kit/pulls)

**🎯 50+ Datasets • 🚀 380+ Subsets • 📊 9 Metrics • 🔊 21 Audio Tasks**

</div>

## 📋 Overview

HEAR-Kit is a standardized, efficient and highly customizable open-source framework for evaluating audio-based language models on Audio-to-Text tasks. Built for researchers and developers, HEAR-Kit provides a comprehensive suite of tools to benchmark and compare the performance of various audio processing models across a wide range of tasks.


## ❓ Why HEAR-Kit?

1. 🚀 **Blazing Fast**:
   - Any amount of models are run against any amount of datasets and metrics, each with their own separate Engines, allowing for parallelization of the entire evaluation pipeline
   - Model inference and evaluation is batched, with the only bottleneck being user-set batch size
   - Dataset Sharing is implemented for linearly scalable inference throughput

<p align='center'>
  <img src="assets/images/eval_kit_comparison.png" alt="Evaluation Kit Comparison" width="80%", height="auto"/>
</p>

2. 🔧 **Immensely Customizable**:
   - Dataset and Samples can be customized by accents, language, length, and more
   - Models can be customized by temperature, request parameters, and batch size
   - Score reporting can be customized through the aggregation parameter
```yaml
dataset_path: fixie-ai/covost2
split: test
preprocessor: Covost2Preprocessor
postprocessor: Covost2Postprocessor
audio_column: audio
target_column: translation
instruction_column: null
target_language: en
user_prompt: |
  Please translate the given speech from {{source_language_name}} to {{target_language_name}}. Return ONLY the translated text without any other prefix text such as the given speech is. Be brief. Do not provide explanations.
long_audio_processing_logic: chunk

generation_kwargs:
  temperature: 0.2
  do_sample: false
  max_gen_tokens: 64

metrics:
  - metric: bleu
  - metric: meteor
  ```

3. 📦 **Super Modular**:
   - Streamlined evaluation processes allow for better understanding of the codebase
   - Modularized functions allow for easy extension and customization

4. 🎯 **Wide Task Coverage**:
   - We support 21 unique tasks over 6 different categories
   - Over 50 unique datasets, with 380+ unique subsets
   - 9 different metrics for broader evaluation coverage


## 📊 Task Taxonomy & Structure

<div align="center" style="margin: 30px 0;">
  <img src="assets/images/taxonomy.png" alt="HEAR-Kit Task Taxonomy" style="width: 60%; max-width: 600px; border-radius: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.15);">
</div>

### 📁 Task Organization

<details>
<summary><b>🗣️ <a href="./tasks/speech_recognition/README.md"> Speech Recognition </a></b> <i>(3 tasks)</i></summary>

- [**asr**](./tasks/speech_recognition/asr/) - Automatic speech recognition
  - *Datasets: librispeech, voxpopuli, common voice, and more*
- [**code_switching_asr**](./tasks/speech_recognition/code_switching_asr/) - Transcribe utterances with mixed-language speech.
- [**long_form_asr**](./tasks/speech_recognition/long_form_asr/) - Transcribe extended audio content

</details>

<details>
<summary><b>🎭 <a href="./tasks/paralinguistics/README.md"> Paralinguistics </a></b> <i>(5 tasks)</i></summary>

- [**emotion_recognition**](./tasks/paralinguistics/emotion_recognition/) - Detect emotional states from speech
- [**accent_recognition**](./tasks/paralinguistics/accent_recognition/) - Identify speaker accents and dialects
- [**gender_recognition**](./tasks/paralinguistics/gender_recognition/) - Classify speaker gender from voice
- [**speaker_recognition**](./tasks/paralinguistics/speaker_recognition/) - Identify speaker(s) present in the audio.
- [**speaker_diarization**](./tasks/paralinguistics/speaker_diarization/) - Segment speech into audio segments attributed to different speakers

</details>

<details>
<summary><b>🔊 <a href="./tasks/audio_understanding/README.md"> Audio Understanding </a></b> <i>(2 tasks)</i></summary>

- [**music_understanding**](./tasks/audio_understanding/music_understanding/) - Analyze and understand musical content
- [**scene_understanding**](./tasks/audio_understanding/scene_understanding/) - Identify and classify audio scenes based on the ambient sound information.

</details>

<details>
<summary><b>🧠 <a href="./tasks/spoken_language_understanding/README.md"> Spoken Language Understanding </a> </b> <i>(5 tasks)</i></summary>

- [**intent_classification**](./tasks/spoken_language_understanding/intent_classification/) - Classify user intents from spoken inputs
- [**speech_qa**](./tasks/spoken_language_understanding/speech_qa/) - Answer questions based on spoken content
- [**sqqa**](./tasks/spoken_language_understanding/sqqa/) - Spoken query question-answering with context
- [**spoken_dialogue_summarization**](./tasks/spoken_language_understanding/spoken_dialogue_summarization/) - Summarize spoken conversations
- [**translation**](./tasks/spoken_language_understanding/translation/) - Translate given speech into the target language. 

</details>

<details>
<summary><b>🧩 <a href="./tasks/spoken_language_reasoning/README.md"> Spoken Language Reasoning </a></b> <i>(4 tasks)</i></summary>

- [**ifeval**](./tasks/spoken_language_reasoning/ifeval/) - Speech Instruction-following capability evaluation
- [**bfcl**](./tasks/spoken_language_reasoning/bfcl) - Speech Function Calling capability evaluation
- [**mtbench**](./tasks/spoken_language_reasoning/mtbench/) - Complex multi-turn Instruction-following capability evaluation
- [**speech_to_sql**](./tasks/spoken_language_reasoning/mtbench/) - Speech-to-Coding capability

</details>

<details>
<summary><b>🔐 <a href="./tasks/safety_and_security/README.md"> Safety and Security </a></b> <i>(2 tasks)</i></summary>

- [**safety**](./tasks/safety_and_security/safety/) - Evaluate model safety and robustness
- [**spooling**](./tasks/safety_and_security/spoofing/) - Detect synthetic or manipulated audio

</details>

## 🏗️ Architecture

### General Evaluation Flow

<p align='center'>
  <img src="assets/images/overview.png" alt="Taxonomy Figure" width="80%", height="auto"/>
</p>
The evaluation flow in HEAR-Kit follows a highly concurrent architecture:

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

## 🚀 Quick Start

Get up and running in under a minute:

```bash
# Clone and install
git clone https://github.com/ServiceNow/LALMEval.git
cd LALMEval
pip install -r requirements.txt

# Run your first evaluation
cp sample_config.yaml config.yaml
bash evaluate.sh
```

Results will be generated in `logs/` with detailed metrics and analysis.

## 💻 Usage

HEAR-Kit requires setting up a running configuration file (`config.yaml`) to define your evaluation parameters. This file controls which models, datasets, and metrics are used in your evaluation.

To get started with HEAR-Kit:

1. Clone this repository
2. Setup your environment:
```bash
python -m venv myEnv
source myEnv/bin/activate
pip install -r requirements.txt
```
3. Populate your `config.yaml` file based on the example provided in `sample_config.yaml` and instructions below - the given 'config.yaml' already has the mandatory fields
4. Run the end-to-end evaluation:
```bash
bash evaluate.sh
```

### 🧩 Running Configuration Options

The `config.yaml` file supports the following customization options. Sample running configurations are available for reference at [sample_config.yaml](./sample_config.yaml).

#### Dataset and Metrics
```yaml
dataset_metric:
  - ["librispeech_test_other", "word_error_rate"] #evaluate by dataset
  - ["emotion_recognition", "llm_judge_binary"] # evaluate by task type
  - ["spoken_language_understanding", "llm_judge_binary"] # evaluate by task category
```

#### Sampling and Filtering
```yaml
filter:
  num_samples: 300 # optional - number of samples to run(remove for all)
  length_filter: [1.0, 30.0] # optional - filters for only audio samples in this length(seconds)
```

#### Result Aggregation
```yaml
# Optional - allows for custom score aggregation at the end. Currently only simple average is supported
# Follow the format of [x, [y1, y2]] where x is a valid metric, and each y is a valid task or a group (of tasks)
aggregate:
  - ["llm_judge_binary", ["emotion_recognition"]]
  - ["llm_judge_detailed", ["alpaca_audio_test", "openhermes_instruction_test"]]
  - ["word_error_rate", ["librispeech"]]
```

#### Generation parameters override
```yaml
# Generation parameters are generally defined for each task in their task configs
# This can be overriden for specific models and tasks using the following format.
generation_params_override:
  # Task override - Apply for this task for all models
  - task: <TASK1>
    generation_params:
      temperature: <temperature>
      max_gen_tokens: <max_gen_tokens>
  # Model override - Apply for this model for all tasks
  - model: <MODEL1>
    generation_params:
      temperature: <temperature>
      max_gen_tokens: <max_gen_tokens>
  # Model and Task override - Apply for this model and task
  - model: <MODEL1>
    task: <TASK1>
    generation_params:
      temperature: <temperature>
      max_gen_tokens: <max_gen_tokens>
```

#### System and User prompt override
```yaml
# System prompts and user prompts (high level task instructions) can be overriden from the run config
prompt_overrides:
  # User prompt override mandatorily requires a task name because these are generally task specific
  user_prompt:
    - task: "task_name"
      model: "model_name" (optional)
      prompt: "prompt_text"
  # System prompt override mandatorily requires a model name because these are generally model specific
  system_prompt:
    - model: "model_name"
      task: "task_name" (optional)
      prompt: "prompt_text"
```

#### Model Configuration
```yaml
models:
  - name: "gpt-4o-mini-audio-preview-1" # Mandatory - must be unique
    inference_type: "openai"  # openai(openai), vllm(vllm), or audio transcription(transcription)
    url: ${ENDPOINT_URL} # Mandatory
    delay: 100 # Optional
    retry_attempts: 8 # Optional
    timeout: 30 # Optional
    model: "gpt-4o-mini-audio-preview" # Mandatory
    auth_token: ${AUTH_TOKEN} # Mandatory
    api_version: ${API_VERSION} # Mandatory
    batch_size: 350 # Mandatory
    chunk_size: 30  # Optional - Max audio length in seconds
    
  - name: "qwen_2.5_omni" # Mandatory
    inference_type: "vllm"  # openai, vllm, or audio transcription
    url: ${ENDPOINT_URL} # Mandatory
    delay: 100 # Optional
    retry_attempts: 8 # Optional
    timeout: 30 # Optional
    model: "qwen_2.5_omni" # Mandatory
    auth_token: ${AUTH_TOKEN} # Mandatory
    batch_size: 150 # Mandatory
    chunk_size: 30  # Optional - Max audio length in seconds
```

**Note**: Batch-size proportional dataset sharding is implemented when multiple endpoints of the same model are provided. Be sure to have unique 'name' attributes for each unique endpoint, as shown above

##### Inference Types

| Client           | Inference Type                       |
|------------------|--------------------------------------|
| "openai"         | AsyncAzureOpenAI (Chat Completions)  |
| "vllm"           | AsyncOpenAI (Chat Completions)       |
| "transcription"  | AsyncOpenAI (Transcriptions)         |

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

### 📝 Task Configuration Options
#### Adding Datasets

HEAR-Kit supports adding custom datasets through `task_config` YAML files. These files define the dataset properties and how they should be processed.

#### Creating a TaskConfig File

Create a YAML file in the `tasks` directory under the appropriate task category. Each dataset should be defined with the following properties, down to the most specific subset:

```yaml
task_name: "unique_task_name"
dataset_path: "huggingface_repo or local_dataset_path" // Mandatory
subset: "subset" // Optional (recommended)
split: "usa" // Mandatory
lange: "english" // Mandatory
modality: "audio", // Optional 
preprocessor: "PreprocessorClass", // Mandatory
postprocessor: "PostprocessorClass" //Mandatory
audio_column: "audio", // Optional
target_column: "reference", // Optional (recommended)
instruction_column: "instruction" // Optional (recommended)
long_audio_processing_logic: truncate //

generation_kwargs:  // Additional kwargs to constrain model decoding behaviors
  temperature: 0.0001 
  max_completion_tokens: 64

metrics:
  - metric: llm_judge_binary // Metric from the allowed pre-defined metrics
```

**Important Note:** It is HIGHLY Recommended to add a "user_prompt" field tailored specifically to the datasets you are running for the best results, especially for complex tasks.

#### Example

Here's an example task_config for intent classification (SLURP-Intent) datasets:

```yaml
task_name: SLURP-intent
dataset_path: DynamicSuperb/SuperbIC_SLURP-Intent
subset: default
split: test
language: english
preprocessor: GeneralPreprocessor
postprocessor: GeneralPostprocessor
audio_column: audio
target_column: label
instruction_column: instruction
long_audio_processing_logic: truncate

generation_kwargs:
  temperature: 0.0001
  do_sample: false
  max_completion_tokens: 64

metrics:
  - metric: llm_judge_binary
```

### ⚙️ Customizations
#### Using Your Dataset

After creating the run_config YAML file, you can reference your dataset in the `config.yaml` file:

```yaml
dataset_metric:
  - "[your_dataset_name, metric_name]" 
```

#### Using Your Own Model

To deploy your own models, look at [/models/inference_boilerplate/](./models/inference_boilerplate/) for more instructions.

### 📈 Analyzing Results

Once your run finishes, you can inspect the outputs in a few ways:

- **Full logs**  
  View the complete log at  
  `default.log` (or whatever you set as `log_file`) in the project root.

- **Per-record details**  
  `/run_logs/{dataset}_{metric}_{model}.csv`

- **All record-level entries for the entire run**  
  `/run_logs/{run.json}`

- **Final aggregated scores**  
  `/run_logs/final_scores.json`



## 📝 Citation

If you use HEAR-Kit in your research, please cite our work:

```bibtex
@software{HEAR-Kit2025,
  title = {HEAR-Kit: A Comprehensive Audio Multimodal LLM Evaluation Toolkit},
  author = {ServiceNow},
  year = {2025},
  url = {https://github.com/ServiceNow/HEAR-Kit},
  version = {0.1.0}
}
```

## 📄 License

HEAR-Kit is licensed under the Apache 2.0 License.
