# 🧩 Spoken Language Reasoning
Spoken Language Reasoning refers to the task category involving performing complex, logical operations on information conveyed through speech. These tasks go beyond simple transcription or understanding a single utterance, as they require the model to comprehend context, make inferences, user external tools, folow complex multi-step instructions and answer questions that demand a deeper understanding of the spoken content.

## Sample Config
```yaml
dataset_metric:
  
  # All datasets within **scene_understanding** sub-task category
  - ['mtbench', 'mt_bench_llm_judge']

  # Individual dataset
  - ["mtbench_audio", "mt_bench_llm_judge"]
```

## 📊 Supported Datasets for Spoken Language Reasoning

| Dataset Name                   | Task type       | config | Description                                                                                       | License              |
|-------------------------------|------------------|----- | ---------------------------------------------------------------------------------------------------|----------------------|
| **MTBench**               | Speech Instruction Following          | [spoken_language_reasoning/mtbench](./mtbench/base.yaml)| Speech-based multi-turn complex instruction following dataset      |    Apache-2.0     |
| **IFEVAL**               | Speech Instruction Following          | [spoken_language_reasoning/ifeval](./ifeval/base.yaml)| Speech-based complex instruction following dataset    |    Apache-2.0     |
| **BFCL**               | Speech Function Calling          | [spoken_language_reasoning/bfcl](./bfcl/base.yaml)| Speech-based complex function calling dataset with audio input       |    Apache-2.0    |
| **SPEECH_TO_SQL**               | Speech-to-Coding         | [spoken_language_reasoning/speech_to_sql](./speech_to_sql/base.yaml)| Speech-based dataset involving following instructions to produce executable code        |    Apache-2.0     |