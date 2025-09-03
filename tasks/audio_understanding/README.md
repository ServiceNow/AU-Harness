# 🔊 Audio Understanding
Audio Understanding refers to the task category involving the process of comprehending the content and context of an audio signal. These signals include ambient sounds, music, speaker, pitches, etc.  This task category aims to capture the unique characteristics of the audio inputs.

## Sample Config
```yaml
dataset_metric:
  # All datasets within **Audio Understanding** task category
  - all
  
  # All datasets within **scene_understanding** sub-task category
  - ['scene_understanding', 'llm_judge_detailed']

  # Individual dataset
  - ["mu_chomusic_test", "llm_judge_binary"]
```

## 📊 Supported Datasets for Audio Understanding

| Dataset Name                   | Task type       | config | Description                                                                                       | License              |
|-------------------------------|------------------|----- | ---------------------------------------------------------------------------------------------------|----------------------|
| **MU_CHOMUSIC**               | Music Understanding          | [music_understanding/mu_chomusic](./music_understanding/mu_chomusic_test.yaml)| Benchmark designed to evaluate music understanding in multimodal audio-language models      |    CC-BY-SA-4.0     |
| **AUDIOCAPS**               | Scene Understanding          | [scene_understanding/audiocaps](./scene_understanding/audiocaps_test.yaml)| Large-scale audio captioning dataset for sound in the wild    |    MIT    |
| **AUDIOCAPS_QA**               | Scene Understanding            | [scene_understanding/audiocaps_qa](./scene_understanding/audiocaps_qa_test.yaml)| Audio Question Answering dataset for evaluating interactive audio understanding       |    MIT    |
| **CLOTHO_AQA**               | Scene Understanding            | [scene_understanding/clotho_aqa](./scene_understanding/clotho_aqa_test.yaml)| Multimodal translation task where a system analyzes an audio signal and a natural language question, to generate a desirable natural language answer      |    MIT     |
| **WAVCAPS_QA**               | Scene Understanding            | [scene_understanding/wavcaps_qa](./scene_understanding/wavcaps_qa_test.yaml)| Large-scale Audio Question Answering dataset for evaluating interactive audio understanding      |     CC-BY-NC 4.0     |
| **WAVCAPS**               | Scene Understanding           | [scene_understanding/wavcaps](./scene_understanding/wavcaps_test.yaml)| Large-scale weakly-labelled audio captioning dataset      |     CC-BY-NC 4.0     |