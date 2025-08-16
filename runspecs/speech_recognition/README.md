# 🗣️ Speech Recognition

Perfomance can improve significantly with good prompting - use our prepared prompts, or create and define your own [here](../../prompts/prompt_add_ons.yaml)!

Sample config for Speech Recognition task is provided in [sample_asr_config](./sample_asr_config.yaml). 
Suggested dataset-metric pairs are provided in [asr_suggestions.yaml](./asr_suggestions.yaml)
```yaml
dataset_metric:
  - ["librispeech_test_clean", "word_error_rate"]

filters:
  prompt_add_ons: ["asr"]
```
