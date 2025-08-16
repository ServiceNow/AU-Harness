Perfomance improves significantly with good system prompting - use our premade prompts, out create your own!

```yaml
dataset_metric:
  # All datasets within **SPEECH RECOGNITION** task category (including: asr + code_switching_asr + long_form_asr)
  - ['speech_recognition', 'word_error_rate']
  
  # All datasets within **long_form_asr** sub-task category
  - ['long_form_asr', 'word_error_rate']

  # Individual dataset
  - ["librispeech_test_clean", "word_error_rate"]

filters:
  system_prompts: ["asr_expert"]
```
