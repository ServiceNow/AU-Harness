# 🗣️ Speech Recognition
Speech Recognition refers to the task category involving the process of converting spoken language into into a machine-readable text format. This task category serves as the critical step for machines to understand and process humans' speech.

## Sample Config
```yaml
dataset_metric:
  # All datasets within **SPEECH RECOGNITION** task category (including: asr + code_switching_asr + long_form_asr)
  - ['speech_recognition', 'word_error_rate']
  
  # All datasets within **long_form_asr** sub-task category
  - ['long_form_asr', 'word_error_rate']

  # Individual dataset
  - ["librispeech_test_clean", "word_error_rate"]
```

## 📊 Supported Datasets for Speech Recognition

| Dataset Name                   | Language Coverage       | config | Description                                                                                       | License              |
|-------------------------------|------------------|----- | ---------------------------------------------------------------------------------------------------|----------------------|
| **LibriSpeech**               | English          | [asr/librispeech](./asr/librispeech/librispeech_test_clean.yaml)| Audiobook-derived speech corpus with clean and noisy subsets.      | CC BY 4.0        |
| **Common Voice**              | Multilingual             | [asr/common_voice](./asr/common_voice_15/base.yaml)| Crowdsourced multilingual dataset from Mozilla.                   | CC0 1.0 Universal |
| **VoxPopuli**                 | Multilingual              | [asr/voxpopuli](./asr/voxpopuli/base.yaml)| A large-scale multilingual speech corpus collected from European Parliament recordings.  | CC0  |
| **Multilingual LibriSpeech (MLS)** | Multilingual           | [asr/librispeech_multilingual](./asr/librispeech_multilingual/base.yaml) | Extension of LibriSpeech including English, German, French, Spanish, Italian, Dutch, Polish, Portuguese. | CC BY 4.0            |
| **TEDLIUM3**                  | English          | [asr/tedlium3](./asr/tedlium3/tedlium3_test.yaml) | Transcribed TED talks suitable for ASR and speaker adaptation.                                    | CC BY-NC-ND 3.0      |
| **PEOPLE'S SPEECH**          | English          | [asr/peoples_speech](./asr/peoples_speech/peoples_speech_test.yaml) | Large-scale open-source English speech recognition dataset with diverse speakers, accents and domains.                                     | CC-BY-SA      |
| **SPGISPEECH**                  | English          | [asr/spgispeech](./asr/spgispeech/spgispeech_test.yaml) | Transcriptions of Financial Meetings.                                    | Kensho User Agreement      |
| **AISHELL-1**                 | Mandarin Chinese | [asr/aishell_1](./asr/aishell_1/aishell_1_test.yaml) | High-quality open-source Mandarin speech dataset.                                                 | Apache 2.0           |
| **GigaSpeech**                | Multilingual          | [asr/gigaspeech](./asr/gigaspeech/gigaspeech_test.yaml) | Large-scale audio and transcription corpus for end-to-end ASR.       | Apache 2.0           |
| **AMI Meeting Corpus**        | English          | [asr/ami](./asr/ami/base.yaml) | Multispeaker meeting recordings with annotations for ASR and diarization.       | CC BY 4.0            |
| **FLEURS**                    | Multilingual              | [asr/fleurs](./asr/fleurs/base.yaml)| Google's aligned multilingual speech dataset for ASR and translation.         | CC BY 4.0            |
| **CallHome**                     | Multilingual          | [asr/callhome_asr](./asr/callhome_asr/base.yaml) | Conversational Speech corpus containing dialogues across multiple languages                        | LDC User Agreement for Non-Members           |
| **MNSC**                     | Multilingual          | [asr/mnsc_asr](./asr/mnsc/base.yaml) |   Large-scale multitask speech corpus                     |       MNSC: Publicly released    |
| **Earnings21/22**                     | English          | [long_form_asr/earnings21](./long_form_asr/earnings21.yaml) |  Real-world earnings call recordings for long-form, domain-specific speech evaluation                      |   CC-BY-SA-4.0        |
| **SEAME**                     | Mandarin-English          | [code_switching_asr/seame_dev_man](./code_switching_asr/seame_dev_man.yaml) |   Open-source Mandarin-English code-switching speech dataset.                      |   	LDC2015S04       |