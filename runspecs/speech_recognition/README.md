# 🗣️ Speech Recognition
Speech Recognition refers to the task category involving the process of converting spoken language into written text.

Perfomance can improve significantly with good prompting - use our prepared prompts, or create and define your own [here](../../prompts/prompt_add_ons.yaml)!

## Sample Config
Sample config for Speech Recognition task is provided in [sample_asr_config.yaml](./sample_asr_config.yaml).

Suggested dataset-metric pairs are provided in [asr_suggestions.yaml](./asr_suggestions.yaml).
```yaml
dataset_metric:
  - ["librispeech_test_clean", "word_error_rate"]

filters:
  prompt_add_ons: ["asr"]
```

## 📊 Supported Datasets for Speech Recognition

| Dataset Name                   | Language Coverage       | config | Description                                                                                       | License              |
|-------------------------------|------------------|----- | ---------------------------------------------------------------------------------------------------|----------------------|
| **LibriSpeech**               | English          | [general_asr](./asr/general_asr.json)| Audiobook-derived speech corpus with clean and noisy subsets.      | CC BY 4.0        |
| **Common Voice**              | Multilingual             | [common_voice](./asr/common_voice.json)| Crowdsourced multilingual dataset from Mozilla.                   | CC0 1.0 Universal |
| **VoxPopuli**                 | Multilingual              | [voxpopuli](./asr/voxpopuli.json)| A large-scale multilingual speech corpus collected from European Parliament recordings.  | CC0  |
| **Multilingual LibriSpeech (MLS)** | Multilingual           | [mls](./asr/mls.json) | Extension of LibriSpeech including English, German, French, Spanish, Italian, Dutch, Polish, Portuguese. | CC BY 4.0            |
| **TEDLIUM3**                  | English          | [general_asr](./asr/general_asr.json) | Transcribed TED talks suitable for ASR and speaker adaptation.                                    | CC BY-NC-ND 3.0      |
| **PEOPLE'S SPEECH**          | English          | [general_asr](./asr/general_asr.json) | Large-scale open-source English speech recognition dataset with diverse speakers, accents and domains.                                     | CC-BY-SA      |
| **SPGISPEECH**                  | English          | [general_asr](./asr/general_asr.json) | Transcriptions of Financial Meetings.                                    | Kensho User Agreement      |
| **AISHELL-1**                 | Mandarin Chinese | [general_asr](./asr/general_asr.json) | High-quality open-source Mandarin speech dataset.                                                 | Apache 2.0           |
| **GigaSpeech**                | Multilingual          | [gigaspeech](./asr/gigaspeech.json) | Large-scale audio and transcription corpus for end-to-end ASR.       | Apache 2.0           |
| **AMI Meeting Corpus**        | English          | [general_asr](./asr/general_asr.json) | Multispeaker meeting recordings with annotations for ASR and diarization.       | CC BY 4.0            |
| **FLEURS**                    | Multilingual              | [fleurs](./asr/fleurs.json)| Google's aligned multilingual speech dataset for ASR and translation.         | CC BY 4.0            |
| **CallHome**                     | Multilingual          | [callhome_asr](./asr/callhome_asr.json) | Conversational Speech corpus containing dialogues across multiple languages                        | LDC User Agreement for Non-Members           |
| **MNSC**                     | Multilingual          | [mnsc_asr](./asr/mnsc_asr.json) |   Large-scale multitask speech corpus                     |       MNSC: Publicly released    |
| **Earnings21/22**                     | English          | [long_form_asr](./long_form_asr.json) |  Real-world earnings call recordings for long-form, domain-specific speech evaluation                      |   CC-BY-SA-4.0        |
| **SEAME**                     | Mandarin-English          | [code_switching_asr](./code_switching_asr.json) |   Open-source Mandarin-English code-switching speech dataset.                      |   	LDC2015S04       |