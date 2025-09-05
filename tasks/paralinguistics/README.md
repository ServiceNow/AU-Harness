# 🎭 Paralinguistics  
Paralinguistics refers to the task category involving the process of intepreting non-verbal signals accompanying the speech and audio. This task category provides differentiating factor for speech and audio systems by capturing the nuances that cannot be conveyed through text alone. 

## Sample Config
```yaml
dataset_metric:
  # All datasets within **Paralinguistics** task category (including: asr + code_switching_asr + long_form_asr)
  - ['paralinguistics', 'llm_judge_binary']
  
  # All datasets within **emotion_recognition** sub-task category
  - ['emotion_recognition', 'llm_judge_binary']

  # Individual dataset
  - ["iemocap_emotion_recognition", "llm_judge_binary"]
```


## CalllHome Dataset
CallHome dataset can be utilized for Speaker Diarization evaluation. Follow the steps presented in [speech_recognition/asr/callhome_asr](../speech_recognition/asr/callhome_asr/README.md) for further instructions on the dataset preparation.

## 📊 Supported Datasets for Paralinguistics

| Dataset Name                   | Task type       | config | Description                                                                                       | License              |
|-------------------------------|------------------|----- | ---------------------------------------------------------------------------------------------------|----------------------|
| **IEMOCAP_EMOTION**               | Emotion Recognition          | [emotion_recognition/iemocap_emotion_recognition](./emotion_recognition/iemocap_emotion_recognition.yaml)| Multimodal and multi-speaker dataset with humans' emotion expression elicitation      |    GPL-3.0     |
| **MELD_EMOTION**               | Emotion Recognition          | [emotion_recognition/meld_emotion](./emotion_recognition/meld_emotion_test.yaml)| Dataset of multi-party conversation with various emotional expressions    |    GPL-3.0     |
| **MELD_SENTIMENT**               | Emotion Recognition          | [emotion_recognition/meld_sentiment](./emotion_recognition/meld_sentiment_test.yaml)| Dataset of multi-party conversation with various sentiment expressions       |    GPL-3.0     |
| **VOXCELEB_ACCENT**               | Accent Recognition          | [accent_recognition/voxceleb_accent](./accent_recognition/voxceleb_accent_test.yaml)| Speech datasets from speakers with different ethnicities, accents, professions and ages      |    CC BY 4.0     |
| **MNSC_AR_DIALOGUE**               | Accent Recognition          | [accent_recognition/mnsc_ar_dialogue](./accent_recognition/mnsc_pqa_ar_dialogue_test.yaml)| Dataset to evaluate speech models' capability in recognizing human's emotions      |     MNSC: Publicly released     |
| **MNSC_AR_SENTENCE**               | Accent Recognition          | [accent_recognition//mnsc_ar_sentence](.//accent_recognition/mnsc_pqa_ar_sentence_test.yaml)| Dataset to evaluate speech models' capability in recognizing human's emotions      |     MNSC: Publicly released     |
| **IEMOCAP_GENDER**               | Gender Recognition          | [gender_recognition/iemocap_gender_recognition](./gender_recognition/iemocap_gender_recognition.yaml)| Multimodal and multi-speaker dataset with humans' emotion expression elicitation      |   GPL-3.0      |
| **VOXCELEB_Gender**               | Gender Recognition          | [gender_recognition/voxceleb_gender](./gender_recognition/voxceleb_gender_test.yaml)| Speech datasets from speakers with different ethnicities, accents, professions and ages      |    CC BY 4.0     |
| **MNSC_GR_DIALOGUE**               | Gender Recognition          | [gender_recognition/mnsc_gr_dialogue](.//gender_recognition/mnsc_pqa_gr_dialogue_test.yaml)| Dataset to evaluate speech models' capability in recognizing human's emotions      |    MNSC: Publicly released     |
| **MNSC_GR_SENTENCE**               | Gender Recognition          | [gender_recognition/mnsc_gr_sentence](./gender_recognition/mnsc_pqa_gr_sentence_test.yaml)| Dataset to evaluate speech models' capability in recognizing human's emotions      |    MNSC: Publicly released     |
| **MMAU_MINI**              | Speaker Recognition             | [speaker_recognition/mmau_mini](./speaker_recognition/mmau_mini.yaml)| Multi-modal Audio Datasets from multiple speakers                |  Apache 2.0 |
| **CALLHOME_DIARIZATION**              | Speaker Diarization             | [speaker_diarization/callhome_diarization_eng](./speaker_diarization/callhome_diarization_eng.yaml)| Multilingual telephone conversations between multiple native speakers                  | CC-BY-NC-SA-4.0 |
  