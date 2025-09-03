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


# CallHome

To run the CallHome dataset for ASR and Speaker Diaraization, follow these steps to prepare the dataset:

## Download the datasets

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
  - ["callhome_eng", "word_error_rate"]
  - ["callhome_spa, "speaker_diarization"] #etc.
```

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
  