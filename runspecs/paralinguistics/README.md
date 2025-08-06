# Clean System Prompting

Especially for many of the emotion, gender, and accent recognition runspecs, perfomance is better and more representative of LALM capability with good system prompts.

Feel free to use our premade prompts, or make your own

For example:

```yaml
dataset_metric: 
  - ["iemocap_emotion_recognition", "llm_judge_binary"]

filters:
  system_prompts: ["iemocap_emotion"] 
```

Feel free to chain multiple system prompts together:

```yaml
filters:
    system_prompts: ["audio_expert", "try_best"]
```


# CallHome

To run the CallHome dataset for ASR and Speaker Diaraization, follow these steps:

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
  - ["callhome, "speaker_diarization"] #etc.