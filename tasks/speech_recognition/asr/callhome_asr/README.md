# CallHome

To run the CallHome dataset for ASR and Speaker Diarization, follow these steps:

##### Download the datasets

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

##### Create a local HF dataset
Run the following commands to create a local HF dataset
```python
cd AU-Harness/
python data/scripts/prepare_callhome_dataset.py --data_dir /private_datasets/{specific_language_folder}
```
This creates a hf_dataset folder in the same path as data_dir

Run the specific dataset or all datasets across word_error_rate(utterance by utterance), llm_judge_binary (holistic view of 30 second turn by turn transcription), or speaker diarization.

```yaml
dataset_metric:
  - [callhome_asr, word_error_rate]
  - [callhome_diarization, speaker_diarization] 
```