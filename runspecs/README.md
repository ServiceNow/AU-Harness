# Summary

The framework allows runs by `Task category` and `Task Name` as demonstrated below. Each cell in the table represents the accepted corresponding variables in the `config.yaml` file. The currently `Supported Metrics` are also included for each task and task category for further clarity. 

| Task Category                 | Task Name                                   | Supported Metrics     |
| ----------------------------- | ------------------------------------------- | --------------------- |
| speech_recognition            | asr                                         | word_error_rate       |
|| long_form_ASR                 | word_error_rate                             |									| 
| ---- | ---- | ---|
| paralinguistics               | emotion_recognition                         | llm_judge_binary      |
|| gender_recognition            | llm_judge_binary                            |
|| accent_recognition            | llm_judge_binary                            |
|| speaker_recognition           | llm_judge_binary                            |
|| speaker_diarization           | diarization_metrics                         |
| spoken_language_understanding | speech_qa                                   | llm_judge_binary      |
|| sqqa                          | llm_judge_big_bench_audio, llm_judge_binary |
|| translation                   | bleu, bertscore, meteor                     |
|| scene_understanding           | llm_judge_detailed, llm_judge_binary        |
|| spoken_dialogue_summarization | llm_judge_detailed                          |
|| intent_classification         | llm_judge_binary                            |
|| music_understanding           | llm_judge_binary                            |
| spoken_language_reasoning     | bfcl                                        | bfcl_match_score      |
|                               | speech_to_sql                               | sql_score             |
|                               | ifeval                                      | instruction_following |
| safety_and_security           | safety                                      | detailed_judge_prompt |
|| spoofing                      | detailed_judge_prompt, llm_judge_binary     |