| Task Category                 | Task Name                     | Dataset/Benchmark        | Metric                      | Num_samples | Voxtral-Mini-3B | Qwen-2.5-Omni-7B | GPT-4o |
| ----------------------------- | ----------------------------- | ------------------------ | --------------------------- | ----------- | ------------------------------ | ---------------- | ------ |
| Speech Recognition            | ASR                           | Librispeech              | WER                         | 2617        | 2.10                          | 1.74           | 6.25 |
| Paralinguistics               | Emotion                       | MELD                     | llm_judge_binary            | 2610        | 28.4                           | 49.8             | 20.2   |
| Paralinguistics               | Gender                        | IEMOCAP                  | llm_judge_binary            | 1003        | 54.9                           | 85.8             | 0*      |
| Paralinguistics               | Accent                        | VoxCeleb                 | llm_judge_binary            | 4818        | 13                             | 28.7             |  0*      |
| Paralinguistics               | Speaker Recognition           | mmau_mini                | llm_judge_binary            | 1000        | 45.8                           | 62.3             | 42     |
| Paralinguistics               | Speaker Diarization           | CallHome                 | WDER                        | 112         | 35.38                          | 35.4             | 37.14  |
| Spoken Language Understanding | Spoken QA                     | public_sg_speech_qa_test | llm_judge_detailed          | 688         | 62.12                          | 69.4             | 70.2   |
| Spoken Language Understanding | Spoken Query QA               | BigBench Audio           | llm_judge_big_bench_audio   | 1000        | 43.5                           | 53.8             | 65     |
| Spoken Language Understanding | Speech Translation            | Covost2 (zh-CN->EN)      | BLEU                        | 4898        | 15.27                          | 28.41            | 21.68  |
| Spoken Language Understanding | Spoken Dialogue Summarization | mnsc_sds (P3)            | llm_judge_detailed          | 100         | 52.2                           | 52               | 61.2   |
| Spoken Language Understanding | Intent Classification         | SLURP                    | llm_judge_binary            | 200         | 42.5                           | 57               | 48     |
| Audio Understanding           | Scene Understanding           | audiocaps_qa             | llm_judge_detailed          | 313         | 14.96                          | 38.4             | 15.08  |
| Audio Understanding           | Music Understanding           | mu_chomusic_test         | llm_judge_binary            | 1187        | 45.4                           | 59.3             | 50.2   |
| Spoken Language Reasoning     | Speech Instruction Following  | IFEval                   | instruction_following_score | 345         | 38.06                          | 50.83            | 72.15  |
| Spoken Language Reasoning     | Speech Instruction Following  | MTBench                  | llm_judge_mt_bench          | 80          | 64.12                          | 62.88            | 62.44  |
| Spoken Language Reasoning     | Speech-to-Coding              | Spider                   | sql_score (EM)              | 1001        | 30.17                          | 38.46            | 45.15  |
| Spoken Language Reasoning     | Speech Function Calling       | BFCL                     | bfcl_match_score            | 1240        | 78.5                           | 68               | 86.65  |
| Safety and Security           | Safety                        | advbench                 | redteaming_judge            | 520         | 78.5                           | 98.3             | 88.1   |
| Safety and Security           | Spoofing                      | avspoof                  | llm_judge_binary            | 200         | 91.5                           | 30               | 0*      |

\* Reported performance is impacted by content filtering system from Azure OpenAI