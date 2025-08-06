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
