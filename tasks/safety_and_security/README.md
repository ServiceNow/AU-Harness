# 🔐 Safety and Security** 
Safety and Security refers to the task category involving the process of assessing model behavior around safety, robustness, and vulnerability to spoofing or adversarial content.

## Sample Config
```yaml
dataset_metric:  
  # All datasets within **safety** sub-task category
  - ['safety', 'llm_judge_redteaming']

  # Individual dataset
  - ["advbench", "llm_judge_redteaming"]
```

## 📊 Supported Datasets for Safety and Security

| Dataset Name                   | Language Coverage       | config | Description                                                                                       | License              |
|-------------------------------|------------------|----- | ---------------------------------------------------------------------------------------------------|----------------------|
| **advbench**               | English          | [safety/advbench](./safety/advbench)| Speech dataset assessing models' ability to withold response generation against adversary/ harmful instructions      | Apache 2.0        |
| **AVSpoof2017**              | English             | [spoofing/avspoof](./spoofing/avspoof.yaml)| Speech dataset assessing spoofing attack detection accuracy with ‘out in the wild’ conditions                   | CC BY-NC 4.0 |