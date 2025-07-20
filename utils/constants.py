#inference server types
INFERENCE_SERVER_VLLM_CHAT_COMPLETION = "vllm"
OPENAI_CHAT_COMPLETION = "openai"
INFERENCE_SERVER_VLLM_TRANSCRIPTION = "vllm_transcription"
OPENAI_TRANSCRIPTION = "openai_transcription"
#extras
ROUND_DIGITS = 3
VLLM_MAX_TOKEN_RETRY_BUFFER = 50
INVERTED_METRIC_INDICATOR = "↓"

# Dictionary mapping metric names to their implementation details (module, class)
metric_map = {
    "word_error_rate": ("metrics.word_error_rate_metrics", "WERMetrics"),
    "bleu": ("metrics.bleu_metrics", "BleuMetrics"),
    "llm_judge_binary": ("metrics.llm_judge", "BinaryLLMJudgeMetric"),
    "llm_judge_detailed": ("metrics.llm_judge", "DetailedLLMJudgeMetric"),
    "llm_judge_callhome": ("metrics.llm_judge", "CallHomeLLMJudgeMetric"),
    "meteor": ("metrics.meteor_score", "MeteorScore"),
    "llm_judge_big_bench_audio": ("metrics.llm_judge", "BigBenchAudioLLMJudgeMetric"),
    'bertscore': ("metrics.bertscore", "BertScore"),
    'diarization_metrics': ('metrics.diarization_metrics','DiarizationMetrics')
}