#inference server types
INFERENCE_SERVER_VLLM_CHAT_COMPLETION = 'vllm'
OPENAI_CHAT_COMPLETION = 'openai'
INFERENCE_SERVER_VLLM_TRANSCRIPTION = 'vllm_transcription'
OPENAI_TRANSCRIPTION = 'openai_transcription'
#extras
ROUND_DIGITS = 3
VLLM_MAX_TOKEN_RETRY_BUFFER = 50
INVERTED_METRIC_INDICATOR = '↓'

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
}

# Dictionary mapping language names to their standard ISO 639-1 codes
language_map = {
    'english': 'en',
    'japanese': 'ja',
    'spanish': 'es',
    'french': 'fr',
    'german': 'de',
    'chinese': 'zh',
    'russian': 'ru',
    'portuguese': 'pt',
    'arabic': 'ar',
    'italian': 'it',
    'dutch': 'nl',
    'turkish': 'tr',
    'swedish': 'sv',
    'catalan': 'ca',
    'persian': 'fa',
    'estonian': 'et',
    'mongolian': 'mn',
    'latvian': 'lv',
    'slovenian': 'sl',
    'tamil': 'ta',
    'indonesian': 'id',
    'welsh': 'cy',
    'hindi': 'hi',
    'bengali': 'bn',
    'korean': 'ko',
    'vietnamese': 'vi',
    'polish': 'pl',
    'ukrainian': 'uk',
}

def get_language_code(language_input):
    """
    Get standardized language code from any language input format
    (code or full name, case insensitive)
    
    Args:
        language_input (str): Language input in any format (code, name, mixed case, etc.)
        
    Returns:
        str: Standardized language code or the original input if no mapping found
    """
    if language_input is None:
        return None
        
    # Convert input to lowercase for case-insensitive lookup
    normalized_input = str(language_input).lower().strip()
    
    # Check if the input is already a valid language code (2-letter code)
    if normalized_input in language_map.values():
        return normalized_input
    
    # Otherwise look up in the language map
    return language_map.get(normalized_input, language_input)
