import re
import sys

CLAE = "CLAE"
EVAL = "EVAL_"

MODEL_INFO_FILE = "config/models.json"
PAYLOAD_CFG_FILE = "config/payload_cfg.json"
# path where internal usecases are maintained (data/config/chain etc)
# make sure to add slash at the end, if not empty
CLAE_INTERNAL_PATH = "internal/"
CLAE_INTERNAL = "internal"
# set this path in future, if you are moving code from root
# make sure to add slash at the end, if not empty
CLAE_ROOT_PATH = ""
PARENT_RUNSPEC_KEYS = {"aggregator", "child_runspecs", "description"}

MODEL_ALIAS = "MODEL_ALIAS"

PROMPT_FILE_NAME = "prompts.yaml"

MODEL_GPT4 = "openai_gpt4"
MODEL_GPT4O = "openai_gpt4o"
MODEL_GPT4O_MINI = "openai_gpt4o_mini"
MODEL_OPENAI = "openai"
MODEL_OPENAI_JUDGE = "openai_gpt4_judge"
MODEL_OPENAI_GPT41_JUDGE = "openai_gpt41_judge"
OPENAI_JUDGE_NAMES = [MODEL_OPENAI_JUDGE, MODEL_OPENAI_GPT41_JUDGE]

MODEL_ANTHROPIC = "anthropic"
MODEL_JURY = "jury"
MODEL_AZURE_CLIENT = "azure"
MODEL_CLIENT_BEDROCK = "bedrock"
MODEL_CONTENT_MODERATION = "content_moderator"
MODEL_GEMINI = "gemini"
MODEL_GRANITE = "granite"
MODEL_LA_PLATFORME_CLIENT = "la_platforme"
MODEL_LLAMA2 = "llama2"
MODEL_MISTRAL = "mistral"
MODEL_NOWLLM = "nowllm"
MODEL_NOWLLM_REASONING = "nowllm_reasoning"
MODEL_NVIDIA = "nvidia"
MODEL_OPEN_ROUTER = "open_router"
MODEL_QWEN_QWQ_32B = "qwen_qwq_32b"
INPUT_PARAM_FLASK_SKILL = "skills_evaluated"

MODEL_TRANSCRIPTIONS_DEEPGRAM = "audio_transcriptions_deepgram"
MODEL_TRANSCRIPTIONS_NVIDIA = "audio_transcriptions_nvidia"
MODEL_TRANSCRIPTIONS_OPENAI = "audio_transcriptions_openai"

STANDARD_METRICS_PARAM_ID = "id"
STANDARD_METRICS_PARAM_INPUT_PRETOK = "inputs_pretokenized"
STANDARD_METRICS_PARAM_TARGET_PRETOK = "targets_pretokenized"
STANDARD_TOPK_INTENTS = "topK"
STANDARD_EXPECTED_ENTITIES = "expected_entities"
STANDARD_EXPECTED_ENTITY = "expected_entity"
STANDARD_EXPECTED_INTENT = "expected_intent"
STANDARD_LABEL = "label"
STANDARD_LANGUAGE = "language"
STANDARD_KB_DOCUMENTS = "kb_documents"
STANDARD_CATALOG_ITEMS = "catalog_items"
STANDARD_CATALOGS = "catalogs"
STANDARD_CONVERSATION = "conversation_history"
STANDARD_ANSWERABLE = "answerable"
STANDARD_UNANSWERABLE = "unanswerable"
STANDARD_RATING = "rating"
QNA_INVALID_PROMPT = "Invalid_prompt"

LANGUAGE_METRICS_ENGLISH_JSON_KEYS = "language_metrics_json_keys_in_english"
LANGUAGE_METRICS_JSON_VALUES_EXPECTED_LANGUAGE = "language_metrics_json_values_expected_language"
LANGUAGE_METRICS_TEXT_EXPECTED_LANGUAGE = "language_metrics_text_expected_language"
LANGUAGE_METRICS_FLUENCY_SCORE = "language_metrics_fluency_score"
LANGUAGE_METRICS_FLUENCY_JUSTIFICATION = "language_metrics_fluency_justification"
ALL_LANGUAGE_METRICS = [
    LANGUAGE_METRICS_ENGLISH_JSON_KEYS,
    LANGUAGE_METRICS_JSON_VALUES_EXPECTED_LANGUAGE,
    LANGUAGE_METRICS_TEXT_EXPECTED_LANGUAGE,
    LANGUAGE_METRICS_FLUENCY_SCORE,
]

# supported data keys
ROLE_KEY_ASSISTANT = "assistant"
ROLE_KEY_MODEL = "model"  # equivalent to assistant for gemini models
ROLE_KEY_SYSTEM = "system"
ROLE_KEY_USER = "user"
ROLE_KEY_TOOL_RESULT = "tool"
ROLE_KEY_CONTENT = "content"

# The order is important for the return value of `BaseFormatter.get_role_specific_msg()`.
ROLES = (ROLE_KEY_SYSTEM, ROLE_KEY_USER, ROLE_KEY_ASSISTANT, ROLE_KEY_TOOL_RESULT, ROLE_KEY_CONTENT)

# regex patterns for user, system, and assistant tags
USER_PROMPT_REGEX = re.compile(r"(?:<user>)(.*)(?:</user>)", re.DOTALL)
SYSTEM_PROMPT_REGEX = re.compile(r"(?:<system>)(.*)(?:</system>)", re.DOTALL)
ASSISTANT_PROMPT_REGEX = re.compile(r"(?:<assistant>)(.*)(?:</assistant>)", re.DOTALL)

# supported common tags
TAG_SYSTEM = "SYSTEM"
TAG_USER = "USER"
TAG_ASSISTANT = "ASSISTANT"
# extra tag for Llama2
TAG_INSTRUCTION = "INSTRUCTION"
TAG_TURN = "TURN"
TAG_TOOL = "TOOL"
TAG_CONTENT = "CONTENT"
TAG_TOOL_CALL = "TOOL_CALL"

# TAG start and end
TAG_START = "START"
TAG_END = "END"

INVERTED_METRICS = (
    "toxic",
    "relative_toxicity",
    "inconclusive",  # biasness
    "fp",
    "fn",
    "inaccuracy",  # helpfulness (search_qa)
    "toxicity",
    "hate_severity",
    "self_harm_severity",
    "sexual_severity",
    "violence_severity",
    # ToxicityMetrics, AzureContentSafetyMetrics
    "ordinal_distance_score",  # OrdinalDistanceMetrics
    "format_invalid_candidate",
    "format_invalid_reference",
    "format_mismatch",
)  # SlotFillingMetrics
INVERTED_METRIC_INDICATOR = "↓"

TGI_OPENAI_PARAM_NAME_MAPPER = {
    "max_new_tokens": "max_tokens",
    "do_sample": "",
    "num_beams": "",
}

KEEP_PARAMS = [
    "stop",
    "reasoning_effort",
    "include_reasoning",
    "provider",
    "seed",
    "extra_body",
]

MODEL_PARAMS_TO_KEEP_IN_RUN_PARAMS = []

# multi turn related
KEY_1TURN = "1turn"
KEY_2TURN = "2turn"
KEY_FAILED = f"failed_rating_{INVERTED_METRIC_INDICATOR}"

# Config file paths

# redteaming config files
# {"english": "en", "german": "de", "french": "fr", "italian": "it", "dutch": "nl", "japanese": "ja", "spanish": "es", "french (canadian)": "fr_ca", "portugese (brazillian)":"pt_br"}
REDTEAMING_SUPPORTED_LANGUAGES = [
    "en",
    "de",
    "fr",
    "it",
    "nl",
    "ja",
    "es",
    "fr_ca",
    "pt_br",
]

#inference server type
INFERENCE_SERVER_TGI = "tgi"
INFERENCE_SERVER_TGI_GENERATE = "tgi_generate"
INFERENCE_SERVER_VLLM = "vllm"
INFERENCE_SERVER_VLLM_COMPLETIONS = "vllm_completions"
INFERENCE_SERVER_AZURE = "azure"  # this uses http raw post call, may delete in future
INFERENCE_SERVER_AZURE_AI = "azure_ai"  # adding new flow for azure ai inference client
INFERENCE_SERVER_HF = "hf"
INFERENCE_SERVER_API = "api"
INFERENCE_SERVER = "inference_server"
INFERENCE_SERVER_OPENAI = "openai"

# Redteaming constants
THRESHOLD_JAILBREAK = 2
OPENAI_VIOLATION_SUBSTRING = "ResponsibleAIPolicyViolation"

MAX_RECORD_COUNT = sys.maxsize

# Judge model : Default ('openai_gpt4')
JUDGE_MODEL = "openai_gpt41_judge"
REDTEAMING_JUDGE_MODEL = "openai_gpt41_judge"

# Model Parameter Mapper file
MODEL_PARAM_MAPPER_FILE = "config/model_param_mapper.json"
THINKING_PARSER_FILE = "config/thinking_parsers.json"

# Force use of cli model parameters
CLI_MODEL_PARAM_FORCE_USE = "apply_all_time"

# Default threshold for content moderation model
CONTENT_MODERATION_DEFAULT_THRESHOLD = 0.5

# supported payload key in the config file
PAYLOAD_JSON = "payload_json"
TEST_PAYLOAD = "test_payload"
RESPONSE_KEY = "response_key"

CALL_ID_TEST = "Test"
MESSAGES_TEST_AVAILABILITY = [{"role": "user", "content": "Are you available?"}]



ROUND_DIGITS = 3

FORMATTER_PLAIN_TEXT = "plain_text"
FORMATTER_LLAMA = "llama"
FORMATTER_CHAT_COMPLETIONS = "chat_completions"
FORMATTER_BEDROCK = "bedrock"
FORMATTER_GEMINI = "gemini"
FORMATTER_NOWLLM = "nowllm"
FORMATTER_NOWSLM = "nowslm"
FORMATTER_NOWLLM_REASONING = "nowllm_reasoning"

APPEND_ID_SEPERATOR = "***DATASET ID***"

# Constants for capturing api errors
ERROR_PATTERN = "~~##@@ERROR_{}@@##~~_~~##@@ERRORMSG_{}@@##~~"
ERROR_MSG_SPLIT_STR = "~~##@@ERRORMSG_"
ERROR_CODE_SPLIT_STR = "~~##@@ERROR_"
SERVER_CODE = "server_code"
SERVER_MESSAGE = "server_message"
SERVER_CODE_SPLIT_STR = "_" + SERVER_CODE
SERVER_MESSAGE_SPLIT_STR = "_" + SERVER_MESSAGE
SERVER_FAILURE = "_server_failure"
EVALUATION_FAILURE = "_evaluation_failure"
RECORDS_SELECTED = "_records_selected"
RECORDS_FAILED = "_records_failed"
SKIP_KEYS_IN_OUTPUT_FILE = [RECORDS_SELECTED, RECORDS_FAILED]
ASR_LIST = "asr_list"
ASR = "asr"
FAILURE_RATE = "failure_rate"
AGGREGATE_AFR = "aggregate_afr"
FAILURE_THRESHOLD = "failure_threshold"

TEST_MODEL = "test_model"
FORMATTER = "formatter"
TOKEN = "token"
TOKEN_TYPE = "token_type"
BEARER = "bearer"
BASIC = "basic"
AUTH_TOKEN = "auth_token"
URL = "url"
WEIGHTS = "weights"
SEED = "seed"
REPLICA = "_replica_"

GOOGLE_API_AUTH_BASE_URL = "https://www.googleapis.com/auth/cloud-platform"

VLLM_MAX_TOKEN_RETRY_BUFFER = 50
