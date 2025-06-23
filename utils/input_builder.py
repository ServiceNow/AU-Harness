from ast import literal_eval

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
from utils import constants

from utils.language import add_language_prompt
from utils.model_info import get_model_info


class InputBuilder:
    """Build the input for the LLM model."""

    def __init__(self, model_name: str):
        self.model_name = model_name


    @staticmethod
    def _get_chat_template_params(model_name: str) -> tuple[bool, str]:
        """Get the chat template information from model info."""
        has_system_turn = True
        try:
            model_info = get_model_info(model_name)
            has_system_turn = model_info.get("has_system_turn", has_system_turn)
            model_type = model_info.get("model_type") if model_info.get("model_type") else ""
            return has_system_turn, model_type
        except Exception:
            # get_model_info throws an exception for model defined as jinja or empty
            return has_system_turn, ""

    @staticmethod
    def _get_model_prefix(model_name: str) -> str:
        """Get the model prefix from the model name."""
        model_prefix = ""
        for char in model_name:
            if char.isalpha():
                model_prefix += char
            else:
                break
        return model_prefix

    def is_language_prompt_needed(self, run_params: dict) -> bool:
        """Verify if language prompt should be added. Skips language injection for nowllm reasoning models."""
        return run_params.get("language") and self.model_name != constants.MODEL_OPENAI_JUDGE

    def build_conversation(self, messages: list[dict], run_params: dict | None = None) -> str:
        """Build the conversation from the input dictionary by simply concatenating message contents."""
        run_params = run_params or {}
        inputs_pretokenized = [{item["role"]: item["content"]} for item in messages]
        if self.is_language_prompt_needed(run_params):
            language_prompt = run_params.get("language_prompt")
            if language_prompt:
                language_prompt = literal_eval(language_prompt)
            language_prompt_template = run_params.get("language_template", "default")
            inputs_pretokenized = add_language_prompt(
                inputs_pretokenized, language_prompt_template, run_params.get("language"), language_prompt
            )
        # Concatenate all message contents in order
        return "\n".join(next(iter(d.values())) for d in inputs_pretokenized)
