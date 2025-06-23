from jinja2 import Template

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
from utils.util import get_common_prompts, get_language_prompt_template


class Language:
    """Maps language name, dynamic translation code, and alternate code."""

    def __init__(self, name, dynamic_translation_code, default_language_code):
        self.name = name
        self.dynamic_translation_code = dynamic_translation_code
        self.default_language_code = default_language_code


LANGUAGES = [
    Language("dutch", "nl", "nl"),
    Language("english", "en", "en"),
    Language("canadian french", "fr", "fq"),
    Language("french", "fr", "fr"),
    Language("german", "de", "de"),
    Language("italian", "it", "it"),
    Language("japanese", "ja", "ja"),
    Language("brazilian portuguese", "pt", "pb"),
    Language("portuguese", "pt", "pt"),
    Language("spanish", "es", "es"),
]

# Data structures for easy look ups
all_codes = {lang.dynamic_translation_code for lang in LANGUAGES} | {lang.default_language_code for lang in LANGUAGES}
default_code_to_language = {lang.default_language_code: lang.name for lang in LANGUAGES}
language_to_dynamic_code = {lang.name: lang.dynamic_translation_code for lang in LANGUAGES}
language_to_alternate_code = {lang.name: lang.default_language_code for lang in LANGUAGES}
default_language_codes = [lang.default_language_code for lang in LANGUAGES]


def get_language_name_by_default_code(code):
    """Takes in alternate code and returns language name (ex. fq -> canadian french)."""
    return default_code_to_language.get(code)


def get_code_from_language(language_name):
    """Takes in language name and returns dynamic translation code (ex. spanish -> es)."""
    return language_to_dynamic_code.get(language_name.lower())


def get_code_from_alternate_language(language_name):
    """Takes in language name and returns alternate code (ex. chinese -> zh-cn)."""
    return language_to_alternate_code.get(language_name.lower())


def is_valid_language_code(code):
    """Checks if the given code corresponds to a valid language (dynamic or alternate)."""
    return code in all_codes


def get_language_prompts(language: str) -> str:
    """Return system and assistant language prompts based on the language."""
    common_prompts = get_common_prompts("default")
    language_user_prompt = common_prompts.get("prompt_templates.default.language_prompt_template", "")
    language_user_prompt = Template(language_user_prompt)
    user_injection = language_user_prompt.render(language=language)

    return user_injection


def add_language_prompt(
    input_text: list,
    language_prompt_template_name: str,
    language_code: str = "en",
    language_prompt: dict = None,
) -> list:
    """Add language prompt to the input text."""
    language_name = get_language_name_by_default_code(language_code).title()
    assistant_language_name = language_name
    # If language is Japanese, instruct model to use full sentences
    if language_code == "ja":
        assistant_language_name = f"{language_name} using full sentences."
    if not language_prompt:
        language_prompt = get_language_prompt_template(language_prompt_template_name)

    # Convert list of dicts to a single dict
    inputs = {k: v for item in input_text for k, v in item.items()}

    # Loop through tags in language_prompt and add the prompt to the proper input text
    for prompt_tag, prompt_value in language_prompt.items():
        template = prompt_value.get("default")
        prepend = prompt_value.get("prepend", False)
        if template is not None:
            rendered_prompt = Template(template).render(
                language=language_name, assistant_language=assistant_language_name
            )
            logger.debug(f"Language prompt: {rendered_prompt}")
            if prompt_tag in inputs:
                if prepend:
                    inputs[prompt_tag] = f"{rendered_prompt}\n{inputs[prompt_tag]}"
                else:
                    inputs[prompt_tag] += f"\n{rendered_prompt}"
            else:
                inputs[prompt_tag] = rendered_prompt

    # convert back to original list format and force order
    # Todo: Handle multi-turn conversations
    return [{k: inputs[k]} for k in ["system", "user", "assistant"] if k in inputs]
