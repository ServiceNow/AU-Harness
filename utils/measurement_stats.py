import json

import regex as re

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
from models.model_response import ModelResponse
from utils import constants
from utils.json_util import extract_and_load_json_ast, extract_and_load_json_iter, is_minified, is_valid_json
from utils.stats import Stats


class MeasurementStats(Stats):
    """Singleton instance to track measurements for all LLM outputs.

    Currently, the class tracks the following measurements:
    - valid_json: Whether the output is a valid JSON
    - extractable_json: Whether the output is a valid JSON or contains a valid JSON that can be extracted from the text.
    - extra_text_around_json: Whether there was extra text around the JSON (JSON needed to be extracted).
    - minified_json: Whether the JSON is minified.
    - minified_json_non_strict: Whether the JSON is minified but without strict separator check.
    - compacted_json: Whether the JSON is compacted (has newlines but no tabs) or minified.
    """

    def append_prefill_if_present(self, inference_type: str, model_response: ModelResponse) -> tuple[str, str]:
        """Append prefill to the llm response if it exists. Else return the response as is."""
        input_prompt = model_response.input_prompt
        llm_response = model_response.llm_response if 200 <= model_response.response_code < 300 else ""
        if inference_type == constants.INFERENCE_SERVER_TGI:
            if "<|assistant|>" in input_prompt:
                try:
                    input_prompt_json = extract_and_load_json_ast(input_prompt)
                    inputs = input_prompt_json["inputs"]
                    match = re.search(r"(?<=<\|assistant\|>).*?(?=<\|end\|>|$)", inputs, re.DOTALL)
                    assistant_text = match.group(0).strip()
                    return assistant_text + llm_response, assistant_text
                except Exception as e:
                    logger.error(f"Error extracting assistant text from input prompt for TGI inference: {e}")
        elif inference_type == constants.INFERENCE_SERVER_TRITON:
            try:
                try:
                    # if json mode was specified to triton, we don't need to append prefill anymore
                    _ = json.loads(llm_response, strict=False)
                    return llm_response, ""
                except Exception:
                    input_prompt_json = extract_and_load_json_ast(input_prompt)
                    data = json.loads(input_prompt_json["inputs"][0]["data"][0])
                    if any(d.get("role") == "assistant" for d in data):
                        assistant_text = next(d["content"] for d in data if d.get("role") == "assistant")
                        return assistant_text + llm_response, assistant_text
            except Exception as e:
                logger.error(f"Error extracting data from input prompt for Triton inference: {e}")
        return llm_response, llm_response

    def add_all_stats(
        self, runspec_name: str, model_name: str, model_response: ModelResponse, inference_type: str = ""
    ) -> None:
        """Add all measurements to the stats list for one specific llm output."""
        output, _ = self.append_prefill_if_present(inference_type, model_response)
        self.add(runspec_name, model_name, "full_response", output)

        if output:
            valid_json = is_valid_json(output)
            if not valid_json:
                _, output = next(extract_and_load_json_iter(output), (None, ""))
            extractable_json = output != ""
        else:
            valid_json = None
            extractable_json = None

        self.add(runspec_name, model_name, "valid_json", valid_json)
        self.add(runspec_name, model_name, "extractable_json", extractable_json)
        self.add(runspec_name, model_name, "extra_text_around_json", not valid_json and extractable_json)
        self.add(runspec_name, model_name, "minified_json", is_minified(output) if extractable_json else None)
        self.add(
            runspec_name,
            model_name,
            "minified_json_non_strict",
            is_minified(output, allow_spaces_in_separators=True) if extractable_json else None,
        )
        self.add(
            runspec_name,
            model_name,
            "compacted_json",
            is_minified(output, allow_spaces_in_separators=True, allow_newlines=True) if extractable_json else None,
        )
