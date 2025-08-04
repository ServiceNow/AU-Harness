import ast
import json
import logging
import re

from models.model_response import ModelResponse
from postprocessors.base import Postprocessor

logger = logging.getLogger(__name__)
logger.propagate = True


class BfclPostprocessor(Postprocessor):
    """
    Postprocessor for bfcl predictions.
    """

    @staticmethod
    def extract_json_from_message(message: str):
        """
        Extracts the JSON object from a message.
        """

        def fix_json_like_string(s):
            # Add double quotes to all keys
            s = re.sub(r'([{,])\s*([a-zA-Z_][\w\.]*)\s*:', r'\1"\2":', s)
            return s

        pattern = r"```json(.*?)```"
        match = re.search(pattern, message, re.DOTALL)
        if match:
            # Remove leading/trailing whitespace and parse JSON
            json_str = fix_json_like_string(match.group(1).strip())
            try:
                json_decode = json.loads(json_str)
                return json_decode
            except:
                try:
                    json_decode = ast.literal_eval(json_str)
                    return json_decode
                except:
                    return None

    def process(
            self,
            dataset: list[dict],
            predictions: ModelResponse,
            metric
    ) -> tuple[list[tuple[str, str]], dict[str, list[str]], list, list] | dict:
        """
        Process and clean model predictions and prepare target-label pairs.
        """
        logger.info("Processing predictions with VoiceBenchIfevalPostprocessor...")

        processed_predictions: dict[str, list[str]] = {}
        for model_name, preds in predictions.items():
            processed = []
            for pred, dataset_row in zip(preds, dataset):
                tool_responses = []
                if isinstance(pred.raw_response, dict):
                    tools = pred.raw_response.get('choices', [])[0]['message']['tool_calls']
                    raw_llm_response = pred.llm_response
                    raw_tool_responses = tools
                else:
                    tools = None
                    tool_responses = None
                    raw_llm_response = None
                    raw_tool_responses = None
                if dataset_row.get('tools', None) is None:
                    # We ran in prediction in prompt mode
                    tool_responses = self.extract_json_from_message(pred.llm_response.strip())
                    pred.llm_response = ''
                if tools:
                    for tool in tools:
                        tool_name = tool['function']['name']
                        tool_arguments = json.loads(tool['function']['arguments'])

                        tool_responses.append({tool_name: tool_arguments})

                processed_pred = {"llm_response": pred.llm_response.strip(),
                                  "tool_response": tool_responses,
                                  "raw_tool_response": raw_tool_responses,
                                  "raw_llm_response": raw_llm_response}
                processed.append(processed_pred)
            processed_predictions[model_name] = processed

        output = {
            "instructions": [record.get("instruction", "") for record in dataset],
            "model_targets": [record["model_target"] for record in dataset if "model_target" in record],
            "processed_predictions": processed_predictions,
        }
        self.validate_output(output)
        return output
