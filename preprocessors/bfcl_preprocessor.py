import ast
import json
import logging
import re
from typing import Dict, List, Optional, Any

import numpy as np
from tqdm import tqdm

from preprocessors.base import Preprocessor

logger = logging.getLogger(__name__)


class BfclPreprocessor(Preprocessor):
    """
    A preprocessor for the BFCL dataset, designed for
    instruction following evaluation of audio LLMs.
    """

    def process(
            self,
            dataset: Dict[str, List[Any]],
            num_samples: Optional[int] = None,
            properties: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Process the BFCL dataset.

        Args:
            dataset: Dictionary containing audio data
            num_samples: Optional number of samples to process
            properties: Optional dict of properties

        Returns:
            A list of dictionaries where each dictionary represents a sample
        """

        logger.info("In [BFCLPreprocessor] Processing dataset...")

        # Extract properties using the base class method
        props = self.extract_properties(properties)
        modality = props.get("dataset_info", {}).get("modality", "audio")
        prompt_mode = props.get("dataset_info", {}).get("with_prompt", False)

        # Get dataset info using base class method
        dataset_keys = list(dataset.keys())
        dataset_size = len(dataset.get("id", []))
        self.log_dataset_info(dataset_keys, dataset_size)
        SYSTEM_PROMPT = "You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.\nIf none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.\nYou should only return the function calls in your response.\n\nIf you decide to invoke any of the function(s), you MUST put it in the format of \n\n```json\n[{func_name1:{params_name1=params_value1, params_name2=params_value2...}}, {func_name2:{params_name1=params_value1, params_name2=params_value2, params_name3=params_value3...}}]\n``` \n\nYou SHOULD NOT include any other text in the response. If no relevant function matches then return empty list in json like ```json\n [] \n ```. Make sure an appropriate json is always there in the response. \n\nAt each turn, you should try your best to complete the tasks requested by the user within the current turn. Continue to output functions to call until you have fulfilled the user's request to the best of your ability. Once you have no more functions to call, the system will consider the current turn complete and proceed to the next turn or task.\n\n"
        processed_data = []
        dataset_size = len(dataset.get("id", []))
        indices = range(dataset_size if num_samples is None else min(dataset_size, num_samples))
        system_prompt = ''
        for i in tqdm(indices, desc="Processing samples"):
            id = dataset["id"][i]

            if modality == "text":
                audio_data = {
                    "array": np.array([]),  # Placeholder, not used in text-only evals
                    "sampling_rate": 16000
                }
            else:
                audio_data = dataset["audio"][i]
                if isinstance(audio_data, list) and len(audio_data) == 1:
                    audio_data = audio_data[0]
                elif isinstance(audio_data, list) and len(audio_data) > 1:
                    logger.warning(f"[{id}] Support single audio only right now!")
                    continue
                elif isinstance(audio_data, dict):
                    audio_data = audio_data

            prompt = dataset["question"][i]
            if isinstance(prompt, str):
                prompt = ast.literal_eval(prompt)

            if len(prompt) == 1:  # For now handle single turn only
                prompt = prompt[0]
            else:
                logger.warning(f"[{id}] Support only single turn")
                continue
            function = dataset["tools"][i]
            reference = dataset["reference"][i]

            if isinstance(reference, str):
                reference = json.loads(reference)
            if isinstance(function, str):
                function = json.loads(function)

            # Lot of models don't support dot(.) in function name so to handle that we replace it with _ for easy evals
            new_functions = []
            for item in function:
                item["name"] = re.sub(r"\.", "_", item["name"])
                new_functions.append(item)
            function = new_functions

            new_references = []
            for item in reference:
                func_name = list(item.keys())[0]
                new_func_name = re.sub(r"\.", "_", func_name)
                new_references.append({new_func_name: item[func_name]})
            reference = new_references

            required_fields = [tool['parameters']['required'] for tool in function]

            # Validate audio data structure
            if not isinstance(audio_data, dict):
                logger.warning(f"[{id}] Invalid audio format. Skipping sample.")
                continue

            # Convert to NumPy array
            audio_array = np.array(audio_data.get("array"))
            sr = audio_data.get("sampling_rate")

            if modality == "audio":
                if sr is None:
                    logger.warning(f"[{id}] Sampling rate missing. Assuming 16kHz.")
                    sr = 16000

                # Use base class method to resample audio
                audio_array, sr = self.resample_audio(audio_array, sr)

            # Ensure prompt exists
            if not prompt:
                logger.warning(f"[{id}] Missing prompt. Skipping sample.")
                continue

            if modality == "text":
                instruction = prompt
            else:
                # For audio modality, we can define a generic instruction
                instruction = f""

            # In prompt mode we get tool calls from llm response,
            # This doesn't evals the model on tool calling via capability sense,
            # but still allow to check ability to choose function
            if prompt_mode:
                system_prompt = SYSTEM_PROMPT
                system_prompt += 'Here is a list of functions in JSON format that you can invoke.\n{functions}\n'.format(
                    functions=json.dumps(function, indent=4))
                function = None

            # Create structured sample
            sample = {
                "id": id,
                "array": audio_array,
                "sampling_rate": sr,
                "audio_content_in_text": prompt,
                "system_prompt": system_prompt,
                "instruction": instruction,
                "tools": function,
                "reference": reference,
                "model_target": [reference, required_fields],
            }

            processed_data.append(sample)

        self.log_dataset_info(dataset_keys, dataset_size, len(processed_data))
        return processed_data
