import logging

import numpy as np
from tqdm import tqdm

from preprocessors.base import Preprocessor

logger = logging.getLogger(__name__)


class GeneralPreprocessor(Preprocessor):
    """Preprocessor for Audio benchmarks from "AudioLLMs" and more on HF."""

    # Using the extract_audio_info method from the base class

    def process(self, dataset: dict, num_samples: int = None, properties: dict = None) -> list[dict]:
        """Process the dataset and flatten audio/context structure (expects dict-of-lists).
        
        Args:
            dataset: Dictionary containing audio data
            properties: Optional dict of properties, may include 'length_filter' tuple (min_seconds, max_seconds)
                       to filter samples by audio length.
        """
        logger.info("In [GeneralPreprocessor] Processing dataset...")

        # Extract common properties using base class method
        props = self.extract_properties(properties)
        user_prompt_add_ons = props["user_prompt_add_ons"]
        system_prompts = props["system_prompts"]
        length_filter = props["length_filter"]
        modality = props.get("dataset_info", {}).get("modality", "audio")
        audio_column_name = props.get("dataset_info", {}).get("audio_column", None)
        target_column_name = props.get("dataset_info", {}).get("target_column", None)
        user_instruction_column_name = props.get("dataset_info", {}).get("additional_instruction_column", None)
        user_query_column_name = props.get("dataset_info", {}).get("textual_input_column", None)

        # Load prompt add-ons and system prompts using base class method
        prompt_add_ons = self.load_yaml_file("prompt_add_ons.yaml")
        system_prompts_mapping = self.load_yaml_file("system_prompts.yaml")

        # Get dataset keys and size
        keys = list(dataset.keys())
        dataset_size = len(dataset[keys[0]]) if keys else 0
        self.log_dataset_info(keys, dataset_size)

        total_duration = 0
        new_dataset = []
        dataset_size = len(dataset[keys[0]]) if keys else 0
        indices = range(dataset_size if num_samples is None else min(dataset_size, num_samples))

        for i in tqdm(indices, desc="Processing samples"):
            instruction = ""
            # Create record by accessing each feature by index
            record = {k: dataset[k][i] for k in keys}

            # Extract audio information
            self.extract_audio_info(record, audio_column_name=audio_column_name)

            if modality == "text":
                record["array"] = np.array([])  # Placeholder, not used in text-only evals
                record["sampling_rate"] = 16000
                instruction = record.get(user_query_column_name, "")

            # Calculate audio duration in seconds
            audio_duration = len(record["array"]) / record["sampling_rate"]
            total_duration += audio_duration

            # Apply length filtering if specified
            if not self.check_audio_length(record["array"], record["sampling_rate"], length_filter):
                continue

            if target_column_name and target_column_name in record:
                record["model_target"] = record.get(target_column_name, None)
            else:
                possible_keys = ["reference", "answer", "text", "transcription", "sentence", "transcript",
                                 "normalized_text"]
                record["model_target"] = next((record[k] for k in possible_keys if k in record), None)

            if record["model_target"] is None:
                raise ValueError("No valid target key found in record")

            if user_instruction_column_name and user_instruction_column_name in record:
                instruction = record.get(user_instruction_column_name, "")
            else:
                instruction = record.get("instruction") or record.get("question") or ""
            # Append any user-specified prompt add-ons
            instruction += " " + " ".join(prompt_add_ons[k] for k in user_prompt_add_ons if k in prompt_add_ons)
            record["instruction"] = instruction.strip()

            # Process system prompts
            system_prompt_text = "\n\n".join(
                system_prompts_mapping[k] for k in system_prompts if k in system_prompts_mapping)
            if system_prompt_text:
                record["system_prompt"] = system_prompt_text

            record["judge_type"] = properties.get("judge_type", "detailed")
            new_dataset.append(record)

        self.log_dataset_info(keys, dataset_size, len(new_dataset), total_duration)
        return new_dataset
