import logging

import numpy as np
from tqdm import tqdm

from preprocessors.base import Preprocessor

logger = logging.getLogger(__name__)


class MultiturnPreprocessor(Preprocessor):
    """Preprocessor for Audio benchmarks from "AudioLLMs" and more on HF."""

    # Using the extract_audio_info method from the base class

    def process(self, dataset: dict, num_samples: int = None, properties: dict = None) -> list[dict]:
        """Process the dataset and flatten audio/context structure (expects dict-of-lists).

        Args:
            dataset: Dictionary containing audio data
            properties: Optional dict of properties, may include 'length_filter' tuple (min_seconds, max_seconds)
                       to filter samples by audio length.
        """

        # Extract common properties using base class method
        props = self.extract_properties(properties)
        length_filter = props["length_filter"]
        user_prompt_add_ons = props["user_prompt_add_ons"]
        modality = props.get("dataset_info", {}).get("modality", "audio")
        audio_column_name = props.get("dataset_info", {}).get("audio_column", None)
        target_column_name = props.get("dataset_info", {}).get("target_column", None)
        user_query_column_name = props.get("dataset_info", {}).get("textual_input_column", None)
        id_column = props.get("dataset_info", {}).get("id_column", None)

        if user_query_column_name is None:
            raise ValueError("user_query_column_name must be defined")

        # Get dataset keys and size
        keys = list(dataset.keys())
        if id_column is None or id_column not in keys:
            ValueError("Dataset must contain a column with id, pass it via dataset_info using \"id_column\"")
        dataset_size = len(dataset[keys[0]]) if keys else 0
        self.log_dataset_info(keys, dataset_size)

        new_dataset = []
        dataset_size = len(dataset[keys[0]]) if keys else 0

        # keys = list(dataset[0].keys())
        # dataset_size = len(dataset) if keys else 0
        # self.log_dataset_info(keys, dataset_size)
        #
        # new_dataset = []
        # dataset_size = len(dataset) if keys else 0

        indices = range(dataset_size if num_samples is None else min(dataset_size, num_samples))
        for i in tqdm(indices, desc="Processing samples"):
            instruction = []
            # Create record by accessing each feature by index
            record = {k: dataset[k][i] for k in keys}

            textual_inputs = record.get(user_query_column_name)
            if not textual_inputs:
                raise ValueError("Instruction need to be present as text form for judge")

            if modality == "text":
                audio_data = [{
                    "array": np.array([]),  # Placeholder, not used in text-only evals
                    "sampling_rate": 16000
                }]
                instruction = textual_inputs
            else:
                audio_data = dataset[audio_column_name][i]
                if isinstance(audio_data, list) and len(audio_data) == 2:
                    if isinstance(audio_data[0], dict) and isinstance(audio_data[1], dict):
                        pass
                    elif len(audio_data[0])==1 and len(audio_data[1])==1:
                        logger.warning("Multiple audios for one instruction not supported yet.")
                        continue
                elif isinstance(audio_data, dict):
                    audio_data = [audio_data]

                # Check if ALL audios pass the length filter
                if not all(
                        self.check_audio_length(audio["array"], audio["sampling_rate"], length_filter)
                        for audio in audio_data if len(audio["array"]) > 0
                ):
                    logger.warning(
                        f"Audio in item {i} filtered because of length "
                        f"Supported: {length_filter[0]}–{length_filter[1]} sec."
                    )
                    continue


            audio_arrays = [i["array"] for i in audio_data]
            sampling_rate = audio_data[0]["sampling_rate"]

            record['id'] = record.get(id_column)
            record['model_target'] = record.get(target_column_name, None)
            record['array'] = audio_arrays
            record['sampling_rate'] = sampling_rate
            record['instruction'] = instruction
            record['textual_input'] = textual_inputs
            record['process_turn'] = 0
            record['total_turns'] = max(len(audio_data), len(instruction))
            record['is_multiturn'] = True
            new_dataset.append(record)

        return new_dataset
