import logging
from typing import Dict, List, Any

from datasets import Dataset

from tqdm import tqdm
from typing import Dict, List, Any

import numpy as np

from datasets import Dataset
from preprocessors.base import Preprocessor

logger = logging.getLogger(__name__)


class MultiturnPreprocessor(Preprocessor):
    """Preprocessor for Audio benchmarks from "AudioLLMs" and more on HF."""

    def process(self, dataset: Dataset, task_config: Dict[str, Any], 
                run_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run pre-processing on MT-Bench type of datasets.
        
        Args:
            dataset: The task dataset to pre-process
            task_config: Dictionary containing task configuration parameters
            run_config: Dictionary containing run configuration parameters
            
        Returns:
            List of dictionaries where each dictionary represents a pre-processed sample
        """

        # Extract common properties from task_config
        modality = task_config.get('modality', 'audio')
        audio_column_name = task_config.get('audio_column', None)
        target_column_name = task_config.get('target_column', None)
        user_query_column_name = task_config.get('textual_input_column', None)
        id_column = task_config.get('id_column', None)

        # Obtain task-specific prompt (if provided)
        user_prompt = task_config.get('user_prompt', '')

        if user_query_column_name is None:
            raise ValueError("user_query_column_name must be defined")
        
        # Get dataset info
        dataset_keys = list(dataset.features.keys())
        if id_column is None or id_column not in dataset_keys:
            raise ValueError("Dataset must contain a column with id, pass it via dataset_info using \"id_column\"")
        dataset_size = len(dataset)
        self.log_dataset_info(dataset_keys, dataset_size)

        # Get dataset filters
        length_filter, num_samples_filter = self.get_dataset_filters(run_config.get('filter', None), dataset_size)

        # Preparing aggregating information
        indices = range(dataset_size)
        new_dataset = []
        sample_count = 0
        total_duration = 0

        for i in tqdm(indices, desc="Processing samples"):
            instruction = []
            # Create record by accessing each feature by index
            record = {k: dataset[k][i] for k in dataset_keys}

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
                audio_data = row[audio_column_name]
                if isinstance(audio_data, list) and len(audio_data) == 2:
                    if isinstance(audio_data[0], dict) and isinstance(audio_data[1], dict):
                        pass
                    elif len(audio_data[0])==1 and len(audio_data[1])==1:
                        logger.warning("Multiple audios for one instruction not supported yet.")
                        continue
                elif isinstance(audio_data, dict):
                    audio_data = [audio_data]

                # Check if ALL audios pass the length filter
                if (length_filter):
                    def _check_multiple_audios_for_length_filter(audio_data, length_filter):
                        keep_sample = True
                        audio_duration = 0.0
                        for audio in audio_data:
                            if (len(audio['array']) <=0):
                                keep_sample = False
                            else:
                                keep_sample = keep_sample and self.check_audio_length(audio['array'], audio['sampling_rate'], length_filter)
                                audio_duration += len(audio['array']) / audio['sampling_rate']
                        return keep_sample, audio_duration
                    
                    keep_sample, audio_duration = _check_multiple_audios_for_length_filter(audio_data, length_filter)
                    if not (keep_sample):
                        logger.warning(
                            f"Audio in item {i} filtered because of length "
                            f"Supported: {length_filter[0]}–{length_filter[1]} sec."
                        )
                        continue
                    
                    total_duration += audio_duration
                else:
                    audio_duration = [len(audio['array'])/ audio['sampling_rate'] for audio in audio_data if len(audio['array']> 0)]
                    total_duration += sum(audio_duration)

            # Stop processing if num_samples filtering is set and more than num_samples_filter samples are processed
            if (num_samples_filter and sample_count >= num_samples_filter):
                break

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
            sample_count += 1
        
        self.log_dataset_info(dataset_keys, dataset_size, sample_count, total_duration)
        return new_dataset