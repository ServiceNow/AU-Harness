import logging
from preprocessors.base import Preprocessor

logger = logging.getLogger(__name__)


class IfevalAudioPreprocessor(Preprocessor):
    """Preprocessor for Audio benchmarks from AudioBench on HF."""

    # Using the extract_audio_info method from the base class

    def process(self, dataset: dict,
                num_samples: int = None,
                properties: dict = None,
                ) -> list[dict]:
        """Process the dataset and flatten audio/context structure (expects dict-of-lists).

        Args:
            dataset: Dictionary containing audio data
            num_samples: Number of samples to extract from the dataset
            properties: Optional dict of properties, may include 'length_filter' tuple (min_seconds, max_seconds)
                       to filter samples by audio length.
        """
        logger.info("In [IfEvalPreprocessor] Processing dataset...")
        props = self.extract_properties(properties)
        length_filter = props["length_filter"]

        # Get dataset keys and size
        keys = list(dataset.keys())
        dataset_size = len(dataset[keys[0]]) if keys else 0
        self.log_dataset_info(keys, dataset_size)

        # Convert from columnar to row-wise format
        row_data = self.columnar_to_row_wise(dataset)
        
        total_duration = 0
        new_dataset = []
        
        for record in row_data:
            # Extract audio information
            self.extract_audio_info(record)

            # Calculate audio duration in seconds
            audio_duration = len(record["array"]) / record["sampling_rate"]
            total_duration += audio_duration

            # Apply length filtering if specified
            if not self.check_audio_length(record["array"], record["sampling_rate"], length_filter):
                continue

            instruction = {"instruction_id_list": record["instruction_id_list"],
                           "kwargs": record["kwargs"]}
            record["instruction"] = ''
            record["supporting_instructions"] = instruction
            new_dataset.append(record)
            
        self.log_dataset_info(keys, dataset_size, len(new_dataset), total_duration)
        return new_dataset
