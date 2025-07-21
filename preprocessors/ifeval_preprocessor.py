import logging

logger = logging.getLogger(__name__)
from tqdm import tqdm
from preprocessors.base import Preprocessor


class IfevalPreprocessor(Preprocessor):
    """Preprocessor for Audio benchmarks from AudioBench on HF."""

    def extract_audio_info(self, record):
        if "audio" in record:
            record["array"] = record["audio"]["array"]
            record["sampling_rate"] = record["audio"]["sampling_rate"]
            record.pop("audio")
        elif "context" in record:
            record["array"] = record["context"]["array"]
            record["sampling_rate"] = record["context"]["sampling_rate"]
            record.pop("context")
        else:
            raise KeyError("Neither 'audio' nor 'context' keys found in data")

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


        total_duration = 0
        new_dataset = []
        keys = list(dataset.keys())
        num_samples = len(dataset[keys[0]]) if keys else 0
        # logger.info(f"Dataset keys: {keys}, num_samples: {num_samples}")
        for i in tqdm(range(num_samples), desc="Preprocessing"):
            record = {k: dataset[k][i] for k in keys}
            self.extract_audio_info(record)

            # Calculate audio duration in seconds
            audio_duration = len(record["array"]) / record["sampling_rate"]
            total_duration += audio_duration

            # Apply length filtering if specified
            if length_filter and isinstance(length_filter, tuple) and len(length_filter) == 2:
                min_length, max_length = length_filter
                if audio_duration < min_length or audio_duration > max_length:
                    continue

            instruction = {"instruction_id_list": record["instruction_id_list"],
                           "kwargs": record["kwargs"]}
            record["instruction"] = instruction
            new_dataset.append(record)
        logger.info(f"Dataset is {total_duration / 3600:.2f} hours long")
        return new_dataset
