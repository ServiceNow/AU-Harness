import logging

from tqdm import tqdm

from preprocessors.base import Preprocessor

logger = logging.getLogger(__name__)
from utils.constants import language_map  # Import language_map from constants


class Covost2Preprocessor(Preprocessor):
    """Preprocessor for Covost2 dataset from fixie-ai/covost2."""

    def process(self, dataset: dict, num_samples: int = None, properties: dict = None) -> list[dict]:
        """Process the dataset and flatten audio/context structure (expects dict-of-lists).
        
        Args:
            dataset: Dictionary containing audio data
            properties: Optional dict of properties, may include 'length_filter' tuple (min_seconds, max_seconds)
                       to filter samples by audio length.
        """
        logger.info("In [Covost2Preprocessor] Processing dataset...")

        props = self.extract_properties(properties)

        # Get dataset info
        dataset_keys = list(dataset.keys())
        dataset_size = len(dataset.get("id", []))
        self.log_dataset_info(dataset_keys, dataset_size)

        processed_data = []
        dataset_size = len(dataset.get("id", []))
        indices = range(dataset_size if num_samples is None else min(dataset_size, num_samples))

        for i in tqdm(indices, desc="Processing samples"):
            sample_id = dataset["id"][i]
            translation = dataset["translation"][i]
            source_sentence = dataset["sentence"][i]
            audio = dataset["audio"][i]
            audio_array = audio["array"]
            sampling_rate = audio["sampling_rate"]

            if sampling_rate is None:
                logger.warning(f"[{sample_id}] Sampling rate missing. Assuming 16kHz.")
                sampling_rate = 16000

            # Resample if needed using base class method
            audio_array, sampling_rate = self.resample_audio(audio_array, sampling_rate)

            # Get the target language from dataset_info
            try:
                target_language_code = props["dataset_info"].get("target_language")
                # Convert language code to full language name using the language_map
                target_language_name = language_map.get(target_language_code, target_language_code)
                target_language_name = target_language_name.capitalize()
            except KeyError:
                raise ValueError("Target language not found. Please specify target_language in dataset config")

            instruction = f"Please translate the given speech to {target_language_name}. Return ONLY the translated speech in text format without any other prefix text."

            # Create structured sample
            sample = {
                "id": sample_id,
                "array": audio_array,
                "sampling_rate": sampling_rate,
                "model_target": translation.strip(),
                "source_sentence": source_sentence.strip(),
                "instruction": instruction,
            }
            processed_data.append(sample)

        self.log_dataset_info(dataset_keys, dataset_size, len(processed_data))
        return processed_data
