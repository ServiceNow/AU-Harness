import logging
logger = logging.getLogger(__name__)
from tqdm import tqdm
from pathlib import Path
from scipy.signal import resample
import yaml
from preprocessors.base import Preprocessor
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

        dataset_keys = list(dataset.keys())
        dataset_size = len(dataset.get("id", []))
        logger.info(f"Dataset keys: {dataset_keys}, total samples: {dataset_size}")

        target_language = properties.get("target_language", "en")
        processed_data: List[Dict[str, Any]] = []

        for idx in tqdm(range(dataset_size), desc="Processing samples"):
            sample_id = dataset["id"][idx]
            translation = dataset["translation"][idx]
            audio = dataset["audio"][idx]
            audio_array = audio["array"]
            sampling_rate = audio["sampling_rate"]

            if sampling_rate is None:
                logger.warning(f"[{sample_id}] Sampling rate missing. Assuming 16kHz.")
                sampling_rate = 16000

            # Resample if needed
            if sampling_rate != 16000:
                target_length = int(16000 * len(audio_array) / sampling_rate)
                audio_array = resample(audio_array, target_length)
                sampling_rate = 16000
            
            # Get the target language from dataset_info
            try:
                target_language_code = properties["dataset_info"].get("target_language")
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
                "instruction": instruction,
            }
            processed_data.append(sample)

        logger.info(f"Processed dataset size: {len(processed_data)}")
        return processed_data