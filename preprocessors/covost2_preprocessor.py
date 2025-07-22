import logging
from preprocessors.base import Preprocessor

logger = logging.getLogger(__name__)

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

        # Extract common properties using base class method
        props = self.extract_properties(properties)
        
        # Get dataset info
        dataset_keys = list(dataset.keys())
        dataset_size = len(dataset.get("id", []))
        self.log_dataset_info(dataset_keys, dataset_size)
        
        # Set default target language
        target_language = "en"
        
        # Convert from columnar to row-wise format
        row_data = self.columnar_to_row_wise(dataset)
        processed_data = []
        
        for record in row_data:
            sample_id = record["id"]
            translation = record["translation"]
            audio = record["audio"]
            audio_array = audio["array"]
            sampling_rate = audio["sampling_rate"]

            if sampling_rate is None:
                logger.warning(f"[{sample_id}] Sampling rate missing. Assuming 16kHz.")
                sampling_rate = 16000

            # Resample if needed using base class method
            audio_array, sampling_rate = self.resample_audio(audio_array, sampling_rate)
            
            # Get the target language from dataset_info
            try:
                dataset_info = props["dataset_info"]
                if dataset_info and "target_language" in dataset_info:
                    target_language = dataset_info["target_language"]
            except KeyError:
                pass  # Keep default language if not found

            instruction = f"Please translate the given speech to {target_language}. Return ONLY the translated speech in text format without any other prefix text."
            
            # Create structured sample
            sample = {
                "id": sample_id,
                "array": audio_array,
                "sampling_rate": sampling_rate,
                "model_target": translation.strip(),
                "instruction": instruction,
            }
            processed_data.append(sample)

        self.log_dataset_info(dataset_keys, dataset_size, len(processed_data))
        return processed_data