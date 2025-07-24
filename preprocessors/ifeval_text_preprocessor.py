import logging
from preprocessors.ifeval_audio_preprocessor import IfevalAudioPreprocessor
from preprocessors.base import Preprocessor
import tqdm

logger = logging.getLogger(__name__)


class IfevalTextPreprocessor(Preprocessor):


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
        # Process data using ifeval_audio_processor first
        ifeval_audio_processor = IfevalAudioPreprocessor()
        processed_dataset = ifeval_audio_processor.process(dataset, num_samples, properties)
        
        # Get dataset keys to log information
        keys = list(dataset.keys())
        dataset_size = len(dataset[keys[0]]) if keys else 0
        
        logger.info("In [IfEvalTextPreprocessor] Processing dataset...")
        
        # Process for text-only
        new_dataset = []
        for record in processed_dataset:
            record['array'] = None
            if 'prompt' in record:
                record['instruction'] = record['prompt']
            new_dataset.append(record)
            
        self.log_dataset_info(keys, dataset_size, len(new_dataset))
        return new_dataset
