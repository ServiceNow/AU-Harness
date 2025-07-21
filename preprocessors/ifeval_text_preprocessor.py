import logging

logger = logging.getLogger(__name__)
from tqdm import tqdm
from preprocessors.ifeval_audio_preprocessor import IfevalAudioPreprocessor
from preprocessors.base import Preprocessor



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
        ifeval_audio_processor = IfevalAudioPreprocessor()
        dataset = ifeval_audio_processor.process(dataset, num_samples, properties)
        new_dataset = []
        for i in range(len(dataset)):
            dataset[i]['array'] = None
            dataset[i]['instruction'] = dataset[i]['prompt']
            new_dataset.append(dataset[i])
        return new_dataset
