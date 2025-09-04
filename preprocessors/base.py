
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datasets import Dataset
import yaml
from scipy.signal import resample

logger = logging.getLogger(__name__)


class Preprocessor():
    """Base preprocessor class for handling audio and text data processing.
    
    This class provides common utilities for preprocessing datasets including audio resampling,
    length filtering, property extraction, and prompt management. It serves as a foundation
    for task-specific preprocessors which should override the `process` method.
    
    The preprocessor handles standardization of audio formats, dataset information logging,
    and management of task-specific prompt add-ons.
    """

    def process(self, dataset: Dataset, task_config: Dict[str, Any], 
                run_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Base implementation of the process method to be overridden by subclasses.
        
        Args:
            dataset: The task dataset to pre-process
            task_config: Dictionary containing task configuration parameters
            run_config: Dictionary containing run configuration parameters
            
        Returns:
            List of dictionaries where each dictionary represents a pre-processed sample
        """
        raise NotImplementedError

    def load_yaml_file(self, file_name):
        """
        Load a YAML file from the prompts directory.
        
        Args:
            file_name (str): Name of the YAML file in the prompts directory
            
        Returns:
            dict: Contents of the YAML file, or empty dict if file not found
        """
        yaml_path = Path(__file__).resolve().parent.parent / "prompts" / file_name
        try:
            with open(yaml_path, "r") as f:
                content = yaml.safe_load(f) or {}
            return content
        except FileNotFoundError:
            logger.warning(f"File not found at {yaml_path}. Returning empty dictionary.")
            return {}

    def extract_audio_info(self, record, audio_column_name=None):
        """
        Extract audio information from a record, standardizing the format.
        
        Args:
            record (dict): Dictionary containing record data with audio information
            
        Returns:
            None: Modifies the record in-place
            
        Raises:
            KeyError: If neither 'audio' nor 'context' keys are found in the record
        """
        if audio_column_name is not None and audio_column_name in record:
            record["array"] = record[audio_column_name]["array"]
            record["sampling_rate"] = record[audio_column_name]["sampling_rate"]
            record.pop(audio_column_name)
        elif "audio" in record:
            record["array"] = record["audio"]["array"]
            record["sampling_rate"] = record["audio"]["sampling_rate"]
            record.pop("audio")
        elif "context" in record:
            record["array"] = record["context"]["array"]
            record["sampling_rate"] = record["context"]["sampling_rate"]
            record.pop("context")
        else:
            raise KeyError(
                "Neither 'audio' nor 'context' keys found in data, try passing audio column name via runspec using key \"audio_column\"")

    def resample_audio(self, audio_array, source_sr, target_sr=16000):
        """
        Resample audio array to target sampling rate.
        
        Args:
            audio_array (np.ndarray): Audio data as numpy array
            source_sr (int): Source sampling rate
            target_sr (int): Target sampling rate, defaults to 16000
            
        Returns:
            np.ndarray: Resampled audio array
        """
        if source_sr != target_sr:
            target_length = int(target_sr * len(audio_array) / source_sr)
            return resample(audio_array, target_length), target_sr
        return audio_array, source_sr

    def check_audio_length(self, audio_array, sampling_rate, length_filter=None):
        """
        Check if audio duration meets the specified length filter.
        
        Args:
            audio_array (np.ndarray): Audio data as numpy array
            sampling_rate (int): Sampling rate of the audio
            length_filter (tuple, optional): Tuple of (min_seconds, max_seconds)
            
        Returns:
            bool: True if audio passes length filter or if length_filter is None, False otherwise
        """
        if length_filter is None:
            return True

        if isinstance(length_filter, tuple) and len(length_filter) == 2:
            min_length, max_length = length_filter
            audio_duration = len(audio_array) / sampling_rate
            return min_length <= audio_duration <= max_length

        return True
    
    def get_dataset_filters(self, filters: dict, dataset_size: int):
        """
        Process and validate dataset filters.
        
        Args:
            filters (dict): Dictionary containing filter settings with keys like 'length' and 'num_samples'.
            dataset_size (int): The total size of the dataset being filtered.
            
        Returns:
            tuple: A tuple containing (length_filter, num_samples_filter) where:
                - length_filter: Tuple of (min_length, max_length) in seconds or None if not applicable
                - num_samples_filter: Number of samples to include or None if all samples should be included
        """
        if filters is None:
            return None, None
        
        length_filter = filters.get('length', None)
        num_samples_filter = filters.get('num_samples', None)

        if length_filter and not isinstance(length_filter, tuple) and not (len(length_filter) == 2) and not (length_filter[1] > length_filter[0]):
            logger.warning("Length filter must be a tuple of (min_seconds, max_seconds)")
            length_filter = None

        if num_samples_filter and not isinstance(num_samples_filter, int) and not (num_samples_filter < dataset_size):
            logger.warning("Num samples filter must be an integer and less that dataset size")
            num_samples_filter = None

        return length_filter, num_samples_filter
        

    def log_dataset_info(self, dataset_keys, original_size, processed_size=None, total_duration=None):
        """
        Log information about the dataset being processed.
        
        Args:
            dataset_keys (list): List of keys in the dataset
            original_size (int): Original size of the dataset
            processed_size (int, optional): Size of processed dataset
            total_duration (float, optional): Total audio duration in seconds
        """
        if processed_size is not None:
            logger.info(f"Processed dataset size: {processed_size}")

        if total_duration is not None:
            logger.info(f"Dataset is {total_duration / 3600:.2f} hours long")