import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import yaml
from scipy.signal import resample

logger = logging.getLogger(__name__)

class Preprocessor():
    def process(self, dataset: Dict[str, List[Any]], num_samples: Optional[int] = None, properties: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Base implementation of the process method to be overridden by subclasses.
        
        Args:
            dataset: Dictionary containing data
            num_samples: Number of samples to extract from the dataset
            properties: Optional dict of properties for preprocessing configuration
            
        Returns:
            List of dictionaries where each dictionary represents a processed sample
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
        
    def extract_properties(self, properties=None):
        """
        Extract common properties from the properties dictionary with default values.
        
        Args:
            properties (dict, optional): Dictionary containing properties for preprocessing.
                                         Defaults to None (empty dict).
                                         
        Returns:
            dict: Dictionary containing extracted properties with defaults applied.
        """
        if properties is None:
            properties = {}
            
        extracted = {
            "metric": properties.get("metric", None),
            "user_prompt_add_ons": properties.get("user_prompt_add_ons", []),
            "system_prompts": properties.get("system_prompts", []),
            "length_filter": properties.get("length_filter", None),  # Optional (min_seconds, max_seconds) tuple
            "dataset_info": properties.get("dataset_info", {}),
            "modality": properties.get("modality", ""),
            "judge_type": properties.get("judge_type", "")
        }
        
        logger.debug(f"Extracted properties: {extracted}")
        return extracted

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
        
    def log_dataset_info(self, dataset_keys, original_size, processed_size=None, total_duration=None):
        """
        Log information about the dataset being processed.
        
        Args:
            dataset_keys (list): List of keys in the dataset
            original_size (int): Original size of the dataset
            processed_size (int, optional): Size of processed dataset
            total_duration (float, optional): Total audio duration in seconds
        """
        logger.info(f"Dataset keys: {dataset_keys}, total samples: {original_size}")
        
        if processed_size is not None:
            logger.info(f"Processed dataset size: {processed_size}")
            
        if total_duration is not None:
            logger.info(f"Dataset is {total_duration / 3600:.2f} hours long")
