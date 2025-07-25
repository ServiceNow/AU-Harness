import logging
from typing import Dict, List, Optional, Any
import numpy as np
from scipy.signal import resample
from tqdm import tqdm

logger = logging.getLogger(__name__)

class VoiceBenchIfevalPreprocessor:
    """
    A preprocessor for the VoiceBench IFEval dataset, designed for
    Speech Query Question Answering (SQQA) tasks.
    """

    def process(
        self, 
        dataset: Dict[str, List[Any]], 
        num_samples: Optional[int] = None, 
        properties: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Process the VoiceBench IFEval dataset.

        Args:
            dataset: Dictionary containing audio data
            num_samples: Optional number of samples to process
            properties: Optional dict of properties

        Returns:
            A list of dictionaries where each dictionary represents a sample
        """
        
        logger.info("In [VoiceBenchIfevalPreprocessor] Processing dataset...")

        modality = properties.get("modality", "audio")
        logger.info(f"Processing modality: {modality}")

        dataset_keys = list(dataset.keys())
        dataset_size = len(dataset.get("key", []))
        logger.info(f"Dataset keys: {dataset_keys}, total samples: {dataset_size}")

        processed_data: List[Dict[str, Any]] = []

        for i in tqdm(range(dataset_size), desc="Processing samples"):
            key = dataset["key"][i]

            if modality == "text":
                audio_data = {
                    "array": np.array([]),  # Placeholder, not used in text-only evals
                    "sampling_rate": 16000
                }
            else:
                audio_data = dataset["audio"][i]

            prompt = dataset["prompt"][i]
            instruction_id_list = dataset["instruction_id_list"][i]
            kwargs = dataset["kwargs"][i]

            if modality == "audio":
                # Validate audio data structure
                if not isinstance(audio_data, dict):
                    logger.warning(f"[{sample_id}] Invalid audio format. Skipping sample.")
                    continue

                # Convert to NumPy array
                audio_array = np.array(audio_data.get("array"))
                sr = audio_data.get("sampling_rate")

                if sr is None:
                    logger.warning(f"[{sample_id}] Sampling rate missing. Assuming 16kHz.")
                    sr = 16000

                # Resample if needed
                if sr != 16000:
                    target_length = int(16000 * len(audio_array) / sr)
                    audio_array = resample(audio_array, target_length)
                    sr = 16000

            # Ensure prompt exists
            if not prompt:
                logger.warning(f"[{sample_id}] Missing prompt. Skipping sample.")
                continue

            if modality == "text":
                instruction = prompt
            else:
                # For audio modality, we can define a generic instruction
                instruction = f"Answer the question provided in the audio."

            # Create structured sample
            sample = {
                "id": key,
                "array": audio_array if modality == "audio" else audio_data["array"],
                "sampling_rate": sr if modality == "audio" else audio_data["sampling_rate"],
                "audio_content_in_text": prompt,
                "instruction": instruction,
                "instruction_id_list": instruction_id_list,
                "kwargs": kwargs,
                "model_target": (instruction_id_list, kwargs, prompt),
            }

            processed_data.append(sample)

        logger.info(f"Processed dataset size: {len(processed_data)}")
        return processed_data
