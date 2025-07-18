import logging
logger = logging.getLogger(__name__)
from tqdm import tqdm
from pathlib import Path
import yaml
from preprocessors.base import Preprocessor

class AudiobenchPreprocessor(Preprocessor):
    """Preprocessor for Audio benchmarks from AudioBench on HF."""

    def process(self, dataset: dict, num_samples: int = None, properties: dict = None) -> list[dict]:
        """Process the dataset and flatten audio/context structure (expects dict-of-lists).
        
        Args:
            dataset: Dictionary containing audio data
            properties: Optional dict of properties, may include 'length_filter' tuple (min_seconds, max_seconds)
                       to filter samples by audio length.
        """
        logger.info("In [AudiobenchPreprocessor] Processing dataset...")
        #logger.info(dataset)
        user_prompt_add_ons = properties.get("user_prompt_add_ons", [])
        system_prompts = properties.get("system_prompts", [])
        length_filter = properties.get("length_filter", None)  # Optional (min_seconds, max_seconds) tuple
        
        # Load prompt add-ons mapping
        prompt_yaml_path = Path(__file__).resolve().parent.parent / "prompts" / "prompt_add_ons.yaml"
        try:
            with open(prompt_yaml_path, "r") as f:
                prompt_add_ons = yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning(f"Prompt add-ons file not found at {prompt_yaml_path}. Proceeding without add-ons.")
            prompt_add_ons = {}
            
        # Load system prompts mapping
        system_prompts_path = Path(__file__).resolve().parent.parent / "prompts" / "system_prompts.yaml"
        try:
            with open(system_prompts_path, "r") as f:
                system_prompts_mapping = yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning(f"System prompts file not found at {system_prompts_path}. Proceeding without system prompts.")
            system_prompts_mapping = {}

        total_duration = 0
        new_dataset = []
        keys = list(dataset.keys())
        num_samples = len(dataset[keys[0]]) if keys else 0
        #logger.info(f"Dataset keys: {keys}, num_samples: {num_samples}")
        for i in tqdm(range(num_samples), desc="Preprocessing"):
            record = {k: dataset[k][i] for k in keys}
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

            # Calculate audio duration in seconds
            audio_duration = len(record["array"]) / record["sampling_rate"]
            total_duration += audio_duration
            
            # Apply length filtering if specified
            if length_filter and isinstance(length_filter, tuple) and len(length_filter) == 2:
                min_length, max_length = length_filter
                if audio_duration < min_length or audio_duration > max_length:
                    continue
            
            possible_keys = ["reference", "answer", "text", "transcription", "sentence", "transcript", "normalized_text"]
            for key in possible_keys:
                if key in record:
                    record["model_target"] = record[key]
                    break
            if "model_target" not in record:
                raise ValueError("No valid target key found in record")

            instruction = record.get("instruction") or record.get("question") or ""
            # Append any user-specified prompt add-ons
            for k in user_prompt_add_ons:
                add_on = prompt_add_ons.get(k)
                if add_on:
                    instruction = f"{instruction} {add_on}"
            record["instruction"] = instruction
            
            # Process system prompts
            system_prompt_text = ""
            for k in system_prompts:
                prompt = system_prompts_mapping.get(k)
                if prompt:
                    if system_prompt_text:
                        system_prompt_text += "\n\n" + prompt
                    else:
                        system_prompt_text = prompt
            if system_prompt_text:
                record["system_prompt"] = system_prompt_text
                
            record["judge_type"] = properties.get("judge_type", "detailed")
            new_dataset.append(record)

        logger.info(f"Dataset is {total_duration / 3600:.2f} hours long")
        #print("DEBUG: Flattened record keys:", new_dataset[0].keys())
        return new_dataset
