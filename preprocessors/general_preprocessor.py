import logging
logger = logging.getLogger(__name__)
from tqdm import tqdm
from preprocessors.base import Preprocessor

class GeneralPreprocessor(Preprocessor):
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

    def process(self, dataset: dict, num_samples: int = None, properties: dict = None) -> list[dict]:
        """Process the dataset and flatten audio/context structure (expects dict-of-lists).
        
        Args:
            dataset: Dictionary containing audio data
            properties: Optional dict of properties, may include 'length_filter' tuple (min_seconds, max_seconds)
                       to filter samples by audio length.
        """
        logger.info("In [GeneralPreprocessor] Processing dataset...")
        #logger.info(dataset)
        # Extract common properties using base class method
        props = self.extract_properties(properties)
        user_prompt_add_ons = props["user_prompt_add_ons"]
        system_prompts = props["system_prompts"]
        length_filter = props["length_filter"]
        
        # Load prompt add-ons and system prompts using base class method
        prompt_add_ons = self.load_yaml_file("prompt_add_ons.yaml")
        system_prompts_mapping = self.load_yaml_file("system_prompts.yaml")

        total_duration = 0
        new_dataset = []
        keys = list(dataset.keys())
        num_samples = len(dataset[keys[0]]) if keys else 0
        #logger.info(f"Dataset keys: {keys}, num_samples: {num_samples}")
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
            
            possible_keys = ["reference", "answer", "text", "transcription", "sentence", "transcript", "normalized_text"]
            record["model_target"] = next((record[k] for k in possible_keys if k in record), None)
            if record["model_target"] is None:
                raise ValueError("No valid target key found in record")

            instruction = record.get("instruction") or record.get("question") or ""
            # Append any user-specified prompt add-ons
            instruction += " " + " ".join(prompt_add_ons[k] for k in user_prompt_add_ons if k in prompt_add_ons)
            record["instruction"] = instruction
            
            # Process system prompts
            system_prompt_text = "\n\n".join(system_prompts_mapping[k] for k in system_prompts if k in system_prompts_mapping)
            if system_prompt_text:
                record["system_prompt"] = system_prompt_text
                
            record["judge_type"] = properties.get("judge_type", "detailed")
            new_dataset.append(record)

        logger.info(f"Dataset is {total_duration / 3600:.2f} hours long")
        #print("DEBUG: Flattened record keys:", new_dataset[0].keys())
        return new_dataset
