import logging
logger = logging.getLogger(__name__)
from tqdm import tqdm
from pathlib import Path
import yaml


class AudiobenchPreprocessor():
    """Preprocessor for Audio benchmarks from AudioBench on HF."""

    def process(self, dataset: dict, properties: dict | None) -> list[dict]:
        """Process the dataset and flatten audio/context structure (expects dict-of-lists)."""
        logger.info("In [AudiobenchPreprocessor] Processing dataset...")
        #logger.info(dataset)
        user_prompt_add_ons = properties.get("user_prompt_add_ons", [])
        # Load prompt add-ons mapping
        prompt_yaml_path = Path(__file__).resolve().parent.parent / "prompts" / "prompt_add_ons.yaml"
        try:
            with open(prompt_yaml_path, "r") as f:
                prompt_add_ons = yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning(f"Prompt add-ons file not found at {prompt_yaml_path}. Proceeding without add-ons.")
            prompt_add_ons = {}

        total_duration = 0
        new_dataset = []
        keys = list(dataset.keys())
        num_samples = len(dataset[keys[0]]) if keys else 0
        logger.info(f"Dataset keys: {keys}, num_samples: {num_samples}")
        for i in tqdm(range(num_samples), desc="Preprocessing"):
            record = {k: dataset[k][i] for k in keys}
            logger.debug(f"Processing sample {i}: {record}")
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

            total_duration += len(record["array"]) / record["sampling_rate"]

            if "reference" in record:
                record["model_target"] = record["reference"]
            elif "answer" in record:
                record["model_target"] = record["answer"]
            else:
                record["model_target"] = "no reference - use your judgement"

            instruction = record.get("instruction") or record.get("question") or "no instruction - use your judgement"
            # Append any user-specified prompt add-ons
            for k in user_prompt_add_ons:
                add_on = prompt_add_ons.get(k)
                if add_on:
                    instruction = f"{instruction} {add_on}"
            #logger.info(f"[AudiobenchPreprocessor] Final instruction: {instruction}")
            record["instruction"] = instruction
            record["judge_type"] = properties.get("judge_type", "detailed")
            new_dataset.append(record)

        logger.info(f"Dataset is {total_duration / 3600:.2f} hours long")
        #print("DEBUG: Flattened record keys:", new_dataset[0].keys())
        return new_dataset
