import logging
logger = logging.getLogger(__name__)
from tqdm import tqdm


class AudiobenchPreprocessor():
    """Preprocessor for Audio benchmarks from AudioBench on HF."""

    def process(self, dataset: dict, properties: dict | None) -> list[dict]:
        """Process the dataset and flatten audio/context structure (expects dict-of-lists)."""
        logger.info("In [AudiobenchPreprocessor] Processing dataset...")
        #logger.info(dataset)
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

            record["instruction"] = record["instruction"] if "instruction" in record else record["question"] if "question" in record else "no instruction - use your judgement"
            record["judge_type"] = properties.get("judge_type", "detailed")
            new_dataset.append(record)

        logger.info(f"Dataset is {total_duration / 3600:.2f} hours long")
        #print("DEBUG: Flattened record keys:", new_dataset[0].keys())
        return new_dataset
