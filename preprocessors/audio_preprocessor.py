import logging
logger = logging.getLogger(__name__)
from utils.multimodal import encode_audio_array_base64  # noqa: E402


class AudioBenchPreprocessor():
    """Preprocessor for Audio benchmarks from AudioBench on HF."""

    def process(self, dataset: list[dict], properties: dict | None) -> list[dict]:
        """Process the dataset and flatten audio/context structure."""
        total_duration = 0
        new_dataset = []
        for data in dataset:
            record = dict(data)  # shallow copy
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

            encoded_url = encode_audio_array_base64(record["array"], record["sampling_rate"])
            instruction = (
                "Question:\n" + record["instruction"] + "\n Choices:\n" + choices
                if (choices := record.get("choices"))
                else record["instruction"]
            )

            record["model_inputs"] = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {
                            "type": "audio_url",
                            "audio_url": {"url": encoded_url},
                        },
                    ],
                }
            ]
            if "reference" in record:
                record["model_target"] = record["reference"]
            elif "answer" in record:
                record["model_target"] = record["answer"]
            else:
                record["model_target"] = "no reference - use your judgement"

            record["question"] = record["instruction"]
            record["judge_type"] = properties.get("judge_type", "detailed")
            new_dataset.append(record)

        logger.info(f"Dataset is {total_duration / 3600:.2f} hours long")
        print("DEBUG: Flattened record keys:", new_dataset[0].keys())
        return new_dataset
