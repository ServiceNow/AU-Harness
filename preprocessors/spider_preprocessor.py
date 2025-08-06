"""Spider preprocessor module for LALMEval framework.

This module provides a preprocessor for the Spider dataset, designed for
audio2SQL tasks with support for both audio and text modalities.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Any

import numpy as np
from tqdm import tqdm

from preprocessors.base import Preprocessor

logger = logging.getLogger(__name__)

# Constants for file paths and data selection
SPIDER_DATA_DIR = "data/spider/"
SPIDER_DB_DIR = "data/spider/database"

# Database columns
COLUMN_NAMES_ORIGINAL = "column_names_original"
TABLE_NAMES_ORIGINAL = "table_names_original"

class SpiderPreprocessor(Preprocessor):
    """
    Preprocessor for the Spider dataset, designed for audio2SQL tasks.
    Reference: https://github.com/awslabs/unified-text2sql-benchmark/blob/main/scripts/prepare_spider.py
    """

    def _load_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load a JSONL file and return a list of dictionaries.

        Args:
            file_path: Path to the JSONL file
        Returns:
            List of dictionaries representing the JSONL data
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]

    def _prepare_db_schemas(self, tables: list[dict]) -> dict:
        """Prepares database schemas from tables data.

        Args:
            tables: List of table information.

        Returns:
            A dictionary of database schemas.
        """
        db_schemas = {}
        for db in tables:
            schema = ""
            for i, table in enumerate(db[TABLE_NAMES_ORIGINAL]):
                schema += f"# {table} ("
                columns = []
                for col in db[COLUMN_NAMES_ORIGINAL]:
                    if col[0] == i:
                        col_name = col[1]
                        col_type = db["column_types"][
                            db[COLUMN_NAMES_ORIGINAL].index(col)
                        ]
                        columns.append(f"{col_name} {col_type}")
                schema += ", ".join(columns)
                schema += ")\n"

            # Add primary key constraints
            for pk in db["primary_keys"]:
                pk_col = db[COLUMN_NAMES_ORIGINAL][pk][1]
                pk_table = db[TABLE_NAMES_ORIGINAL][db[COLUMN_NAMES_ORIGINAL][pk][0]]
                schema += f"# Primary Key ({pk_table}): {pk_col}\n"

            # Add foreign key constraints
            for fk in db["foreign_keys"]:
                fk_col = db[COLUMN_NAMES_ORIGINAL][fk[0]][1]
                pk_col = db[COLUMN_NAMES_ORIGINAL][fk[1]][1]
                fk_table = db[TABLE_NAMES_ORIGINAL][db[COLUMN_NAMES_ORIGINAL][fk[0]][0]]
                pk_table = db[TABLE_NAMES_ORIGINAL][db[COLUMN_NAMES_ORIGINAL][fk[1]][0]]
                schema += f"# Foreign Key: {fk_table}.{fk_col} = {pk_table}.{pk_col}\n"

            db_schemas[db["db_id"]] = schema.rstrip()

        return db_schemas


    def process(
        self,
        dataset: Dict[str, List[Any]],
        num_samples: Optional[int] = None,
        properties: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Process the Spider dataset.

        Args:
            dataset: Dictionary containing audio data
            num_samples: Optional number of samples to process
            properties: Optional dict of properties

        Returns:
            A list of dictionaries where each dictionary represents a sample
        """


        # Extract properties using the base class method
        props = self.extract_properties(properties)
        modality = props.get("modality", "audio")

        # Get dataset info using base class method
        dataset_keys = list(dataset.keys())
        dataset_size = len(dataset.get("question", []))
        self.log_dataset_info(dataset_keys, dataset_size)
        indices = range(dataset_size if num_samples is None else min(dataset_size, num_samples))

        tables = self._load_jsonl(os.path.join(SPIDER_DATA_DIR, "tables.jsonl"))

        db_schemas = self._prepare_db_schemas(tables)

        processed_data: List[Dict[str, Any]] = []
        for i in tqdm(indices, desc="Processing samples"):
            db_id = dataset["db_id"][i]
            question = dataset["question"][i]
            query = dataset["query"][i]
            if modality == "text":
                audio_data = {
                    "array": np.array([]),  # Placeholder, not used in text-only evals
                    "sampling_rate": 16000
                }
            else:
                audio_data = dataset["audio"][i]

                # Validate audio data structure
                if not isinstance(audio_data, dict):
                    logger.warning("[%d] Invalid audio format. Skipping sample.", i)
                    continue

                # Convert to NumPy array
                audio_array = np.array(audio_data.get("array"))
                sr = audio_data.get("sampling_rate")

                if sr is None:
                    logger.warning("[%d] Sampling rate missing. Assuming 16kHz.", i)
                    sr = 16000

                # Use base class method to resample audio
                audio_array, sr = self.resample_audio(audio_array, sr)


            prompt = (
                "### Complete sqlite SQL query only and with no explanation, "
                "and do not select extra columns that are not explicitly requested "
                "in the query. Enclose the SQL query between ```sql and ```\n"
            )
            if modality == "audio":
                user_text = (
                    prompt
                    + "### Sqlite SQL tables, with their properties: \n#\n"
                    + db_schemas[db_id]
                    + "\n#\n### "
                )
            else:
                user_text = (
                    prompt
                    + "### Sqlite SQL tables, with their properties: \n#\n"
                    + db_schemas[db_id]
                    + "\n#\n### "
                    + question
                    + "\n"
                )

            processed_data.append(
                {
                    "id": i,
                    "instruction": user_text,
                    "array": audio_array if modality == "audio" else audio_data["array"],
                    "sampling_rate": sr if modality == "audio" else audio_data["sampling_rate"],
                    "model_target": query + "\t" + db_id,
                    "db_id": db_id,
                    "question": question,
                }
            )

        self.log_dataset_info(dataset_keys, dataset_size, len(processed_data))
        return processed_data
