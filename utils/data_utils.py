import os
from pathlib import Path
from datasets import load_dataset
from utils.util import get_class_from_module
import logging

logger = logging.getLogger(__name__)

def _load_callhome_dataset(repo, preprocessor_name, num_samples, properties):
    """Load and process a CallHome dataset using the specified preprocessor."""
    repo = Path(repo).resolve()
    # Dynamically load the preprocessor
    preprocessor_class = get_class_from_module('preprocessors', preprocessor_name)
    if preprocessor_class is None:
        error_msg = f"Could not load preprocessor {preprocessor_name}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    dataset = preprocessor_class().process(repo, num_samples=num_samples, properties=properties)
    dataset_size = len(dataset) if dataset else 0
    return dataset, dataset_size

def load_dataset_with_args(dataset_path: str, split: str, subset: str, task_name: str):
    """ Load the dataset
    
    Args:
        dataset_path: Path to dataset
        split: Split to load
        subset: Subset to load
        task_name: Name of the task
    
    Returns:
        dataset: Dataset loaded and transformed
    """
    if dataset_path is None:
        raise ValueError(f'Dataset path is missing for task {task_name}')
    
    if split is None:
        raise ValueError(f'Dataset split is missing for task {task_name}')

    token=os.getenv("HF_TOKEN")

    # Load dataset
    try: 
        dataset_load_args = {"path": dataset_path, "split": split, "trust_remote_code": True}
        if subset:
            dataset_load_args["name"] = subset
        if token:
            dataset_load_args["token"] = token
        dataset = load_dataset(**dataset_load_args)
    except Exception as e:
        raise ValueError(e)

    if dataset is None:
        raise ValueError(f"Dataset with path {dataset_path}, split {split} and subset {subset} not found")
    
    return dataset