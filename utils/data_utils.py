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

def load_dataset_with_args(dataset_path: str, split: str, subset: str, num_samples: int, task_name: str):
    """ Load the dataset
    
    Args:
        dataset_path: Path to dataset
        split: Split to load
        subset: Subset to load
        num_samples: Number of samples to load
        task_name: Name of the task
    
    Returns:
        dataset: Dataset loaded and transformed
        dataset_size: Size of the dataset
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

    if num_samples is not None:
        if len(dataset) > num_samples:
            dataset = dataset[:num_samples]
        else:
            logger.warning("Number of samples requested is greater than the number of samples in the dataset. Using all samples.")
    
    return dataset


def _load_dataset2(repo=None, filters=None, metric=None, split=None, task_config=None, task_name=None):
    """Load and preprocess a dataset from a local or remote path."""
    # Unwrap filters object
    filters = filters or {}
    num_samples = filters.get("num_samples", None)
    user_prompt_add_ons = filters.get("user_prompt_add_ons", [])
    system_prompts = filters.get("system_prompts", [])
    length_filter = filters.get("length_filter", None)
    
    # Extract parameters from dataset_info
    preprocessor_name = task_config.get("preprocessor", "GeneralPreprocessor") if task_config else "GeneralPreprocessor"
    subset = task_config.get("subset", None) if task_config else None
    
    # Set up properties that will be passed to any preprocessor
    properties = {"metric": metric}
    if user_prompt_add_ons:
        properties["user_prompt_add_ons"] = user_prompt_add_ons
    if system_prompts:
        properties["system_prompts"] = system_prompts
    if length_filter:
        properties["length_filter"] = tuple(length_filter)  # Convert list to tuple
    if task_config:
        properties["task_config"] = task_config
    if task_name:
        properties["task_name"] = task_name

    # Special handling for local CallHome dataset
    if preprocessor_name.startswith("Callhome"):
        return _load_callhome_dataset(repo, preprocessor_name, num_samples, properties)

    # For HuggingFace datasets
    if repo and (repo.startswith("/") or repo.startswith(".//")):
        repo = Path(repo).resolve()

    # Determine the preferred split to load directly (more efficient)
    if split is not None:
        if isinstance(split, str):
            preferred_splits = [split]
        else:
            preferred_splits = list(split)
    else:
        preferred_splits = ["test", "data", "train"]

    # Try to load a specific split directly
    dset = None
    # Try the preferred splits in order
    token=os.getenv("HF_TOKEN")
    for split_name in preferred_splits:
        try:
            args = {"path": repo, "split": split_name, "trust_remote_code": True}
            if subset:
                args["name"] = subset
            if token:
                args["token"] = token
            dset = load_dataset(**args)
            break
        except Exception as e:
            pass
    
    # Raise an error if no valid split was found
    if dset is None:
        try:
            args = {"path": repo, "trust_remote_code": True}
            if subset:
                args["name"] = subset
            if token:
                args["token"] = token
            dset = load_dataset(**args)
        except Exception as e:
            error_msg = f"[_load_dataset] No valid dataset found in {repo}"
            logger.error(error_msg)
            raise ValueError(e)

    if num_samples is not None:
        dset = dset[:num_samples]
    else:
        dset = dset[:len(dset)]
    preprocessor_class = get_class_from_module('preprocessors', preprocessor_name)
    if preprocessor_class is None:
        error_msg = f"Could not load preprocessor {preprocessor_name}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    processed = preprocessor_class().process(dset, num_samples, properties)
    dataset_size = len(processed)
    
    # Return both the processed dataset and its size
    return processed, dataset_size