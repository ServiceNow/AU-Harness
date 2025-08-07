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
    PreprocessorClass = get_class_from_module('preprocessors', preprocessor_name)
    if PreprocessorClass is None:
        error_msg = f"Could not load preprocessor {preprocessor_name}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    dataset = PreprocessorClass().process(repo, num_samples=num_samples, properties=properties)
    dataset_size = len(dataset) if dataset else 0
    return dataset, dataset_size

def _load_dataset(repo=None, filters=None, metric=None, split=None, dataset_info=None, dataset_name=None):
    """Load and preprocess a dataset from a local or remote path."""
    # Unwrap filters object
    filters = filters or {}
    num_samples = filters.get("num_samples", None)
    user_prompt_add_ons = filters.get("user_prompt_add_ons", [])
    system_prompts = filters.get("system_prompts", [])
    length_filter = filters.get("length_filter", None)
    
    # Extract parameters from dataset_info
    preprocessor_name = dataset_info.get("preprocessor", "GeneralPreprocessor") if dataset_info else "GeneralPreprocessor"
    subset = dataset_info.get("subset", None) if dataset_info else None
    
    # Set up properties that will be passed to any preprocessor
    properties = {"metric": metric}
    if user_prompt_add_ons:
        properties["user_prompt_add_ons"] = user_prompt_add_ons
    if system_prompts:
        properties["system_prompts"] = system_prompts
    if length_filter:
        properties["length_filter"] = tuple(length_filter)  # Convert list to tuple
    if dataset_info:
        properties["dataset_info"] = dataset_info
    if dataset_name:
        properties["dataset_name"] = dataset_name

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
    PreprocessorClass = get_class_from_module('preprocessors', preprocessor_name)
    if PreprocessorClass is None:
        error_msg = f"Could not load preprocessor {preprocessor_name}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    processed = PreprocessorClass().process(dset, num_samples, properties)
    dataset_size = len(processed)
    
    # Return both the processed dataset and its size
    return processed, dataset_size