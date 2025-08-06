import logging

from models.model import Model
from utils.request_manager import CentralRequestController

logger = logging.getLogger(__name__)


def _get_temperature_override(model_name: str, task_type: str, temperature_overrides: list[dict]) -> float | None:
    """Check if there's a temperature override for this model and task combination.
    
    Args:
        model_name: The name of the model
        task_type: The type of task being performed
        temperature_overrides: List of override dictionaries from config.yaml
        
    Returns:
        The override temperature if found, None otherwise
    """
    if not temperature_overrides:
        return None
        
    for override in temperature_overrides:
        # Get the temperature value if present
        temp = override.get("temperature")
        if temp is None:
            continue
            
        # Check if this override applies to our model/task
        override_model = override.get("model")
        override_task = override.get("task")
        
        # Case 1: Model+Task specific override
        if override_model == model_name and override_task == task_type:
            return float(temp)
            
        # Case 2: Model-only override
        if override_model == model_name and not override_task:
            return float(temp)
            
        # Case 3: Task-only override
        if not override_model and override_task == task_type:
            return float(temp)
    
    return None

def _get_system_prompt(model_name: str, dataset_name: str, system_prompts: list) -> str | None:
    """Get system prompt for this model and dataset combination.
    
    Args:
        model_name: The name of the model
        dataset_name: The name of the dataset
        system_prompts: List of system prompt configurations from filters
        
    Returns:
        Concatenated system prompts if found, None otherwise
    """
    if not system_prompts:
        return None
    
    # Import here to avoid circular imports
    from utils.util import find_runspec_files, _find_runspec_by_name, _get_task_type_datasets
    
    matched_prompts = []
    
    for prompt_config in system_prompts:
        # Each config should be [prompt_key, [model_name, dataset_name]]
        if not isinstance(prompt_config, list) or len(prompt_config) != 2:
            continue
            
        prompt_key, match_criteria = prompt_config
        
        if not isinstance(match_criteria, list) or len(match_criteria) != 2:
            continue
            
        criteria_model, criteria_dataset = match_criteria
        
        # Check if this prompt applies to our model first
        if criteria_model != model_name:
            continue
        
        # Now check dataset - need to flatten runspecs like temperature override does
        runspec_files = find_runspec_files()
        
        # Check if criteria_dataset is a runspec name (exact file match)
        found_runspec, runspec_data, _ = _find_runspec_by_name(criteria_dataset, runspec_files)
        
        if found_runspec:
            # If it's a runspec, check if our dataset is in it
            if dataset_name in runspec_data:
                matched_prompts.append(prompt_key)
        else:
            # Check if it's a task type (directory name like "paralinguistics")
            task_type_datasets = _get_task_type_datasets(criteria_dataset, runspec_files)
            
            if dataset_name in task_type_datasets:
                matched_prompts.append(prompt_key)
            elif criteria_dataset == dataset_name:
                # Direct dataset name comparison as fallback
                matched_prompts.append(prompt_key)
    
    # Return concatenated prompts if any matches found
    if matched_prompts:
        return " ".join(matched_prompts)
    
    return None

def register_models_with_controller(cfg_list: list[dict], judge_properties: dict = None) -> tuple[CentralRequestController, list[dict]]:
    """
    Register model types with the central request controller.
    
    Args:
        cfg_list: Configuration list for models
        judge_properties: Optional judge properties
        
    Returns:
        Tuple of (configured central request controller, model configs)
    """
    central_request_controller = CentralRequestController()
    
    # Register all models with the controller
    for cfg in cfg_list:
        model_type = cfg["info"].get("model")
        batch_size = cfg["info"].get("batch_size", 1)
        
        # Register model type with the controller
        if model_type:
            central_request_controller.register_model_type(model_type, batch_size)
    
    # Register judge model if available
    if judge_properties:
        judge_model = judge_properties.get("judge_model")
        judge_concurrency = judge_properties.get("judge_concurrency", 1)
        
        if judge_model:
            central_request_controller.register_model_type(judge_model, judge_concurrency)
    
    return central_request_controller, cfg_list

def load_models(cfg_list: list[dict]) -> list[Model]:
    """
    Load model instances from configuration.
    
    Args:
        cfg_list: Configuration list for models
        
    Returns:
        List of model instances
    """
    models = []
    
    # Create model instances
    for cfg in cfg_list:
        model_obj = Model(cfg["info"])
        models.append(model_obj)
    
    if not models:
        logger.error("[load_models] No valid models found in configuration.")
        raise ValueError("No valid models found in configuration.")
        
    return models