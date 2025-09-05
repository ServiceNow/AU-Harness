import logging

from models.model import Model
from utils.request_manager import CentralRequestController

logger = logging.getLogger(__name__)


def get_generation_params_override(model: str, task_ancestry: list, generation_params_override: list[dict]) -> dict | None:
    """Check if there's a temperature override for this model and task combination.
    
    Args:
        model: The model name
        task_ancestry: The ancestry path of the task (base_dir, intermediate dirs, task_name)
        generation_params_override: List of override dictionaries from config.yaml
        
    Returns:
        The override generation params if found, None otherwise
    """
    if not generation_params_override or not task_ancestry:
        return None
    
    # Get the task name (last element in ancestry)
    task_name = task_ancestry[-1] if task_ancestry else None
    
    # Store best match score and temperature for hierarchical matching
    best_match_score = -1
    best_match_generation_params = None
    
    for override in generation_params_override:
        # Get the temperature value if present
        generation_params = override.get("generation_params", None)
        if generation_params is None:
            continue
            
        # Check if this override applies to our model/task
        override_model = override.get("model", None)
        override_task = override.get("task", None)
        
        # Skip if model doesn't match and override has model constraint
        if override_model and override_model != model:
            continue
            
        # Calculate match score for hierarchical task matching
        match_score = -1
        
        # Case 1: Exact task name match (highest priority)
        if override_task == task_name:
            match_score = 100  # Highest priority
        
        # Case 2: Match with any folder in the ancestry path (medium priority)
        elif override_task and override_task in task_ancestry:
            # Find where in the hierarchy this folder/task appears
            # Items deeper in the hierarchy (closer to task) get higher scores
            position = task_ancestry.index(override_task)
            depth_score = position + 1  # Add 1 to avoid zero scores
            match_score = 10 + depth_score * 5  # Base of 10 plus position bonus
        
        # Case 3: No task constraint but model matches
        elif override_model == model and not override_task:
            match_score = 5  # Lower priority than task-specific overrides
        
        # Update best match if we found a better one
        if match_score > best_match_score:
            best_match_score = match_score
            best_match_generation_params = generation_params
    
    return best_match_generation_params

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
        model_name = cfg.get("name")
        batch_size = cfg.get("batch_size", 1)
        
        # Register model type with the controller
        if model_name:
            central_request_controller.register_model(model_name, batch_size)
    
    # Register judge model if available
    if judge_properties:
        judge_model = judge_properties.get("judge_model")
        judge_concurrency = judge_properties.get("judge_concurrency", 1)
        
        if judge_model:
            central_request_controller.register_model(judge_model, judge_concurrency)
    
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
        model_obj = Model(cfg)
        models.append(model_obj)
    
    if not models:
        logger.error("[load_models] No valid models found in configuration.")
        raise ValueError("No valid models found in configuration.")
        
    return models