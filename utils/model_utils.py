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

def load_models(cfg_list: list[dict], judge_properties: dict = None) -> tuple[list[Model], CentralRequestController]:
    """
    Load models and initialize the central request controller.
    
    Args:
        cfg_list: Configuration list for models
        judge_properties: Optional judge properties
        
    Returns:
        Tuple of (models list, central request controller)
    """
    models = []
    
    # Initialize the central request controller
    central_request_controller = CentralRequestController()
    
    # Register all models with the controller
    for cfg in cfg_list:
        model_name = cfg["info"].get("name")
        batch_size = cfg["info"].get("batch_size", 1)
        
        model_obj = Model(cfg["info"])
        models.append(model_obj)
        
        # Register model name with the controller
        if model_name:
            central_request_controller.register_model(model_name, batch_size)
    
    # Register judge model if available
    if judge_properties:
        judge_model = judge_properties.get("judge_model")
        judge_concurrency = judge_properties.get("judge_concurrency", 1)
        
        if judge_model:
            central_request_controller.register_model(judge_model, judge_concurrency)
    
    if not models:
        logger.error("[_load_models] No valid models found in configuration.")
        raise ValueError("No valid models found in configuration.")
        
    return models, central_request_controller