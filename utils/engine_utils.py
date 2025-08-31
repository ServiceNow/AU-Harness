"""Utility functions for creating and running evaluation engines.

This module provides functionality to create engine instances for specific tasks
and run multiple evaluation engines concurrently. It handles the initialization
of models, task filtering, and result collection.
"""

import asyncio
import logging
import time

from utils.model_utils import load_models
from utils.request_manager import EngineRequestManager

from engine import Engine

logger = logging.getLogger(__name__)

def create_engine(task_info, run_config, central_request_controller):
    """
    Process a task and run evaluation on it.
    
    Args:
        task_info: Tuple of (task_name, metric_name, task_config, ancestry)
        run_config: Run configuration
        central_request_controller: The central request controller instance
        
    Returns:
        tuple: (Engine instance, task_name)
    """
    # Unpack task_info
    task_name, metric_name, task_config, _ = task_info

    filters = run_config.get("filters", {})
    accented_filter = filters.get("accented", None)
    language_filter = filters.get("language", None)
    
    # Check if we need to filter out accented datasets
    if accented_filter is False and task_config.get("accented", False) is True:
        logger.info("[_process_dataset] Skipping task '%s' because it is accented and accented filter is False", task_name)
        return False
        
    # Check if we need to filter by language
    if language_filter is not None:
        task_language = task_config.get("language", "").lower()
        if task_language and language_filter.lower() != task_language:
            logger.info("[_process_dataset] Skipping task '%s' because its language '%s' doesn't match filter '%s'", task_name, task_language, language_filter)
            return False
    
    logger.info("[_process_dataset] Creating engine for task '%s' with metric '%s' ...", task_name, metric_name)
    
    # Load models for this engine - each engine gets its own instances
    models = load_models(run_config.get("models", []))
    
    # Create engine ID and request manager
    engine_id = f"{task_name}_{metric_name}_{int(time.time())}"
    engine_request_manager = EngineRequestManager(engine_id, central_request_controller)
    
    # Create Engine - it will handle preprocessing/inference/postprocessing/evaluations
    task_engine = Engine(models=models,
        task_info=task_info,
        run_config=run_config, 
        engine_id=engine_id, 
        request_manager=engine_request_manager,
    )

    return task_engine, task_name

async def run_all_engines(all_engines):
    """
    Run all engines concurrently and collect results.
    
    Args:
        all_engines: List of (engine, task_name) tuples
        
    Returns:
        Dictionary of all scores
    """
    logger.info("[run_all_engines] Running %d engines concurrently...", len(all_engines))
    
    # Store all scores in a flat dict
    all_scores = {}
    
    # Create tasks for each engine
    tasks = []
    for engine, task_name in all_engines:
        tasks.append(engine.run())
    
    # Run all tasks concurrently and gather results
    results = await asyncio.gather(*tasks)
    
    # Store results in all_scores dictionary
    for (engine, task_name), result in zip(all_engines, results):
        if task_name not in all_scores:
            all_scores[task_name] = {}
            
        # Result is in format: {metric_name: {model_name: scores}}
        for metric_name, model_scores in result.items():
            if metric_name not in all_scores[task_name]:
                all_scores[task_name][metric_name] = {}
                
            # Add model scores to the right metric bucket
            for model_name, scores in model_scores.items():
                all_scores[task_name][metric_name][model_name] = scores           
    return all_scores