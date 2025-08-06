import asyncio
import json
import logging
import math
import time
from tqdm import tqdm

from models.model import Model
from metrics.llm_judge import _BaseLLMJudge
from postprocessors.base import Postprocessor
from utils.constants import metric_map, metric_output
from utils.data_utils import _load_dataset
from utils.metric_utils import _load_metric
from utils.model_utils import _get_temperature_override
from utils.request_manager import EngineRequestManager
from utils.util import get_class_from_module

logger = logging.getLogger(__name__)

class Engine:
    """Evaluate one or many models over the same dataset concurrently."""
    def __init__(self, models: list[Model], dataset_name: str, dataset_info: dict, metric_name: str, 
                 filters: dict, task_type: str = None, temperature_overrides: list[dict] = None, 
                 engine_id: str = None, request_manager = None, judge_properties: dict = None):
        
        # Load dataset using the existing _load_dataset function
        repo = dataset_info.get("hf_repo", None)
        if not repo:
            repo = dataset_info.get("path", None)
        
        split = dataset_info.get("split", None)
        
        self.dataset, _ = _load_dataset(
            repo, filters=filters, metric=metric_name, split=split, dataset_info=dataset_info
        )
        
        # Load metric using the existing _load_metric function
        language = dataset_info.get("language", "en")
        self.metric = _load_metric(metric_name, language=language, judge_settings=judge_properties)
        
        # Load postprocessor using the existing logic
        postprocessor_name = dataset_info.get("postprocessor", "GeneralPostprocessor")
        PostprocessorClass = get_class_from_module('postprocessors', postprocessor_name)
        if PostprocessorClass is None:
            logger.warning(f"Could not load postprocessor {postprocessor_name}, using default GeneralPostprocessor")
            PostprocessorClass = get_class_from_module('postprocessors', 'GeneralPostprocessor')
        self.postprocessor = PostprocessorClass()
        
        logger.info(f"[Engine.__init__] Initializing Engine with {len(models)} model(s), dataset size: {len(self.dataset)}, metric: {self.metric.name}")
        
        self.models = models
        self.dataset_name = dataset_name
        self.engine_id = engine_id
        self.request_manager = request_manager
        # Store task_type for temperature setting
        self.task_type = task_type
        # Store temperature overrides
        self.temperature_overrides = temperature_overrides or []
        
        # Group models by their model attribute for sharding
        self.model_groups = {}
        for model in models:
            model_type = model.model  # The model attribute we're sharding on
            if model_type not in self.model_groups:
                self.model_groups[model_type] = []
            self.model_groups[model_type].append(model)

    # ---------------- internal helpers ----------------x
    # infer by batch size, calling generate text with retry for each sample
    async def _infer_single_model(self, model: Model, samples=None) -> list[str]:
        samples = samples if samples is not None else self.dataset  # Use provided samples or full dataset
        task_type = self.task_type
        
        # Check for temperature override for this specific model and task combination
        override_temp = _get_temperature_override(model.name(), task_type, self.temperature_overrides)
        if override_temp is not None:
            # Use the override temperature directly
            logger.info(f"[Engine._infer_single_model] Using override temperature {override_temp} for model {model.name()} and task {task_type}")
            model.temperature = override_temp
            model.req_resp_hndlr.temperature = override_temp
        else:
            # Use the standard task-based temperature setting
            model.set_temp(task_type)
        # Get model type for request management
        model_type = model.model  # The actual model type (e.g., "gpt-4o-mini-audio-preview")
        # Generate a unique model instance ID
        model_instance_id = f"{model.name()}_{id(model)}"
        
        # Create an adjustable semaphore based on granted tokens
        token_sem = asyncio.Semaphore(0)  # Start with 0 tokens
        
        # Keep track of pending and completed samples
        pending_samples = list(range(len(samples)))  # Indices of samples waiting for tokens
        completed_samples = set()  # Indices of samples that have completed processing
        
        # Continuously ask for tokens(with backoff if none available) based on pending samples
        # Have dynamic wait times for by dataset size - this "layering" of Engine priority gives models in the same Engine similar priority, so they don't wait on each other as often
        async def token_manager():
            request_count = 0
            # Calculate wait times based on dataset size
            dataset_size = len(samples)
            
            # Calculate a scale factor from 0 to 1 based on dataset size
            scale_factor = min(1.0, max(0.0, math.log10(dataset_size + 10) / 4.0))
            
            # Scale the no-token wait time between 0.5s and 2s
            no_token_wait = scale_factor * 2.0
            
            # Double the wait time when tokens are granted
            token_wait = no_token_wait * 2.0
            
            while len(pending_samples) > 0:
                request_count += 1
                # Request as many tokens as we need for pending samples
                request_amount = min(model.batch_size, len(pending_samples))
                
                if request_amount > 0:
                    granted = await self.request_manager.request_tokens(
                        model_type, model_instance_id, request_amount)
                    
                    if granted > 0:
                        # Remove samples from pending list based on granted tokens
                        pending_samples[:] = pending_samples[granted:]
                        # Release semaphore permits for each granted token
                        for _ in range(granted):
                            token_sem.release()
                        # Wait based on dataset size when tokens were granted
                        await asyncio.sleep(token_wait)
                    else:
                        # Backoff when no tokens were granted, based on dataset size
                        # Apply a small multiplier for repeated failures, but cap it
                        backoff_multiplier = min(3.0, 1.0 + (request_count / 10))
                        await asyncio.sleep(no_token_wait * backoff_multiplier)
                else:
                    break
        
        asyncio.create_task(token_manager())
            
        # Wait for token to be available, run task, and add to completed
        async def _call_with_token_mgmt(idx: int, sample: dict):
            # Acquire a token
            await token_sem.acquire()
            try:
                # Process the sample
                resp = await model.generate_text_with_retry(sample, 
                                                          {"chunk_size": model.chunk_size, 
                                                           "metric": self.metric.name})
                result = resp
                
                # Add to completed set
                completed_samples.add(idx)
                
                # Return token to model's pool
                await self.request_manager.return_tokens(model_type, model_instance_id, 1)
                                
                return idx, result
            except Exception as e:
                # Make sure to return token on error
                logger.error(f"[Engine._infer_single_model] Error processing sample {idx} in {self.dataset_name}: {e}")
                completed_samples.add(idx)
                await self.request_manager.return_tokens(model_type, model_instance_id, 1)
                return idx, ""
        
        # Create tasks paired with their original index
        tasks = [_call_with_token_mgmt(i, ex) for i, ex in enumerate(samples)]
        
        # Process results in order of completion
        results: list[str | None] = [None] * len(tasks)
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), 
                          desc=f"Inference ({self.dataset_name} | {model.name()} | {self.metric.name})"):
            idx, resp = await coro
            results[idx] = resp
            
        return results

    # Infer all models concurrently
    async def _infer_all(self):
        results = {}
        all_tasks = []
        task_info = {}

        # Prepare all tasks for concurrent execution
        for model_type, models in self.model_groups.items():
            if len(models) > 1:  # Multiple instances of the same model type - need sharding
                logger.info(
                    f"[Engine._infer_all] Sharding dataset for {len(models)} instances of model type '{model_type}'")
                # Divide dataset among model instances
                shard_size = len(self.dataset) // len(models)
                
                # Track the mapping of original indices to shard indices for recombination
                index_mappings = {}
                sharded_tasks = {}
                
                # Distribute samples and create tasks
                for i, model in enumerate(models):
                    start_idx = i * shard_size
                    # Last model gets any remaining samples
                    end_idx = (i + 1) * shard_size if i < len(models) - 1 else len(self.dataset)
                    shard = self.dataset[start_idx:end_idx]

                    # Keep track of original indices
                    index_mappings[model.name()] = list(range(start_idx, end_idx))
                    
                    task = asyncio.create_task(self._infer_single_model(model, shard))
                    sharded_tasks[model.name()] = task
                    all_tasks.append(task)
                
                # Store info for later reconstruction
                task_info[model_type] = {
                    "is_sharded": True,
                    "tasks": sharded_tasks,
                    "index_mappings": index_mappings
                }
            else:
                # Single instance, normal processing
                model = models[0]
                model_name = model.name()
                task = asyncio.create_task(self._infer_single_model(model))
                all_tasks.append(task)
                task_info[model_name] = {
                    "is_sharded": False,
                    "task": task
                }
        # Wait for all tasks to complete concurrently
        await asyncio.gather(*all_tasks)
        # Process results according to task type
        for key, info in task_info.items():
            if info["is_sharded"]:
                # Reconstruct sharded results
                shard_results = {name: task.result() for name, task in info["tasks"].items()}
                
                # Combine results under model_type as the key
                combined_results = [None] * len(self.dataset)

                # Use index mappings to put results back in correct order
                for model_name, model_results in shard_results.items():
                    original_indices = info["index_mappings"][model_name]
                    for shard_idx, orig_idx in enumerate(original_indices):
                        if shard_idx < len(model_results):
                            combined_results[orig_idx] = model_results[shard_idx]

                # Use the model_type as the key for combined results
                results[key] = combined_results
            else:
                # Single instance result
                results[key] = info["task"].result()
        
        return results

    async def run(self):
        logger.info(f"[Engine.run] Starting evaluation run for {self.dataset_name} with metric {self.metric.name}.")
        raw_predictions = await self._infer_all()
        logger.info(f"[Engine.run] Predictions complete for {self.dataset_name}. Calculating scores...")
        scores = {}
        # Pass the metric name to the postprocessor
        logger.info("raw_predictions: %s", raw_predictions)
        process_result = self.postprocessor.process(dataset=self.dataset, predictions=raw_predictions, metric=self.metric.name)
        # Extract values from the dictionary returned by the postprocessor
        model_targets = process_result["model_targets"]
        predictions = process_result["processed_predictions"]
        instructions = process_result.get("instructions", None)
        ids = process_result.get("ids", [])
        lengths = process_result.get("lengths", [])

        # Determine if this is an LLM-judge metric
        is_llm_judge = isinstance(self.metric, _BaseLLMJudge)
        
        # Get the metric name from the current metric instance
        metric_name = self.metric.name
        
        # Get judge_properties from the metric if it's a LLM judge
        judge_settings = getattr(self.metric, '_judge_properties', None) if is_llm_judge else None
        
        # Get language attribute from the metric if available or default to 'en'
        language = getattr(self.metric, 'language', 'en')
        
        # Create metric instances for each model using _load_metric
        # _load_metric will handle deciding which parameters to use based on the metric type
        metric_instances = {
            model_name: _load_metric(metric_name, language=language, judge_settings=judge_settings)
            for model_name in predictions.keys()
        }
        
        async def score_model_with_tokens(model_name, outs):
            metric = metric_instances[model_name]
            model_responses = raw_predictions.get(model_name, [])
            # Check if this is an LLM judge
            if is_llm_judge:
                # For LLM judges, set the request manager
                # The metric will handle token management internally with the Engine Manager
                metric.set_request_manager(self.request_manager)
                
                if ids and lengths:
                    result = await metric(outs, model_targets, ids, lengths,
                                        instructions=instructions, dataset_name=self.dataset_name, model_name=model_name, model_responses=model_responses)
                else:
                    result = await metric(outs, model_targets,
                                        instructions=instructions, dataset_name=self.dataset_name, model_name=model_name, model_responses=model_responses)
                return model_name, result
            else:
                # For regular metrics, just run them directly (no token management needed)
                if ids and lengths:
                    result = await asyncio.to_thread(
                        metric, outs, model_targets, ids, lengths,
                        instructions=instructions, dataset_name=self.dataset_name, model_name=model_name, model_responses=model_responses
                    )
                else:
                    result = await asyncio.to_thread(
                        metric, outs, model_targets,
                        instructions=instructions, dataset_name=self.dataset_name, model_name=model_name, model_responses=model_responses
                    )
                return model_name, result

        # Run all model scoring concurrently
        tasks = [score_model_with_tokens(model_name, outs) for model_name, outs in predictions.items()]
        results = await asyncio.gather(*tasks)
        for model_name, model_score in results:
            scores[model_name] = model_score
        # Return scores directly without nesting under metric name
        logger.info("[Engine.run] Evaluation complete. Returning scores.")
        return scores

def create_engines(dataset_name, dataset_info, task_type, metric_name, filters, models, temperature_overrides, judge_properties, central_request_controller):
    """
    Process a dataset and run evaluation on it.
    
    Args:
        dataset_name: Name of the dataset to process
        dataset_info: Dictionary containing dataset information
        task_type: Type of task for temperature settings
        metric_name: Name of the metric to use
        filters: Dictionary containing filter settings
        models: List of model instances
        temperature_overrides: List of temperature override configurations
        judge_properties: Dictionary of judge properties
        central_request_controller: The central request controller instance
        
    Returns:
        tuple: (Engine instance, dataset_name)
    """
    # Get needed settings directly from filters
    accented_filter = filters.get("accented", None)
    language_filter = filters.get("language", None)
    
    # Check if we need to filter out accented datasets
    if accented_filter is False and dataset_info.get("accented", False) is True:
        logger.info(f"[_process_dataset] Skipping dataset '{dataset_name}' because it is accented and accented filter is False")
        return False
        
    # Check if we need to filter by language
    if language_filter is not None:
        dataset_language = dataset_info.get("language", "").lower()
        if dataset_language and language_filter.lower() != dataset_language:
            logger.info(f"[_process_dataset] Skipping dataset '{dataset_name}' because its language '{dataset_language}' doesn't match filter '{language_filter}'")
            return False
    
    logger.info(f"[_process_dataset] Creating engine for dataset '{dataset_name}' with metric '{metric_name}' ...")
    
    # Create engine ID and request manager
    engine_id = f"{dataset_name}_{metric_name}_{int(time.time())}"
    engine_request_manager = EngineRequestManager(engine_id, central_request_controller)
    
    # Create Engine - it will handle dataset/metric/postprocessor loading internally
    result = Engine(models=models, dataset_name=dataset_name, dataset_info=dataset_info, 
                   metric_name=metric_name, filters=filters, task_type=task_type, 
                   temperature_overrides=temperature_overrides, engine_id=engine_id, 
                   request_manager=engine_request_manager, judge_properties=judge_properties)

    return result, dataset_name

async def run_all_engines(all_engines):
    """
    Run all engines concurrently and collect results.
    
    Args:
        all_engines: List of (engine, dataset_name) tuples
        
    Returns:
        Dictionary of all scores
    """
    logger.info(f"[run_all_engines] Running {len(all_engines)} engines concurrently...")
    
    # Store all scores in a flat dict
    all_scores = {}
    
    # Create tasks for each engine
    tasks = []
    for engine, dataset_name in all_engines:
        tasks.append(engine.run())
    
    # Run all tasks concurrently and gather results
    results = await asyncio.gather(*tasks)
    
    # Store results in all_scores dictionary
    for (engine, dataset_name), result in zip(all_engines, results):
        if dataset_name not in all_scores:
            all_scores[dataset_name] = {}
            
        # Result is in format: {metric_name: {model_name: scores}}
        for metric_name, model_scores in result.items():
            if metric_name not in all_scores[dataset_name]:
                all_scores[dataset_name][metric_name] = {}
                
            # Add model scores to the right metric bucket
            for model_name, scores in model_scores.items():
                all_scores[dataset_name][metric_name][model_name] = scores
    logger.info(f"[run_all_engines] Evaluation complete. Final results:")
    logger.info(json.dumps(all_scores, indent=2))             
    return all_scores