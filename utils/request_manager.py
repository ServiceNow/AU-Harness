import asyncio
import logging
from typing import Dict

# Configure logging
logger = logging.getLogger(__name__)

class CentralRequestController:
    """
    Central controller for managing request tokens across multiple engines.
    Tracks request limits per model type only, not by engine.
    """
    def __init__(self):
        """
        Initialize the controller with per-model token pools.
        """
        self.lock = asyncio.Lock()
        
        # Track model type -> tokens mapping
        # model_type -> {total_tokens, available_tokens}
        self.model_pools: Dict[str, Dict[str, int]] = {}
            
    def register_model_type(self, model_type: str, batch_size: int):
        """
        Register a model type with its request limit.
        
        Args:
            model_type: Type of the model
            batch_size: Maximum concurrent requests for this model type
        """
        if model_type not in self.model_pools:
            self.model_pools[model_type] = {
                "total_tokens": batch_size,
                "available_tokens": batch_size
            }
            logger.info(f"Registered model type '{model_type}' with batch_size {batch_size}")
        
    def log_token_status(self):
        """
        Log the current token status for all model types.
        """
        logger.info("==== TOKEN POOL STATUS =====")
        for model_type, pool in self.model_pools.items():
            total = pool["total_tokens"]
            available = pool["available_tokens"]
            used = total - available
            usage_percent = (used / total) * 100 if total > 0 else 0
            logger.info(f"Model: {model_type:25} | Total: {total:4} | Used: {used:4} | Available: {available:4} | Usage: {usage_percent:.1f}%")
        logger.info("=============================")
    
    async def request_tokens(self, model_type: str, amount: int) -> int:
        """
        Request tokens for a specific model type.
        
        Args:
            model_type: Type of the model (e.g. 'gpt-4o-mini-audio-preview')
            amount: Number of tokens requested
            
        Returns:
            Number of tokens actually granted (may be less than requested)
        """
        async with self.lock:
            # Ensure the model type is registered
            if model_type not in self.model_pools:
                logger.warning(f"Model type '{model_type}' not registered, automatically registering with batch_size={amount}")
                self.register_model_type(model_type, amount)
                            
            # Get available tokens for this model type
            model_pool = self.model_pools[model_type]
            available = model_pool["available_tokens"]
            
            # Calculate how many tokens we can actually grant
            granted = min(amount, available)
            
            # Update available tokens for this model type
            model_pool["available_tokens"] -= granted
                
            # Return the granted tokens
            return granted
    
    async def return_tokens(self, model_type: str, amount: int) -> None:
        """
        Return tokens to the model's pool.
        
        Args:
            model_type: Type of the model
            amount: Number of tokens to return
        """
        async with self.lock:
            # Check if the model type is registered
            if model_type not in self.model_pools:
                logger.warning(f"Model type '{model_type}' not found in pool, cannot return tokens")
                return
            
            # Update available tokens for this model type
            model_pool = self.model_pools[model_type]
            model_pool["available_tokens"] += amount
            
            # Ensure we don't exceed the model's total tokens
            if model_pool["available_tokens"] > model_pool["total_tokens"]:
                logger.warning(f"Model {model_type} pool exceeded total tokens, capping at {model_pool['total_tokens']}")
                model_pool["available_tokens"] = model_pool["total_tokens"]
    

class EngineRequestManager:
    """
    Manages request tokens for a specific engine instance.
    Interfaces with the CentralRequestController to get and return tokens.
    Tracks allocations per model type and model instance.
    """
    def __init__(self, engine_id: str, central_controller: CentralRequestController):
        """
        Initialize the engine request manager.
        
        Args:
            engine_id: Unique identifier for this engine
            central_controller: Reference to the central request controller
        """
        self.engine_id = engine_id
        self.central_controller = central_controller
        # Track allocations by model_type and model_instance_id
        # model_type -> model_instance_id -> allocation
        self.model_allocations: Dict[str, Dict[str, int]] = {}
        self.lock = asyncio.Lock()
        logger.info(f"Model-specific EngineRequestManager initialized for engine {engine_id}")
    
    async def request_tokens(self, model_type: str, model_instance_id: str, amount: int) -> int:
        """
        Request tokens for a specific model type and instance.
        
        Args:
            model_type: Type of the model
            model_instance_id: Unique identifier for the model instance
            amount: Number of tokens requested
            
        Returns:
            Number of tokens granted
        """
        async with self.lock:
            # Request tokens from central controller for this model type
            #logger.info(f"Engine {self.engine_id}, Model {model_type}/{model_instance_id}: Requesting {amount} tokens")
            granted = await self.central_controller.request_tokens(
                model_type, amount)
            self.model_allocations.setdefault(model_type, {}).setdefault(model_instance_id, 0)
            self.model_allocations[model_type][model_instance_id] += granted
            #logger.info(f"Engine {self.engine_id}, Model {model_type}/{model_instance_id}: Granted {granted} tokens")
            return granted
    
    async def return_tokens(self, model_type: str, model_instance_id: str, amount: int) -> None:
        """
        Return tokens to the model's pool.
        
        Args:
            model_type: Type of the model
            model_instance_id: Unique identifier for the model instance
            amount: Number of tokens to return
        """
        async with self.lock:
            # Validate the return amount against our allocation
            actual_allocation = self.model_allocations.get(model_type, {}).get(model_instance_id, 0)
            if amount > actual_allocation:
                logger.warning(f"Engine {self.engine_id}, Model {model_type}/{model_instance_id}: "
                              f"Attempted to return {amount} tokens but only had {actual_allocation} allocated")
                amount = actual_allocation
            
            # Update our local tracking
            if model_type in self.model_allocations and model_instance_id in self.model_allocations[model_type]:
                self.model_allocations[model_type][model_instance_id] -= amount
            
            # Return tokens to central controller
            await self.central_controller.return_tokens(
                model_type, amount)
    