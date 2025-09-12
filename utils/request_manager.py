"""Request management system for AU-Harness framework."""

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

        # Track model name -> tokens mapping
        # model_name -> {total_tokens, available_tokens}
        self.model_pools: Dict[str, Dict[str, int]] = {}

    def register_model(self, model_name: str, batch_size: int):
        """
        Register a model name with its request limit.

        Args:
            model_name: Name of the model
            batch_size: Maximum concurrent requests for this model name
        """
        if model_name not in self.model_pools:
            self.model_pools[model_name] = {
                "total_tokens": batch_size,
                "available_tokens": batch_size
            }

    async def request_tokens(self, model_name: str, amount: int) -> int:
        """
        Request tokens for a specific model name.

        Args:
            model_name: Name of the model (e.g. 'model_1')
            amount: Number of tokens requested

        Returns:
            Number of tokens actually granted (may be less than requested)
        """
        async with self.lock:
            # Ensure the model name is registered
            if model_name not in self.model_pools:
                logger.warning(
                    "Model name '%s' not registered, automatically registering with batch_size=%s",
                    model_name, amount
                )
                self.register_model(model_name, amount)

            # Get available tokens for this model name
            model_pool = self.model_pools[model_name]
            available = model_pool["available_tokens"]

            # Calculate how many tokens we can actually grant
            granted = min(amount, available)

            # Update available tokens for this model name
            model_pool["available_tokens"] -= granted

            # Return the granted tokens
            return granted

    async def return_tokens(self, model_name: str, amount: int) -> None:
        """
        Return tokens to the model's pool.

        Args:
            model_name: Name of the model
            amount: Number of tokens to return
        """
        async with self.lock:
            # Check if the model name is registered
            if model_name not in self.model_pools:
                logger.warning(
                    "Model name '%s' not found in pool, cannot return tokens",
                    model_name
                )
                return

            # Update available tokens for this model name
            model_pool = self.model_pools[model_name]
            model_pool["available_tokens"] += amount

            # Ensure we don't exceed the model's total tokens
            if model_pool["available_tokens"] > model_pool["total_tokens"]:
                logger.warning(
                    "Model %s pool exceeded total tokens, capping at %s",
                    model_name, model_pool['total_tokens']
                )
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
        # Track allocations by model_name and model_instance_id
        # model_name -> model_instance_id -> allocation
        self.model_allocations: Dict[str, Dict[str, int]] = {}
        self.lock = asyncio.Lock()

    async def request_tokens(self, model_name: str, model_instance_id: str, amount: int) -> int:
        """
        Request tokens for a specific model name and instance.

        Args:
            model_name: Name of the model
            model_instance_id: Unique identifier for the model instance
            amount: Number of tokens requested

        Returns:
            Number of tokens granted
        """
        async with self.lock:
            # Request tokens from central controller for this model name
            granted = await self.central_controller.request_tokens(
                model_name, amount)
            self.model_allocations.setdefault(model_name, {}).setdefault(
                model_instance_id, 0
            )
            self.model_allocations[model_name][model_instance_id] += granted
            return granted

    async def return_tokens(self, model_name: str, model_instance_id: str, amount: int) -> None:
        """
        Return tokens to the model's pool.

        Args:
            model_name: Name of the model
            model_instance_id: Unique identifier for the model instance
            amount: Number of tokens to return
        """
        async with self.lock:
            # Validate the return amount against our allocation
            actual_allocation = self.model_allocations.get(model_name, {}).get(
                model_instance_id, 0
            )
            if amount > actual_allocation:
                logger.warning(
                    "Engine %s, Model %s/%s: Attempted to return %s tokens "
                    "but only had %s allocated",
                    self.engine_id, model_name, model_instance_id, amount, actual_allocation
                )
                amount = actual_allocation

            # Update our local tracking
            if (model_name in self.model_allocations and
                    model_instance_id in self.model_allocations[model_name]):
                self.model_allocations[model_name][model_instance_id] -= amount

            # Return tokens to central controller
            await self.central_controller.return_tokens(
                model_name, amount)
