"""Base metric metadata classes for evaluation metrics.

This module provides the MetricMetadata class for handling metric information,
validation, and serialization using Pydantic models.
"""
from pydantic import BaseModel, ConfigDict, Field

from utils.constants import INVERTED_METRIC_INDICATOR


class MetricMetadata(BaseModel):
    """Class to handle metric metadata."""

    name: str = Field(default=None, description="Name of the metric")
    display_name: str = Field(default=None, description="Display name of the metric")
    description: str = Field(default=None, description="Description of the metric")
    value: float | None = Field(default=None, description="Value of the metric")
    range: tuple[int | None, int | None] = Field(default=(0, 1), description="Range of the metric")
    higher_is_better: bool = Field(default=True, description="Whether higher value is better")

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    def __init__(self, **data):
        """Initialize the MetricMetadata with the provided data.
        
        Args:
            **data: Keyword arguments to initialize the metric metadata
        """
        super().__init__(**data)
        if not self.higher_is_better and self.display_name and INVERTED_METRIC_INDICATOR not in self.display_name:
            self.display_name = INVERTED_METRIC_INDICATOR + " " + self.display_name
