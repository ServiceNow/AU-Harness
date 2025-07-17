from pydantic import BaseModel, ConfigDict, Field, field_validator, model_serializer

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
        super().__init__(**data)
        if not self.higher_is_better and self.display_name and INVERTED_METRIC_INDICATOR not in self.display_name:
            self.display_name = INVERTED_METRIC_INDICATOR + " " + self.display_name

    @field_validator("range")
    def check_increasing_range(cls, range_value):  # noqa: D102
        if len(range_value) != 2:
            raise ValueError("The range must contain exactly two elements.")
        if range_value[0] >= range_value[1]:
            raise ValueError("The range must be in increasing order.")
        return range_value

    @model_serializer()
    def serialize_model(self):  # noqa: D102
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "value": self.value,
            "range": self.range,
            "higher_is_better": self.higher_is_better,
        }

    def metadata_info(self) -> dict:
        """Return metadata info."""
        return {}
