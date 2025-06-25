"""Pydantic models for experiment analysis."""

from typing import Any

from pydantic import BaseModel, ConfigDict


class Hyperparameters(BaseModel):
    """Model for experiment hyperparameters with flattening support."""

    model_config = ConfigDict(extra="allow")

    def flatten(self, prefix: str = "") -> dict[str, Any]:
        """Flatten nested hyperparameters into dot-notation keys."""
        result = {}
        for key, value in self.model_dump().items():
            full_key = f"{prefix}{key}" if prefix else key
            if isinstance(value, dict):
                nested = Hyperparameters(**value)
                result.update(nested.flatten(f"{full_key}."))
            else:
                result[full_key] = value
        return result
