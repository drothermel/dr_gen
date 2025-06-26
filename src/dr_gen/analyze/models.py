"""Pydantic models for experiment analysis."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field
from pydantic_settings import BaseSettings


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


class Run(BaseModel):
    """Model for a single experiment run with metrics and metadata."""

    run_id: str
    hyperparameters: Hyperparameters
    metrics: dict[str, list[float]]
    metadata: dict[str, Any] = {}

    @computed_field
    @property
    def best_train_loss(self) -> float | None:
        """Get best (minimum) training loss."""
        if self.metrics.get("train/loss"):
            return min(self.metrics["train/loss"])
        return None

    @computed_field
    @property
    def best_val_acc(self) -> float | None:
        """Get best (maximum) validation accuracy."""
        if self.metrics.get("val/acc"):
            return max(self.metrics["val/acc"])
        return None

    @computed_field
    @property
    def final_epoch(self) -> int:
        """Get the final epoch number based on metric length."""
        lengths = [len(v) for v in self.metrics.values() if v]
        return max(lengths) - 1 if lengths else 0


class AnalysisConfig(BaseSettings):
    """Configuration for experiment analysis with environment variable support."""

    model_config = ConfigDict(env_prefix="ANALYSIS_", env_file=".env")

    # Data paths
    experiment_dir: str = Field(
        default="./experiments", description="Root experiment directory"
    )
    output_dir: str = Field(default="./analysis_output", description="Output directory")

    # Display mappings
    metric_display_names: dict[str, str] = Field(
        default_factory=lambda: {
            "train/loss": "Training Loss",
            "val/loss": "Validation Loss",
            "val/acc": "Validation Accuracy",
        }
    )
    hparam_display_names: dict[str, str] = Field(
        default_factory=lambda: {"lr": "Learning Rate", "batch_size": "Batch Size"}
    )
