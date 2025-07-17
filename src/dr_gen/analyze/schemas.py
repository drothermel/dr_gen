"""Pydantic models for experiment analysis."""

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Hpms(BaseModel):
    """Model for experiment hyperparameters with flattening support."""

    model_config = ConfigDict(extra="allow")
    _flat_dict: dict[str, Any] = {}
    _flat_dict_prefix: str = ""

    def __setattr__(self, name, value):
        if hasattr(self, "_flat_dict"):
            object.__setattr__(self, "_flat_dict", {})
        super().__setattr__(name, value)

    def flatten(self, prefix: str = "") -> dict[str, Any]:
        """Flatten nested hyperparameters into dot-notation keys."""
        if prefix == self._flat_dict_prefix and self._flat_dict:
            return self._flat_dict
        result = {}
        for key, value in self.model_dump().items():
            full_key = f"{prefix}{key}" if prefix else key
            if isinstance(value, dict):
                nested = Hpms(**value)
                result.update(nested.flatten(f"{full_key}."))
            else:
                result[full_key] = value
        object.__setattr__(self, "_flat_dict", result)
        object.__setattr__(self, "_flat_dict_prefix", prefix)
        return result


class Run(BaseModel):
    """Model for a single experiment run with metrics and metadata."""

    run_id: str
    hpms: Hpms
    metrics: dict[str, list[float]]
    metadata: dict[str, Any] = {}

    def metric_names(self) -> list[str]:
        """Get all metric names."""
        return list(self.metrics.keys())

    def best_metric(self, metric_name: str) -> float | None:
        """Get best (minimum) metric."""
        if self.metrics.get(metric_name):
            return min(self.metrics[metric_name])
        return None

    def last_metric(self, metric_name: str) -> float | None:
        """Get last metric."""
        if self.metrics.get(metric_name):
            return self.metrics[metric_name][-1]
        return None

    def get_important_hpms(self, important_hpms: list[str]) -> dict[str, Any]:
        """Get important hyperparameters."""
        self.hpms.flatten()
        return {k: v for k, v in self.hpms._flat_dict.items() if k in important_hpms}

    @computed_field
    @property
    def best_train_loss(self) -> float | None:
        """Get best (minimum) training loss."""
        if self.metrics.get("train_loss"):
            return min(self.metrics["train_loss"])
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

    model_config = SettingsConfigDict(env_prefix="ANALYSIS_", env_file=".env")

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
    use_runs_filters: dict[str, Callable[[Run], bool]] = Field(default_factory=dict)
    main_hpms: list[str] = Field(default_factory=lambda: ["optim.lr", "batch_size"])
    grouping_exclude_hpms: list[str] = Field(
        default_factory=lambda: ["seed", "run_id", "tag"],
        description="Hyperparameters to exclude when grouping runs",
    )
    grouping_exclude_hpm_display_names: list[str] = Field(
        default_factory=list,
        description="Hyperparameters to exclude from display names when showing groups",
    )
