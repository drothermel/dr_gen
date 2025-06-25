"""ExperimentDB class for unified analysis API."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from dr_gen.analyze.models import AnalysisConfig


class ExperimentDB(BaseModel):
    """Database interface for experiment analysis with lazy loading support."""

    config: AnalysisConfig
    base_path: Path
    lazy: bool = True

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self, config: AnalysisConfig, base_path: Path, lazy: bool = True
    ) -> None:
        """Initialize ExperimentDB with configuration and base path."""
        super().__init__(config=config, base_path=base_path, lazy=lazy)
