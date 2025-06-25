"""ExperimentDB class for unified analysis API."""

from pathlib import Path

import polars as pl
from pydantic import BaseModel, ConfigDict

from dr_gen.analyze.dataframes import runs_to_dataframe, runs_to_metrics_df
from dr_gen.analyze.models import AnalysisConfig
from dr_gen.analyze.parsers import load_runs_from_dir


class ExperimentDB(BaseModel):
    """Database interface for experiment analysis with lazy loading support."""

    config: AnalysisConfig
    base_path: Path
    lazy: bool = True
    _runs_df: pl.DataFrame | None = None
    _metrics_df: pl.DataFrame | None = None
    _errors: list[dict] = []

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self, config: AnalysisConfig, base_path: Path, lazy: bool = True
    ) -> None:
        """Initialize ExperimentDB with configuration and base path."""
        super().__init__(config=config, base_path=base_path, lazy=lazy)

    def load_experiments(self) -> None:
        """Load experiments from base_path directory."""
        runs, errors = load_runs_from_dir(self.base_path)
        self._runs_df = runs_to_dataframe(runs)
        self._metrics_df = runs_to_metrics_df(runs)
        self._errors = errors
