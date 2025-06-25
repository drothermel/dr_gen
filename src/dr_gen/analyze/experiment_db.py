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
        runs = load_runs_from_dir(self.base_path)
        self._runs_df = runs_to_dataframe(runs)
        self._metrics_df = runs_to_metrics_df(runs)
        self._errors = []

    def query_metrics(
        self, metric_filter: str | None = None, run_filter: list[str] | None = None
    ) -> pl.DataFrame:
        """Query metrics with optional filters."""
        from dr_gen.analyze.dataframes import query_metrics

        if self._metrics_df is None and not self.lazy:
            self.load_experiments()
        return query_metrics(self._metrics_df, metric_filter, run_filter)

    def summarize_metrics(self, hparams: list[str]) -> pl.DataFrame:
        """Get summary statistics grouped by hyperparameters."""
        from dr_gen.analyze.dataframes import summarize_by_hparams

        if self._metrics_df is None and not self.lazy:
            self.load_experiments()
        if self._runs_df is None:
            return pl.DataFrame()
        return summarize_by_hparams(self._runs_df, self._metrics_df, hparams)

    def lazy_query(self) -> pl.LazyFrame:
        """Get a lazy frame for efficient querying of large datasets."""
        if self._metrics_df is None:
            # Scan JSONL files lazily
            pattern = str(self.base_path / "*.jsonl")
            return pl.scan_ndjson(pattern).lazy()
        return self._metrics_df.lazy()

    def stream_metrics(self) -> pl.DataFrame:
        """Stream metrics for memory-efficient processing of large datasets."""
        lazy_frame = self.lazy_query()
        return lazy_frame.collect(streaming=True)
