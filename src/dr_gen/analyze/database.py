"""ExperimentDB class for unified analysis API."""

from pathlib import Path
from pprint import pprint
from typing import Any
import polars as pl
from pydantic import BaseModel, ConfigDict

from dr_gen.analyze.dataframes import runs_to_dataframe, runs_to_metrics_df, query_metrics, summarize_by_hparams
from dr_gen.analyze.parsing import load_runs_from_dir
from dr_gen.analyze.schemas import AnalysisConfig, Run


class ExperimentDB(BaseModel):
    """Database interface for experiment analysis with lazy loading support."""

    config: AnalysisConfig
    base_path: Path
    all_runs: list[Run] = []
    active_runs: list[Run] = []
    lazy: bool = True
    _runs_df: pl.DataFrame | None = None
    _metrics_df: pl.DataFrame | None = None
    _errors: list[dict] = []

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self, config: AnalysisConfig, base_path=None, lazy: bool = True
    ) -> None:
        if base_path is None:
            base_path = Path(config.experiment_dir)
        """Initialize ExperimentDB with configuration and base path."""
        super().__init__(config=config, base_path=base_path, lazy=lazy)

    def load_experiments(self) -> None:
        """Load experiments from base_path directory."""
        self.all_runs = load_runs_from_dir(self.base_path, pattern="*metrics.jsonl")
        self.update_filtered_runs()
        self._errors = []

    def update_filtered_runs(self) -> None:
        """Filter runs based on the filter_out_runs configuration."""
        self.active_runs = []
        for run in self.all_runs:
            run.hpms.flatten()
            filter_results = {
                filter_name: filter_fxn(run) for filter_name, filter_fxn in self.config.use_runs_filters.items()
            }
            if all(filter_results.values()):
                self.active_runs.append(run)
        self._runs_df = runs_to_dataframe(self.active_runs)
        self._metrics_df = runs_to_metrics_df(self.active_runs)

    @property
    def important_hpms(self) -> list[str]:
        """Get a list of important hyperparameters."""
        return self.config.main_hpms

    @property
    def active_runs_df(self) -> pl.DataFrame:
        """Get a dataframe of active runs with only the main hyperparameter columns."""
        if self._runs_df is None:
            return pl.DataFrame()
        return self._runs_df.select(self.config.main_hpms)

    @property
    def active_runs_hpms(self) -> list[dict[str, Any]]:
        """Get a list of important hyperparameters for active runs."""
        return [run.get_important_hpms(self.important_hpms) for run in self.active_runs]

    def active_runs_grouped_by_hpms(self, exclude_hpms: list[str] | None = None) -> tuple[list[str], dict[tuple, list[Run]]]:
        """Group active runs by important hyperparameters, excluding specified hpms.
        
        Groups runs by self.important_hpms minus excluded hpms.
        
        Args:
            exclude_hpms: Hpms to exclude from grouping. If None, uses 
                         self.config.grouping_exclude_hpms
        
        Returns:
            Tuple of (grouping_keys, grouped_runs) where:
            - grouping_keys: List of hpm names used for grouping (important_hpms - exclusions)
            - grouped_runs: Dict mapping hpm value tuples to lists of runs
        
        Example:
            >>> # If important_hpms = ['optim.lr', 'optim.weight_decay', 'seed', 'batch_size']
            >>> # And exclude_hpms = ['seed']
            >>> keys, groups = db.active_runs_grouped_by_hpms()
            >>> print(keys)  # ['batch_size', 'optim.lr', 'optim.weight_decay'] (sorted)
            >>> print(next(iter(groups.keys())))  # (128, 0.01, 0.0001)
        """
        # Use config default if not specified
        if exclude_hpms is None:
            exclude_hpms = self.config.grouping_exclude_hpms
        
        # Calculate which hpms to actually group by
        grouping_hpms = [h for h in self.important_hpms if h not in exclude_hpms]
        
        grouped_runs = {}
        grouping_keys = None
        
        for run in self.active_runs:
            # Get only the hpms we want to group by
            run_hpms = run.get_important_hpms(grouping_hpms)
            
            # Sort for consistent ordering
            sorted_items = sorted(run_hpms.items(), key=lambda x: x[0])
            
            # Set grouping_keys once (all runs should have same keys)
            if grouping_keys is None:
                grouping_keys = [k for k, v in sorted_items]
            
            # Create the grouping tuple
            key = tuple(v for k, v in sorted_items)
            
            # Add run to group
            if key not in grouped_runs:
                grouped_runs[key] = []
            grouped_runs[key].append(run)
        
        return grouping_keys or [], grouped_runs

    def active_metrics_df(self) -> pl.DataFrame:
        """Get a dataframe of active metrics with only the main hyperparameter columns."""
        if self._metrics_df is None:
            return pl.DataFrame()
        return self._metrics_df.select(self.config.main_hpms)

    def query_metrics(
        self, metric_filter: str | None = None, run_filter: list[str] | None = None
    ) -> pl.DataFrame:
        """Query metrics with optional filters."""

        if self._metrics_df is None and not self.lazy:
            self.load_experiments()
        return query_metrics(self._metrics_df, metric_filter, run_filter)

    def summarize_metrics(self, hparams: list[str]) -> pl.DataFrame:
        """Get summary statistics grouped by hyperparameters."""

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
