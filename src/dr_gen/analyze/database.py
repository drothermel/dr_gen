"""ExperimentDB class for unified analysis API."""

from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl
from pydantic import BaseModel, ConfigDict

from dr_gen.analyze.dataframes import (
    query_metrics,
    runs_to_dataframe,
    runs_to_metrics_df,
    summarize_by_hparams,
)
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
                filter_name: filter_fxn(run)
                for filter_name, filter_fxn in self.config.use_runs_filters.items()
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

    def active_runs_grouped_by_hpms(
        self, exclude_hpms: list[str] | None = None
    ) -> tuple[list[str], dict[tuple, list[Run]]]:
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

    def get_display_name(self, name: str, name_type: str = "auto") -> str:
        """Get the display name for a metric or hyperparameter.

        Args:
            name: The technical name (e.g., 'train_loss', 'optim.lr')
            name_type: Type of name - 'metric', 'hparam', or 'auto' (default)
                      'auto' will check both mappings

        Returns:
            Display name if found, otherwise returns the original name

        Example:
            >>> db.get_display_name("train_loss")  # 'Train Loss'
            >>> db.get_display_name("optim.lr")  # 'Learning Rate'
            >>> db.get_display_name("unknown_metric")  # 'unknown_metric'
        """
        if name_type in ["metric", "auto"]:
            if name in self.config.metric_display_names:
                return self.config.metric_display_names[name]

        if name_type in ["hparam", "auto"]:
            if name in self.config.hparam_display_names:
                return self.config.hparam_display_names[name]

        # Return original name if no mapping found
        return name

    def format_hparam_value(self, name: str, value: Any) -> str:
        """Format hyperparameter value for display.

        Args:
            name: Hyperparameter name (e.g., 'optim.lr', 'optim.weight_decay')
            value: The value to format

        Returns:
            Formatted string representation

        Example:
            >>> db.format_hparam_value("optim.lr", 0.001)  # '1e-03'
            >>> db.format_hparam_value("batch_size", 128)  # '128'
        """
        # Special formatting for certain hyperparameters
        if name in ["optim.lr", "optim.weight_decay", "lr", "weight_decay", "wd"]:
            if isinstance(value, (int, float)) and 0 < value < 0.01:
                # Use scientific notation for small values
                return f"{value:.0e}"
            if isinstance(value, (int, float)):
                # Remove trailing zeros for larger values
                return f"{value:g}"

        # Default formatting
        return str(value)

    def group_key_to_dict(
        self,
        group_key: tuple,
        hpm_names: list[str] | None = None,
        use_display_names: bool = False,
    ) -> dict[str, Any]:
        """Convert a group key tuple to a hyperparameter dictionary.

        Args:
            group_key: Tuple of hyperparameter values from active_runs_grouped_by_hpms
            hpm_names: List of hyperparameter names. If None, uses the most recent
                      grouping keys from active_runs_grouped_by_hpms
            use_display_names: If True, use display names as dict keys

        Returns:
            Dict mapping hyperparameter names (or display names) to values

        Example:
            >>> hpm_names, run_groups = db.active_runs_grouped_by_hpms()
            >>> for group_key, runs in run_groups.items():
            >>> # Technical names
            >>>     hpm_dict = db.group_key_to_dict(group_key, hpm_names)
            >>>     print(f"LR: {hpm_dict['optim.lr']}, WD: {hpm_dict['optim.weight_decay']}")
            >>> # Display names
            >>>     display_dict = db.group_key_to_dict(group_key, hpm_names, use_display_names=True)
            >>>     print(f"{display_dict['Learning Rate']}, {display_dict['Weight Decay']}")
        """
        if hpm_names is None:
            # Use the default grouping to get hpm names
            exclude_hpms = self.config.grouping_exclude_hpms
            hpm_names = [h for h in self.important_hpms if h not in exclude_hpms]
            hpm_names.sort()  # Ensure consistent ordering

        if len(group_key) != len(hpm_names):
            raise ValueError(
                f"group_key length ({len(group_key)}) doesn't match hpm_names length ({len(hpm_names)})"
            )

        if use_display_names:
            # Create dict with display names as keys
            return {
                self.get_display_name(name, "hparam"): value
                for name, value in zip(hpm_names, group_key, strict=False)
            }
        return dict(zip(hpm_names, group_key, strict=False))

    def format_group_description(
        self, 
        group_key: tuple, 
        hpm_names: list[str] | None = None,
        exclude_from_display: list[str] | None = None
    ) -> str:
        """Create a formatted string description of a hyperparameter group.

        Args:
            group_key: Tuple of hyperparameter values from active_runs_grouped_by_hpms
            hpm_names: List of hyperparameter names. If None, uses the most recent
                      grouping keys from active_runs_grouped_by_hpms
            exclude_from_display: Hpms to exclude from the display string. If None,
                                uses self.config.grouping_exclude_hpm_display_names

        Returns:
            Formatted string with display names and formatted values

        Example:
            >>> desc = db.format_group_description((128, 0.001, 0.0001))
            >>> print(
            ...     desc
            ... )  # "Batch Size: 128, Learning Rate: 1e-03, Weight Decay: 1e-04"
        """
        hpm_dict = self.group_key_to_dict(group_key, hpm_names, use_display_names=False)
        
        # Use config default if not specified
        if exclude_from_display is None:
            exclude_from_display = self.config.grouping_exclude_hpm_display_names

        parts = []
        for name, value in hpm_dict.items():
            # Skip if this hpm should be excluded from display
            if name in exclude_from_display:
                continue
                
            display_name = self.get_display_name(name, "hparam")
            formatted_value = self.format_hparam_value(name, value)
            parts.append(f"{display_name}: {formatted_value}")

        return ", ".join(parts)

    def run_group_to_metric_dfs(
        self, run_group: list[Run], metrics: list[str]
    ) -> dict[str, pd.DataFrame]:
        """Convert a group of runs to metric dataframes with one column per run.

        Assumes all runs in a group have the same length for each metric.
        The DataFrame index is just integers (0, 1, 2, ...) - you can use any
        metric (epoch, step, cumulative_lr, etc.) as your x-axis separately.

        Args:
            run_group: List of Run objects (e.g., from active_runs_grouped_by_hpms)
            metrics: List of metric names to extract

        Returns:
            Dict mapping metric_name to DataFrame where:
            - Index: integers from 0 to length-1
            - Columns: seed_N or run_id (one column per run)
            - Values: metric values

        Example:
            >>> group_key, runs = next(
            ...     iter(db.active_runs_grouped_by_hpms()[1].items())
            ... )
            >>> dfs = db.run_group_to_metric_dfs(
            ...     runs, ["train_loss", "val_acc", "epoch"]
            ... )
            >>> print(dfs["train_loss"].head())
                 seed_0    seed_1    seed_2
            0    2.301     2.298     2.305
            1    1.845     1.850     1.843
            2    1.523     1.531     1.519

            >>> # Use any metric as x-axis
            >>> epochs = dfs["epoch"].iloc[:, 0]  # Get epoch values from first run
            >>> plt.plot(epochs, dfs["train_loss"].mean(axis=1))
        """
        result = {}

        for metric in metrics:
            # Collect data for this metric
            metric_data = {}

            for run in run_group:
                if metric not in run.metrics:
                    continue

                # Use seed as column name if available, otherwise run_id
                seed = run.hpms._flat_dict.get("seed")
                col_name = f"seed_{seed}" if seed is not None else run.run_id

                # Get values directly (no index manipulation)
                values = run.metrics[metric]
                metric_data[col_name] = values

            if metric_data:
                # Create DataFrame from dict - pandas will align by position
                df = pd.DataFrame(metric_data)
                result[metric] = df
            else:
                # Empty DataFrame if no data
                result[metric] = pd.DataFrame()

        return result

    def get_grouped_metric_dfs(
        self, metrics: list[str], exclude_hpms: list[str] | None = None
    ) -> dict[tuple, dict[str, pd.DataFrame]]:
        """Get metric dataframes for all hyperparameter groups.

        Combines active_runs_grouped_by_hpms with run_group_to_metric_dfs.

        Args:
            metrics: List of metric names to extract
            exclude_hpms: Hpms to exclude from grouping (see active_runs_grouped_by_hpms)

        Returns:
            Dict mapping group tuples to metric dataframes:
            {
                (batch_size, arch, lr, wd): {
                    'train_loss': DataFrame,
                    'val_acc': DataFrame,
                    ...
                },
                ...
            }

        Example:
            >>> grouped_dfs = db.get_grouped_metric_dfs(["train_loss", "val_acc"])
            >>> for group_key, metric_dfs in grouped_dfs.items():
            >>>     print(f"Group {group_key}:")
            >>>     print(f"  Train loss shape: {metric_dfs['train_loss'].shape}")
        """
        hpm_names, run_groups = self.active_runs_grouped_by_hpms(exclude_hpms)

        result = {}
        for group_key, runs in run_groups.items():
            result[group_key] = self.run_group_to_metric_dfs(runs, metrics)

        return result

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
