"""GroupedRuns class for managing experiment run grouping and filtering operations."""

from __future__ import annotations

from typing import Any

import pandas as pd

from dr_gen.analyze.schemas import Run

if False:  # TYPE_CHECKING workaround for circular imports
    from dr_gen.analyze.database import ExperimentDB


class GroupedRuns:
    """Immutable class representing grouped experiment runs with filtering and data access operations.

    This class encapsulates runs grouped by hyperparameters and provides a clean API for:
    - Filtering groups by hyperparameter values or specific combinations
    - Converting to metric DataFrames for analysis and plotting
    - Describing groups with formatted hyperparameter descriptions

    Operations are immutable - each filtering operation returns a new GroupedRuns instance.
    """

    def __init__(
        self,
        groups: dict[tuple, list[Run]],
        hpm_names: list[str],
        db: ExperimentDB,
    ) -> None:
        """Initialize GroupedRuns.

        Args:
            groups: Dict mapping hyperparameter value tuples to lists of runs
            hpm_names: List of hyperparameter names corresponding to tuple positions
            db: ExperimentDB instance for accessing display names and formatting
        """
        self.groups = groups
        self.hpm_names = hpm_names
        self.db = db

    @classmethod
    def from_db(
        cls, db: ExperimentDB, exclude_hpms: list[str] | None = None
    ) -> GroupedRuns:
        """Create GroupedRuns from ExperimentDB by grouping active runs.

        Args:
            db: ExperimentDB instance
            exclude_hpms: Hyperparameters to exclude from grouping. If None, uses
                         db.config.grouping_exclude_hpms

        Returns:
            New GroupedRuns instance

        Example:
            >>> grouped = GroupedRuns.from_db(db)
            >>> print(f"{len(grouped)} groups found")
        """
        hpm_names, groups = db.active_runs_grouped_by_hpms(exclude_hpms)
        return cls(groups, hpm_names, db)

    def filter(
        self,
        include: dict[str, list[Any]] | None = None,
        exclude: dict[str, list[Any]] | None = None,
        include_pairs: list[tuple] | None = None,
        exclude_pairs: list[tuple] | None = None,
        min_runs: int | None = None,
        max_runs: int | None = None,
    ) -> GroupedRuns:
        """Filter groups based on hyperparameter values and run counts.

        Args:
            include: Dict of hpm_name -> list of values to include
            exclude: Dict of hpm_name -> list of values to exclude
            include_pairs: List of tuples representing specific combinations to include
            exclude_pairs: List of tuples representing specific combinations to exclude
            min_runs: Minimum number of runs required per group (groups with fewer runs are excluded)
            max_runs: Maximum number of runs allowed per group (groups with more runs are excluded)

        Returns:
            New GroupedRuns instance with filtered groups

        Example:
            >>> # Keep only specific learning rates, exclude high weight decay
            >>> filtered = grouped.filter(
            ...     include={"optim.lr": [0.01, 0.03]},
            ...     exclude={"optim.weight_decay": [0.01]},
            ... )
            >>> # Filter to groups with at least 3 runs
            >>> well_sampled = grouped.filter(min_runs=3)
        """
        from dr_gen.analyze.filtering import filter_groups

        # First apply hyperparameter-based filtering
        filtered_groups = filter_groups(
            self.groups, self.hpm_names, include, exclude, include_pairs, exclude_pairs
        )

        # Then apply run count filtering if specified
        if min_runs is not None or max_runs is not None:
            count_filtered_groups = {}
            for group_key, runs in filtered_groups.items():
                num_runs = len(runs)
                if min_runs is not None and num_runs < min_runs:
                    continue
                if max_runs is not None and num_runs > max_runs:
                    continue
                count_filtered_groups[group_key] = runs
            filtered_groups = count_filtered_groups

        return GroupedRuns(filtered_groups, self.hpm_names, self.db)

    def matching(
        self,
        base_hpms: dict[str, Any],
        varying_hpms: list[str] | None = None,
    ) -> GroupedRuns:
        """Filter to groups that match specific base hyperparameters.

        Useful for getting all combinations of specific hyperparameters (e.g., lr, wd)
        while fixing others (e.g., model architecture, batch size).

        Args:
            base_hpms: Dict of hyperparameters that must match exactly
            varying_hpms: List of hyperparameters that are allowed to vary.
                         If None, uses ['optim.lr', 'optim.weight_decay']

        Returns:
            New GroupedRuns instance with groups matching base_hpms,
            grouped by varying_hpms

        Example:
            >>> # Get all (lr, wd) combinations for specific model config
            >>> lr_wd_groups = grouped.matching(
            ...     base_hpms={"model.architecture": "resnet18", "batch_size": 128},
            ...     varying_hpms=["optim.lr", "optim.weight_decay"],
            ... )
        """
        if varying_hpms is None:
            varying_hpms = ["optim.lr", "optim.weight_decay"]

        # First group by all important hpms except seed/run_id
        all_hpm_names, all_groups = self.db.active_runs_grouped_by_hpms()

        # Filter groups that match base_hpms
        filtered_groups = {}
        varying_indices = None

        for group_key, runs in all_groups.items():
            # Convert group key to dict for easier checking
            group_dict = self.db.group_key_to_dict(group_key, all_hpm_names)

            # Check if all base hpms match
            if all(group_dict.get(k) == v for k, v in base_hpms.items()):
                # Extract only the varying hpms for the new key
                if varying_indices is None:
                    # Find indices of varying hpms in the group key
                    varying_indices = [
                        all_hpm_names.index(h)
                        for h in varying_hpms
                        if h in all_hpm_names
                    ]

                new_key = tuple(group_key[i] for i in varying_indices)
                filtered_groups[new_key] = runs

        # Return hpm names that correspond to the keys
        varying_hpm_names = [h for h in varying_hpms if h in all_hpm_names]

        return GroupedRuns(filtered_groups, varying_hpm_names, self.db)

    def to_metric_dfs(self, metrics: list[str]) -> dict[tuple, dict[str, pd.DataFrame]]:
        """Convert groups to metric DataFrames.

        Args:
            metrics: List of metric names to extract

        Returns:
            Dict mapping group keys to metric DataFrames:
            {
                group_key: {
                    'metric_name': DataFrame with columns for each run/seed,
                    ...
                },
                ...
            }

        Example:
            >>> metric_dfs = grouped.to_metric_dfs(["train_loss", "val_acc", "epoch"])
            >>> # Access specific group's train loss
            >>> group_key = list(metric_dfs.keys())[0]
            >>> train_loss_df = metric_dfs[group_key]["train_loss"]
            >>> print(train_loss_df.head())
        """
        return {
            key: self.db.run_group_to_metric_dfs(runs, metrics)
            for key, runs in self.groups.items()
        }

    def describe_groups(
        self, exclude_from_display: list[str] | None = None
    ) -> dict[tuple, str]:
        """Get formatted descriptions for each group.

        Args:
            exclude_from_display: Hyperparameters to exclude from display strings.
                                If None, uses db.config.grouping_exclude_hpm_display_names

        Returns:
            Dict mapping group keys to formatted description strings

        Example:
            >>> descriptions = grouped.describe_groups()
            >>> for key, desc in descriptions.items():
            ...     print(f"{key}: {desc}")
            (0.01, 0.0001): Learning Rate: 1e-02, Weight Decay: 1e-04
        """
        return {
            key: self.db.format_group_description(
                key, self.hpm_names, exclude_from_display
            )
            for key in self.groups.keys()
        }

    def get_plotting_data(
        self, metrics: list[str], exclude_from_display: list[str] | None = None
    ) -> tuple[
        dict[tuple, dict[str, pd.DataFrame]],
        dict[tuple, str],
        dict[tuple, dict[str, Any]],
    ]:
        """Get all data needed for plotting with color/linestyle control.

        This is a convenience method that returns metric DataFrames, group descriptions,
        and hyperparameter dictionaries in one call - everything needed for plot_metric_group().

        Args:
            metrics: List of metric names to extract
            exclude_from_display: Hyperparameters to exclude from display strings

        Returns:
            Tuple of (metric_dfs, group_descriptions, group_hparams) where:
            - metric_dfs: Output of to_metric_dfs()
            - group_descriptions: Output of describe_groups()
            - group_hparams: Output of group_keys_to_dicts()

        Example:
            >>> metric_dfs, descriptions, hparams = grouped.get_plotting_data(
            ...     ["train_loss", "epoch"]
            ... )
            >>> plot_metric_group(
            ...     metric_dfs,
            ...     x_metric="epoch",
            ...     y_metrics="train_loss",
            ...     db=db,
            ...     group_descriptions=descriptions,
            ...     group_hparams=hparams,
            ...     color_by="group",  # Use colors to distinguish groups instead of metrics
            ... )
        """
        metric_dfs = self.to_metric_dfs(metrics)
        group_descriptions = self.describe_groups(exclude_from_display)
        group_hparams = self.group_keys_to_dicts(use_display_names=False)

        return metric_dfs, group_descriptions, group_hparams

    def group_keys_to_dicts(
        self, use_display_names: bool = False
    ) -> dict[tuple, dict[str, Any]]:
        """Convert group keys to hyperparameter dictionaries.

        Args:
            use_display_names: If True, use display names as dict keys

        Returns:
            Dict mapping group keys to hyperparameter dictionaries

        Example:
            >>> hpm_dicts = grouped.group_keys_to_dicts()
            >>> # Technical names
            >>> print(
            ...     hmp_dicts[(0.01, 0.0001)]
            ... )  # {'optim.lr': 0.01, 'optim.weight_decay': 0.0001}
            >>> # Display names
            >>> display_dicts = grouped.group_keys_to_dicts(use_display_names=True)
            >>> print(
            ...     display_dicts[(0.01, 0.0001)]
            ... )  # {'Learning Rate': 0.01, 'Weight Decay': 0.0001}
        """
        return {
            key: self.db.group_key_to_dict(key, self.hpm_names, use_display_names)
            for key in self.groups.keys()
        }

    def get_group_info(self) -> dict[str, Any]:
        """Get summary information about the grouped runs.

        Returns:
            Dict with summary statistics about the groups
        """
        total_runs = sum(len(runs) for runs in self.groups.values())
        runs_per_group = [len(runs) for runs in self.groups.values()]

        return {
            "num_groups": len(self.groups),
            "total_runs": total_runs,
            "hpm_names": self.hpm_names.copy(),
            "runs_per_group": {
                "min": min(runs_per_group) if runs_per_group else 0,
                "max": max(runs_per_group) if runs_per_group else 0,
                "mean": total_runs / len(self.groups) if self.groups else 0,
            },
        }

    def __len__(self) -> int:
        """Return the number of groups."""
        return len(self.groups)

    def __iter__(self):
        """Iterate over (group_key, runs) pairs."""
        return iter(self.groups.items())

    def __repr__(self) -> str:
        """Return string representation of the GroupedRuns."""
        total_runs = sum(len(runs) for runs in self.groups.values())
        hpm_str = (
            ", ".join(self.hpm_names)
            if len(self.hpm_names) <= 3
            else f"{len(self.hpm_names)} hpms"
        )
        return f"GroupedRuns({len(self.groups)} groups, {total_runs} total runs, grouped by: {hpm_str})"

    def __bool__(self) -> bool:
        """Return True if there are any groups."""
        return bool(self.groups)
