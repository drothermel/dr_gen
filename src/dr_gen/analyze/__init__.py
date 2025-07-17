"""Analyze package for dr_gen - utilities for analyzing training runs."""

from dr_gen.analyze.database import ExperimentDB
from dr_gen.analyze.filtering import filter_groups, filter_groups_interactive
from dr_gen.analyze.schemas import AnalysisConfig, Hpms, Run
from dr_gen.analyze.visualization import plot_metric_group, plot_training_metrics


def check_prefix_exclude(check_string: str, excluded_prefixes: list[str]) -> bool:
    """Check if string starts with any of the excluded prefixes.

    Args:
        check_string: String to check
        excluded_prefixes: List of prefix strings to check against

    Returns:
        True if check_string starts with any excluded prefix, False otherwise
    """
    return any(check_string.startswith(pre) for pre in excluded_prefixes)


__all__ = [
    "AnalysisConfig",
    "ExperimentDB",
    "Hpms",
    "Run",
    "filter_groups",
    "filter_groups_interactive",
    "plot_metric_group",
    "plot_training_metrics",
]
