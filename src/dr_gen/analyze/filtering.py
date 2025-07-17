"""Filtering utilities for experiment analysis."""

from typing import Any


def filter_groups(
    groups: dict[tuple, Any],
    hpm_names: list[str],
    include: dict[str, list[Any]] | None = None,
    exclude: dict[str, list[Any]] | None = None,
    include_pairs: list[tuple] | None = None,
    exclude_pairs: list[tuple] | None = None,
) -> dict[tuple, Any]:
    """Filter groups based on hyperparameter values.

    Provides flexible filtering with include/exclude logic for individual
    hyperparameters and specific combinations.

    Args:
        groups: Dict mapping hyperparameter tuples to data (runs or metric dfs)
        hpm_names: List of hyperparameter names corresponding to tuple positions
        include: Dict of hpm_name -> list of values to include
        exclude: Dict of hpm_name -> list of values to exclude
        include_pairs: List of tuples representing specific combinations to include
        exclude_pairs: List of tuples representing specific combinations to exclude

    Returns:
        Filtered groups dict with same structure as input

    Example:
        >>> # Filter to only lr in [0.01, 0.1] and exclude wd=0.01
        >>> filtered = filter_groups(
        ...     groups,
        ...     hpm_names=["optim.lr", "optim.weight_decay"],
        ...     include={"optim.lr": [0.01, 0.1]},
        ...     exclude={"optim.weight_decay": [0.01]},
        ... )

        >>> # Exclude specific (lr, wd) pairs
        >>> filtered = filter_groups(
        ...     groups,
        ...     hpm_names=["optim.lr", "optim.weight_decay"],
        ...     exclude_pairs=[(0.3, 0.001), (1.0, 0.0001)],
        ... )
    """
    filtered = {}

    # Create index mapping for quick lookup
    hpm_indices = {name: i for i, name in enumerate(hpm_names)}

    for group_key, data in groups.items():
        # Check exclude_pairs first (most specific)
        if exclude_pairs and group_key in exclude_pairs:
            continue

        # Check include_pairs
        if include_pairs and group_key not in include_pairs:
            continue

        # Check individual hpm filters
        include_match = True
        exclude_match = False

        # Check include filters
        if include:
            for hpm_name, allowed_values in include.items():
                if hpm_name in hpm_indices:
                    idx = hpm_indices[hpm_name]
                    if group_key[idx] not in allowed_values:
                        include_match = False
                        break

        # Check exclude filters
        if exclude and include_match:
            for hpm_name, excluded_values in exclude.items():
                if hpm_name in hpm_indices:
                    idx = hpm_indices[hpm_name]
                    if group_key[idx] in excluded_values:
                        exclude_match = True
                        break

        # Add to filtered if passes all checks
        if include_match and not exclude_match:
            filtered[group_key] = data

    return filtered


def filter_groups_interactive(
    groups: dict[tuple, Any],
    hpm_names: list[str],
    initial_include: dict[str, list[Any]] | None = None,
    initial_exclude: dict[str, list[Any]] | None = None,
) -> tuple[dict[tuple, Any], dict[str, list[Any]], dict[str, list[Any]]]:
    """Interactive filtering helper that prints current state and returns filters.

    Useful for iterative filtering in notebooks where you want to see what's
    currently included and easily modify filters.

    Args:
        groups: Dict mapping hyperparameter tuples to data
        hpm_names: List of hyperparameter names
        initial_include: Starting include filters
        initial_exclude: Starting exclude filters

    Returns:
        Tuple of (filtered_groups, include_filters, exclude_filters)
    """
    include = initial_include or {}
    exclude = initial_exclude or {}

    # Apply filters
    filtered = filter_groups(groups, hpm_names, include, exclude)

    # Print current state
    print(f"Filtered groups: {len(filtered)} / {len(groups)}")

    # Show current hyperparameter values
    for i, hpm_name in enumerate(hpm_names):
        values = sorted(set(key[i] for key in filtered.keys()))
        print(f"\n{hpm_name}:")
        print(f"  Current values: {values}")
        if hpm_name in include:
            print(f"  Include filter: {include[hpm_name]}")
        if hpm_name in exclude:
            print(f"  Exclude filter: {exclude[hpm_name]}")

    return filtered, include, exclude
