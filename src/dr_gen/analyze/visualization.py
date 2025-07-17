"""Simplified plotting module for dr_gen analysis.

Replaces plot_utils.py + common_plots.py with direct matplotlib usage.
Eliminates overengineering while maintaining all core functionality.
"""

import random
from collections.abc import Callable
from typing import Any, TypedDict, Unpack

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats

from dr_gen.analyze.database import ExperimentDB

# Colorblind-safe color palettes for scientific publications
COLORBLIND_COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "yellow": "#F0E442",
    "bluish_green": "#56B4E9",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "black": "#000000",
}

# Paul Tol's colorblind-safe qualitative palette
PAUL_TOL_COLORS = [
    "#332288",
    "#88CCEE",
    "#44AA99",
    "#117733",
    "#999933",
    "#DDCC77",
    "#CC6677",
    "#882255",
    "#AA4499",
]

# Colorblind-safe palette for categorical data (up to 8 categories)
CATEGORICAL_COLORS = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#56B4E9",
    "#D55E00",
    "#CC79A7",
    "#F0E442",
    "#999999",
]


def get_color_palette(n_colors: int = 8, palette: str = "colorblind") -> list[str]:
    """Get a colorblind-safe color palette.

    Args:
        n_colors: Number of colors needed
        palette: Palette name ('colorblind', 'paul_tol', 'categorical')

    Returns:
        List of hex color codes
    """
    if palette == "colorblind":
        colors = list(COLORBLIND_COLORS.values())
    elif palette == "paul_tol":
        colors = PAUL_TOL_COLORS
    elif palette == "categorical":
        colors = CATEGORICAL_COLORS
    else:
        # Default matplotlib tab10
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i) for i in range(10)]

    # Cycle colors if we need more than available
    return (
        colors[:n_colors]
        if n_colors <= len(colors)
        else (colors * (n_colors // len(colors) + 1))[:n_colors]
    )


def plot_metric_group(
    metric_dfs: dict[str, pd.DataFrame] | dict[Any, dict[str, pd.DataFrame]],
    x_metric: str = "epoch",
    y_metrics: str | list[str] = "train_loss",
    db: ExperimentDB | None = None,
    group_descriptions: str | dict[Any, str] | None = None,
    figsize: tuple[float, float] = (8, 5),
    show_individual_runs: bool = True,
    individual_alpha: float = 0.25,
    individual_linewidth: float = 0.8,
    color_scheme: str = "colorblind",  # Color palette name
    mean_linewidth: float = 2.5,
    std_band: bool = True,
    std_alpha: float = 0.15,
    grid: bool = True,
    grid_alpha: float = 0.2,
    legend: bool = True,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    xscale: str = "linear",
    yscale: str = "linear",
    return_fig: bool = False,
    publication_style: bool = True,
) -> plt.Figure | None:
    """Plot metrics for one or more groups of runs with mean and standard deviation.

    Uses colorblind-safe colors to distinguish metrics and line styles to distinguish groups.
    Individual runs and std band use the same color/style as the mean with different alpha.

    Args:
        metric_dfs: Either:
            - Dict of metric DataFrames (single group)
            - Dict of dicts where keys are group IDs and values are metric DataFrames
        x_metric: Name of metric to use for x-axis
        y_metrics: Single metric name or list of metric names for y-axis
        db: ExperimentDB instance for display name lookups (optional)
        group_descriptions: Either:
            - String description for single group
            - Dict mapping group IDs to descriptions
            - None to auto-generate or omit
        figsize: Figure size tuple
        show_individual_runs: Whether to plot individual run traces
        individual_alpha: Transparency for individual runs
        individual_linewidth: Line width for individual runs
        color_scheme: Name of color palette to use ('colorblind', 'paul_tol', 'categorical')
        mean_linewidth: Line width for mean line
        std_band: Whether to show standard deviation band
        std_alpha: Transparency for std band
        grid: Whether to show grid
        grid_alpha: Grid transparency
        legend: Whether to show legend
        xlabel: Override x-axis label (uses display name if None)
        ylabel: Override y-axis label (uses display name if None)
        title: Override title (auto-generated if None)
        xlim: X-axis limits
        ylim: Y-axis limits
        xscale: X-axis scale ('linear', 'log', etc.)
        yscale: Y-axis scale ('linear', 'log', etc.)
        return_fig: If True, return the figure instead of showing it
        publication_style: If True, apply publication-quality styling

    Returns:
        Figure if return_fig=True, otherwise None

    Example:
        >>> # Single group, single metric
        >>> metric_dfs = db.run_group_to_metric_dfs(runs, ["epoch", "train_loss"])
        >>> plot_metric_group(
        ...     metric_dfs,
        ...     x_metric="epoch",
        ...     y_metrics="train_loss",
        ...     db=db,
        ...     group_descriptions="LR=0.01, WD=1e-4",
        ... )

        >>> # Single group, multiple metrics
        >>> plot_metric_group(
        ...     metric_dfs,
        ...     x_metric="epoch",
        ...     y_metrics=["train_loss", "val_loss"],
        ...     db=db,
        ... )

        >>> # Multiple groups, single metric
        >>> all_groups = {}
        >>> for group_key, runs in run_groups.items():
        ...     all_groups[group_key] = db.run_group_to_metric_dfs(runs, metrics)
        >>> plot_metric_group(
        ...     all_groups,
        ...     x_metric="epoch",
        ...     y_metrics="train_loss",
        ...     db=db,
        ...     group_descriptions={
        ...         group_key: db.format_group_description(group_key)
        ...         for group_key in all_groups
        ...     },
        ... )
    """
    # Ensure y_metrics is a list
    if isinstance(y_metrics, str):
        y_metrics = [y_metrics]

    # Detect if we have single or multiple groups
    first_value = next(iter(metric_dfs.values()))
    is_single_group = isinstance(first_value, pd.DataFrame)

    # Normalize to dict of groups format
    if is_single_group:
        # Single group case - wrap in a dict
        groups = {"_single": metric_dfs}
        if isinstance(group_descriptions, str):
            group_desc_dict = {"_single": group_descriptions}
        else:
            group_desc_dict = {"_single": ""}
    else:
        # Multiple groups case
        groups = metric_dfs
        if isinstance(group_descriptions, dict):
            group_desc_dict = group_descriptions
        elif isinstance(group_descriptions, str):
            # If single string provided for multiple groups, use for all
            group_desc_dict = {k: group_descriptions for k in groups.keys()}
        else:
            # No descriptions provided
            group_desc_dict = {k: f"Group {k}" for k in groups.keys()}

    # Validate metrics exist in all groups
    for group_id, group_dfs in groups.items():
        if x_metric not in group_dfs:
            raise ValueError(f"x_metric '{x_metric}' not found in group '{group_id}'")
        for y_metric in y_metrics:
            if y_metric not in group_dfs:
                raise ValueError(
                    f"y_metric '{y_metric}' not found in group '{group_id}'"
                )

    # Get color palette for the metrics
    colors = get_color_palette(len(y_metrics), palette=color_scheme)

    # Define line styles for different groups
    line_styles = ["-", "--", ":", "-.", (0, (3, 1, 1, 1)), (0, (5, 2)), (0, (1, 1))]

    # Create figure with publication styling
    if publication_style:
        plt.rcParams.update(
            {
                "font.size": 11,
                "axes.labelsize": 12,
                "axes.titlesize": 14,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "legend.fontsize": 10,
                "figure.titlesize": 14,
                "axes.spines.top": False,
                "axes.spines.right": False,
            }
        )

    fig, ax = plt.subplots(figsize=figsize)

    # Get x-axis values from first group (all groups should have same x values)
    first_group = next(iter(groups.values()))
    x_values = first_group[x_metric].iloc[:, 0].values

    # Plot each group and metric combination
    for group_idx, (group_id, group_dfs) in enumerate(groups.items()):
        # Get line style for this group
        line_style = line_styles[group_idx % len(line_styles)]
        group_desc = group_desc_dict.get(group_id, "")

        # Plot each metric for this group
        for metric_idx, y_metric in enumerate(y_metrics):
            metric_color = colors[metric_idx]

            # Get display name for metric
            if db is not None:
                metric_display_name = db.get_display_name(y_metric)
            else:
                metric_display_name = y_metric

            # Plot individual runs if requested
            if show_individual_runs:
                for col in group_dfs[y_metric].columns:
                    ax.plot(
                        x_values,
                        group_dfs[y_metric][col],
                        color=metric_color,
                        alpha=individual_alpha,
                        linewidth=individual_linewidth,
                        linestyle=line_style,
                    )

            # Calculate and plot mean
            mean_y = group_dfs[y_metric].mean(axis=1)

            # Create label for legend
            if is_single_group:
                # Single group - just show metric name
                label = metric_display_name
            elif len(y_metrics) == 1:
                # Multiple groups, single metric - just show group
                label = group_desc if group_desc else f"Group {group_id}"
            else:
                # Multiple groups and metrics - show both
                group_label = group_desc if group_desc else f"Group {group_id}"
                label = f"{metric_display_name} ({group_label})"

            ax.plot(
                x_values,
                mean_y,
                color=metric_color,
                linewidth=mean_linewidth,
                linestyle=line_style,
                label=label,
            )

            # Add standard deviation band if requested
            if std_band:
                std_y = group_dfs[y_metric].std(axis=1)
                ax.fill_between(
                    x_values,
                    mean_y - std_y,
                    mean_y + std_y,
                    alpha=std_alpha,
                    color=metric_color,
                )

    # Set labels with display names if available
    if xlabel is None and db is not None:
        xlabel = db.get_display_name(x_metric)
    elif xlabel is None:
        xlabel = x_metric
    ax.set_xlabel(xlabel)

    if ylabel is None:
        # For multiple metrics, use a generic label or combine metric names
        if len(y_metrics) == 1:
            if db is not None:
                ylabel = db.get_display_name(y_metrics[0])
            else:
                ylabel = y_metrics[0]
        else:
            ylabel = "Value"  # Generic label for multiple metrics
    ax.set_ylabel(ylabel)

    # Set title
    if title is None:
        # Generate title based on what's being plotted
        if is_single_group:
            # Single group
            group_desc = group_desc_dict.get("_single", "")
            if len(y_metrics) == 1:
                # Single metric, single group
                if db is not None:
                    y_display = db.get_display_name(y_metrics[0])
                else:
                    y_display = y_metrics[0]

                if group_desc:
                    title = f"{y_display} | {group_desc}"
                else:
                    title = y_display
            else:
                # Multiple metrics, single group
                if group_desc:
                    title = f"Metrics | {group_desc}"
                else:
                    title = "Metrics Comparison"
        else:
            # Multiple groups
            if len(y_metrics) == 1:
                # Single metric, multiple groups
                if db is not None:
                    y_display = db.get_display_name(y_metrics[0])
                else:
                    y_display = y_metrics[0]
                title = f"{y_display} Comparison"
            else:
                # Multiple metrics, multiple groups
                title = "Multi-Group Metrics Comparison"
    ax.set_title(title)

    # Apply axis limits and scales
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)

    # Add grid if requested
    if grid:
        ax.grid(alpha=grid_alpha)

    # Add legend if requested
    if legend:
        ax.legend()

    # Show or return figure
    if return_fig:
        return fig
    plt.tight_layout()
    plt.show()
    return None


# NOTE: These functions were part of the dataframe-based approach using group_metric_by_hpms_v2
# They are being kept for backward compatibility but are deprecated in favor of
# the direct Run object approach using db.run_group_to_metric_dfs()

# def prepare_data_for_plotting(
#     grouped_results: Dict[tuple, pd.DataFrame],
#     metric_name: str,
#     db: ExperimentDB,
#     aggregate_seeds: bool = True
# ) -> pd.DataFrame:
#     """
#     Prepare the grouped results for use with plot_training_metrics.
#
#     Args:
#         grouped_results: Output from group_metric_by_hpms_v2
#         metric_name: Name of the metric being plotted
#         db: ExperimentDB instance (to access important_hpms)
#         aggregate_seeds: If True, average across seeds; if False, include seed as a column
#
#     Returns:
#         DataFrame formatted for plot_training_metrics with columns:
#         epoch, lr, wd, [metric_name], and optionally seed
#     """
#     all_dfs = []
#
#     for group_key, df in grouped_results.items():
#         # Extract lr and wd from the group key
#         # Find indices for lr and wd in the group key
#         group_keys = [h for h in db.important_hpms if h not in ['seed', 'run_id']]
#         lr_idx = group_keys.index('optim.lr')
#         wd_idx = group_keys.index('optim.weight_decay')
#
#         lr = group_key[lr_idx]
#         wd = group_key[wd_idx]
#
#         # Add lr and wd columns
#         df = df.copy()
#         df['run_lr'] = lr
#         df['run_wd'] = wd
#
#         if aggregate_seeds:
#             # Average across seeds
#             agg_df = df.groupby(['epoch', 'run_lr', 'run_wd'])[metric_name].mean().reset_index()
#             all_dfs.append(agg_df)
#         else:
#             all_dfs.append(df)
#
#     # Combine all dataframes
#     combined_df = pd.concat(all_dfs, ignore_index=True)
#     return combined_df

# def prep_all_data(
#     db: ExperimentDB,
#     metrics: List[str],
#     hpm_filters: Dict[str, Any],
#     aggregate_seeds: bool = True,
# ) -> None:
#     # Prepare data for each metric
#     metric_dfs = {}
#     for metric in metrics:
#         grouped = group_metric_by_hpms_v2(db, metric, **hpm_filters)
#         if not grouped:
#             print(f"No data found for metric: {metric}")
#             continue
#         metric_dfs[metric] = prepare_data_for_plotting(grouped, metric, db, aggregate_seeds)
#
#     if not metric_dfs:
#         print("No data to plot")
#         return
#
#     # Combine all metric dataframes
#     # We need to merge them on epoch, lr, wd
#     combined_df = None
#     for metric, df in metric_dfs.items():
#         if combined_df is None:
#             combined_df = df
#         else:
#             # Merge on common columns
#             merge_cols = ['epoch', 'run_lr', 'run_wd']
#             if 'seed' in df.columns and 'seed' in combined_df.columns:
#                 merge_cols.append('seed')
#             combined_df = combined_df.merge(df, on=merge_cols, how='outer')
#
#     return combined_df


def plot_training_metrics(
    df: pd.DataFrame,
    title: str = "",
    xlog: bool = False,
    ylog: bool = False,
    xrange: tuple[float, float] | None = None,
    yrange_train_loss: tuple[float, float] | None = None,
    yrange_val_loss: tuple[float, float] | None = None,
    yrange_acc: tuple[float, float] | None = None,
    figsize: tuple[float, float] = (17.0, 4.2),
    leave_space_for_legend: float = 0.1,
    df_std: pd.DataFrame | None = None,
    hbar2: float = 50,
) -> None:
    """Draw the *four* panels (train‑loss, val‑loss, train‑acc, val‑acc) with
    colour = learning‑rate and linestyle = weight‑decay.  The mapping is derived
    automatically from whatever unique values appear in the dataframe.

    Parameters
    ----------
    df : tidy dataframe with at least the columns produced by
         `make_mock_cifar_df`.
    df_std : dataframe with the same structure as df, containing std dev for each metric.
    """
    # ----- discover the hyper‑parameter palette / style -----------------
    lrs = sorted(df["run_lr"].unique())  # e.g. [0.01, 0.03, 0.1]
    wds = sorted(df["run_wd"].unique())  # e.g. [1e‑4, 3e‑4, 1e‑3]

    # colours – reuse tab10 but subsample in case there are more than 10 lrs
    cmap = plt.get_cmap("tab10")
    colour_map = {lr: cmap(i % 10) for i, lr in enumerate(lrs)}

    # linestyles – cycle through the classic trio, then fallback to dash‑dot etc.
    style_cycle = ["solid", "dashed", "dotted", "dashdot", (0, (3, 1, 1, 1))]
    style_map = {wd: style_cycle[i % len(style_cycle)] for i, wd in enumerate(wds)}

    # ----- set‑up the 1×4 sub‑plots ------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=figsize, sharex=False)
    fig.suptitle(title, fontsize=16, fontweight="bold")

    panels = [
        ("train_loss", "Train Loss", ylog),
        ("val_loss", "Val Loss", ylog),
        ("train_acc", "Train Acc", False),
        ("val_acc", "Val Acc", False),
    ]

    # axes is a (2,2) array, so we need to flatten it to iterate over 4 panels
    for (metric, panel_title, met_ylog), ax in zip(panels, axes.flat, strict=False):
        for lr in lrs:
            for wd in wds:
                subset = df[(df.run_lr == lr) & (df.run_wd == wd)]
                ax.plot(
                    subset["epoch"],
                    subset[metric],
                    label=f"lr={lr}, wd={wd}",
                    color=colour_map[lr],
                    linestyle=style_map[wd],
                    linewidth=1.8,
                )
                # max_val_acc_epoch = subset[subset['val_acc'] == subset['val_acc'].max()]['epoch'].values[0]
                # ax.axvline(x=max_val_acc_epoch, color=colour_map[lr], linestyle='-', linewidth=1.2, alpha=0.7)

                train_loss_below_value = subset[subset["train_loss"] < 0.01][
                    "epoch"
                ].values[0]
                ax.axvline(
                    x=train_loss_below_value,
                    color=colour_map[lr],
                    linestyle="--",
                    linewidth=2,
                    alpha=0.7,
                )

                # If std dev dataframe is provided, plot shaded region for mean ± std
                if df_std is not None and metric in df_std.columns:
                    subset_std = df_std[(df_std.lr == lr) & (df_std.wd == wd)]
                    # Only plot if both mean and std have matching epochs
                    if not subset_std.empty and not subset.empty:
                        # Align on epoch
                        merged = pd.merge(
                            subset[["epoch", metric]],
                            subset_std[["epoch", metric]],
                            on="epoch",
                            suffixes=("_mean", "_std"),
                        )
                        if not merged.empty:
                            mean = merged[f"{metric}_mean"]
                            std = merged[f"{metric}_std"]
                            epochs = merged["epoch"]
                            ax.fill_between(
                                epochs,
                                mean - std,
                                mean + std,
                                color=colour_map[lr],
                                alpha=0.18,
                                linewidth=0,
                                zorder=0,
                            )

        ax.axvline(x=5, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)
        # ax.axvline(x=hbar2, color='gray', linestyle='--', linewidth=1.2, alpha=0.7)

        ax.set_title(panel_title)
        ax.grid(alpha=0.3, which="both", linestyle=":")
        if metric == "train_loss" and yrange_train_loss is not None:
            ax.set_ylim(yrange_train_loss)
        if metric == "val_loss" and yrange_val_loss is not None:
            ax.set_ylim(yrange_val_loss)
        elif (metric == "train_acc" or metric == "val_acc") and yrange_acc is not None:
            ax.set_ylim(yrange_acc)
        if xrange is not None:
            ax.set_xlim(xrange)
        if xlog:
            ax.set_xscale("log")
            ax.set_xlabel("Epoch (log scale)")
        else:
            ax.set_xlabel("Epoch")
        if met_ylog:
            ax.set_yscale("log")
            ax.set_ylabel(f"{metric} (log scale)")
        else:
            ax.set_ylabel(f"{metric}")

    # -------------------  Legend (row-per-lr layout)  --------------------

    legend_title = "Learning Rate (color) × Weight Decay (style)"

    handles, labels = [], []

    for lr in lrs:
        # first column of the row: lr label (no visible line)
        handles.append(Line2D([], [], color="none", label=f"lr={lr}:"))
        labels.append(f"lr={lr}:")
        # remaining columns: one entry per wd, styled correctly
        for wd in wds:
            handles.append(
                Line2D(
                    [],
                    [],
                    color=colour_map[lr],
                    linestyle=style_map[wd],
                    linewidth=2,
                    label=f"wd={wd}",
                )
            )
            labels.append(f"wd={wd}")

    ncol = len(lrs)

    leg = fig.legend(
        handles,
        labels,
        ncol=ncol,
        loc="lower center",
        bbox_to_anchor=(0.5, -leave_space_for_legend),
        frameon=False,
        columnspacing=1.5,
        handletextpad=0.6,
    )

    """
        ax.axvline(x=5, color='gray', linestyle='--', linewidth=1.2, alpha=0.7)

        ax.set_title(title)
        ax.grid(alpha=.3, which='both', linestyle=':')
        if (metric == 'train_loss' or metric == 'val_loss') and yrange_loss is not None:
            ax.set_ylim(yrange_loss)
        elif (metric == 'train_acc' or metric == 'val_acc') and yrange_acc is not None:
            ax.set_ylim(yrange_acc)
        if xrange is not None:
            ax.set_xlim(xrange)
        if xlog:
            ax.set_xscale('log')
            ax.set_xlabel('Epoch (log scale)')
        else:
            ax.set_xlabel('Epoch')
        if met_ylog:
            ax.set_yscale('log')
            ax.set_ylabel(f'{metric} (log scale)')
        else:
            ax.set_ylabel(f'{metric}')

    # -------------------  Legend (row-per-lr layout)  --------------------

    title = 'Learning Rate (color) × Weight Decay (style)'

    handles, labels = [], []

    for lr in lrs:
        # first column of the row: lr label (no visible line)
        handles.append(Line2D([], [], color='none', label=f'lr={lr}:'))
        labels.append(f'lr={lr}:')
        # remaining columns: one entry per wd, styled correctly
        for wd in wds:
            handles.append(Line2D([], [], color=colour_map[lr],
                                linestyle=style_map[wd], linewidth=2,
                                label=f'wd={wd}'))
            labels.append(f'wd={wd}')

    # ncol = (# wds + 1)   →  exactly one row per lr group
    #ncol = len(wds) + 1
    ncol = len(lrs)

    leg = fig.legend(handles, labels,
                    ncol=ncol,
                    loc='lower center',
                    bbox_to_anchor=(0.5, -leave_space_for_legend),
                    frameon=False,
                    columnspacing=1.5,
                    handletextpad=0.6)

    """
    # add a bold title and subtitle (two separate lines)
    leg.set_title(f"{title}", prop={"weight": "bold", "size": "medium"})

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # leave room beneath the plots
    plt.show()


# ------------------------- OLD CODE -------------------------


class PlotStyle(TypedDict, total=False):
    """Type definition for plot styling options."""

    figsize: tuple[int, int]
    alpha: float
    linewidth: float
    grid: bool
    legend: bool
    bins: int
    density: bool
    title: str
    xlabel: str
    ylabel: str
    suptitle: str


# Default styling
DEFAULT_STYLE = {
    "figsize": (8, 6),
    "alpha": 0.7,
    "linewidth": 2,
    "grid": True,
    "legend": False,
    "bins": 50,
    "density": False,
}


def merge_defaults(**kwargs: Unpack[PlotStyle]) -> PlotStyle:
    """Merge user kwargs with defaults."""
    return {**DEFAULT_STYLE, **kwargs}  # type: ignore[typeddict-item]


def sample_data(data: list[Any], n: int | None = None) -> list[Any]:
    """Randomly sample n items from data."""
    if n is None or n >= len(data):
        return data
    return random.sample(data, n)


def plot_lines(
    curves: list[float] | list[list[float]],
    sample: int | None = None,
    ax: plt.Axes | None = None,
    **kwargs: Unpack[PlotStyle],
) -> plt.Figure:
    """Plot line curves - single curve or multiple curves."""
    style = merge_defaults(**kwargs)

    if ax is None:
        fig, ax = plt.subplots(figsize=style["figsize"])
    else:
        fig = ax.figure

    # Handle single curve vs multiple curves
    if isinstance(curves[0], int | float | np.number):
        # Single curve
        ax.plot(curves, linewidth=style["linewidth"], alpha=style["alpha"])
    else:
        # Multiple curves
        curve_list = sample_data(curves, sample) if sample else curves
        colors = plt.cm.tab10(np.linspace(0, 1, len(curve_list)))

        for i, curve in enumerate(curve_list):
            ax.plot(
                curve,
                color=colors[i],
                linewidth=style["linewidth"],
                alpha=style["alpha"],
                label=f"Curve {i}" if style["legend"] else None,
            )

    if style["grid"]:
        ax.grid(alpha=0.3)
    if style["legend"]:
        ax.legend()
    if style.get("title"):
        ax.set_title(style["title"])
    if style.get("xlabel"):
        ax.set_xlabel(style["xlabel"])
    if style.get("ylabel"):
        ax.set_ylabel(style["ylabel"])

    if ax is None:  # Created new figure
        plt.tight_layout()
        plt.show()

    return fig


def plot_histogram(
    vals: list[float] | list[list[float]],
    ax: plt.Axes | None = None,
    **kwargs: Unpack[PlotStyle],
) -> plt.Figure:
    """Plot histogram - single or multiple distributions."""
    style = merge_defaults(**kwargs)

    if ax is None:
        fig, ax = plt.subplots(figsize=style["figsize"])
    else:
        fig = ax.figure

    # Handle single vs multiple distributions
    if isinstance(vals[0], int | float | np.number):
        # Single distribution
        ax.hist(
            vals, bins=style["bins"], alpha=style["alpha"], density=style["density"]
        )
    else:
        # Multiple distributions
        colors = plt.cm.tab10(np.linspace(0, 1, len(vals)))

        for i, val_list in enumerate(vals):
            ax.hist(
                val_list,
                bins=style["bins"],
                alpha=style["alpha"],
                density=style["density"],
                color=colors[i],
                label=f"Dist {i}" if style["legend"] else None,
            )

    if style["grid"]:
        ax.grid(alpha=0.3)
    if style["legend"]:
        ax.legend()
    if style.get("title"):
        ax.set_title(style["title"])
    if style.get("xlabel"):
        ax.set_xlabel(style["xlabel"])
    if style.get("ylabel"):
        ax.set_ylabel(style["ylabel"])

    if ax is None:
        plt.tight_layout()
        plt.show()

    return fig


def plot_cdf(
    vals1: list[float],
    vals2: list[float],
    ax: plt.Axes | None = None,
    **kwargs: Unpack[PlotStyle],
) -> plt.Figure:
    """Plot comparative CDFs with KS statistics."""
    style = merge_defaults(**kwargs)

    if ax is None:
        fig, ax = plt.subplots(figsize=style["figsize"])
    else:
        fig = ax.figure

    # Compute empirical CDFs
    x1_sorted = np.sort(vals1)
    x2_sorted = np.sort(vals2)
    y1 = np.arange(1, len(x1_sorted) + 1) / len(x1_sorted)
    y2 = np.arange(1, len(x2_sorted) + 1) / len(x2_sorted)

    ax.plot(x1_sorted, y1, linewidth=style["linewidth"], label="Distribution 1")
    ax.plot(x2_sorted, y2, linewidth=style["linewidth"], label="Distribution 2")

    # Add KS statistic
    ks_stat, p_value = stats.ks_2samp(vals1, vals2)
    ax.text(
        0.05,
        0.95,
        f"KS stat: {ks_stat:.3f}\np-value: {p_value:.3f}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )

    if style["grid"]:
        ax.grid(alpha=0.3)
    ax.legend()
    ax.set_xlabel("Value")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("Empirical CDFs")

    if ax is None:
        plt.tight_layout()
        plt.show()

    return fig


def plot_splits(
    split_data: list[list[list[float]]],
    splits: list[str] | None = None,
    ax: plt.Axes | None = None,
    **kwargs: Unpack[PlotStyle],
) -> plt.Figure:
    """Plot training/validation/eval splits with different line styles."""
    if splits is None:
        splits = ["train", "val"]
    style = merge_defaults(**kwargs)

    if ax is None:
        fig, ax = plt.subplots(figsize=style["figsize"])
    else:
        fig = ax.figure

    linestyles = ["-", "--", ":", "-."]
    colors = plt.cm.tab10(np.linspace(0, 1, len(splits)))

    for i, (split_curves, split_name) in enumerate(
        zip(split_data, splits, strict=False)
    ):
        color = colors[i]
        linestyle = linestyles[i % len(linestyles)]

        for curve in split_curves:
            ax.plot(
                curve,
                color=color,
                linestyle=linestyle,
                linewidth=style["linewidth"],
                alpha=style["alpha"],
            )

        # Add one curve for legend
        ax.plot(
            [],
            [],
            color=color,
            linestyle=linestyle,
            linewidth=style["linewidth"],
            label=split_name.capitalize(),
        )

    if style["grid"]:
        ax.grid(alpha=0.3)
    ax.legend()
    if style.get("title"):
        ax.set_title(style["title"])
    if style.get("xlabel"):
        ax.set_xlabel(style["xlabel"])
    if style.get("ylabel"):
        ax.set_ylabel(style["ylabel"])

    if ax is None:
        plt.tight_layout()
        plt.show()

    return fig


def plot_summary(
    curves_list: list[list[float]],
    uncertainty: str = "std",
    ax: plt.Axes | None = None,
    **kwargs: Unpack[PlotStyle],
) -> plt.Figure:
    """Plot mean curve with uncertainty bands (std, sem, or minmax)."""
    style = merge_defaults(**kwargs)

    if ax is None:
        fig, ax = plt.subplots(figsize=style["figsize"])
    else:
        fig = ax.figure

    # Convert to DataFrame for easy statistics
    data_frame = pd.DataFrame(curves_list).T
    mean_curve = data_frame.mean(axis=1)

    # Plot mean line
    ax.plot(mean_curve, linewidth=style["linewidth"] * 1.5, color="blue", label="Mean")

    # Add uncertainty bands
    if uncertainty == "std":
        std_curve = data_frame.std(axis=1)
        ax.fill_between(
            range(len(mean_curve)),
            mean_curve - std_curve,
            mean_curve + std_curve,
            alpha=0.3,
            color="blue",
            label="±1 std",
        )
    elif uncertainty == "sem":
        sem_curve = data_frame.sem(axis=1)
        ax.fill_between(
            range(len(mean_curve)),
            mean_curve - sem_curve,
            mean_curve + sem_curve,
            alpha=0.3,
            color="blue",
            label="±1 SEM",
        )
    elif uncertainty == "minmax":
        min_curve = data_frame.min(axis=1)
        max_curve = data_frame.max(axis=1)
        ax.fill_between(
            range(len(mean_curve)),
            min_curve,
            max_curve,
            alpha=0.3,
            color="blue",
            label="Min-Max",
        )

    if style["grid"]:
        ax.grid(alpha=0.3)
    ax.legend()
    if style.get("title"):
        ax.set_title(style["title"])
    if style.get("xlabel"):
        ax.set_xlabel(style["xlabel"])
    if style.get("ylabel"):
        ax.set_ylabel(style["ylabel"])

    if ax is None:
        plt.tight_layout()
        plt.show()

    return fig


def plot_grid(
    plot_func: Callable[..., Any],
    data_list: list[Any],
    subplot_shape: tuple[int, int] | None = None,
    sample: int | None = None,
    **kwargs: Unpack[PlotStyle],
) -> plt.Figure:
    """Create grid of plots using specified plot function."""
    style = merge_defaults(**kwargs)

    # Sample data if requested
    if sample and len(data_list) > sample:
        data_list = sample_data(data_list, sample)

    n_plots = len(data_list)

    if subplot_shape is None:
        cols = int(np.ceil(np.sqrt(n_plots)))
        rows = int(np.ceil(n_plots / cols))
    else:
        rows, cols = subplot_shape

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = np.atleast_1d(axes).flatten()

    for i, data in enumerate(data_list):
        if i < len(axes):
            plot_func(data, ax=axes[i], **kwargs)  # type: ignore[arg-type]
            axes[i].set_title(f"Plot {i + 1}")

    # Hide unused subplots
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)

    if style.get("suptitle"):
        fig.suptitle(style["suptitle"])

    plt.tight_layout()
    plt.show()

    return fig


def set_plot_defaults(**kwargs: Unpack[PlotStyle]) -> None:
    """Update default plotting style."""
    DEFAULT_STYLE.update(kwargs)
