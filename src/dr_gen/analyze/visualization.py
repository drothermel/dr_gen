"""Simplified plotting module for dr_gen analysis.

Replaces plot_utils.py + common_plots.py with direct matplotlib usage.
Eliminates overengineering while maintaining all core functionality.
"""

import random
from collections.abc import Callable
from typing import Any, TypedDict, Unpack, Dict, List
from dr_gen.analyze.database import ExperimentDB
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib.lines import Line2D






def group_metric_by_hpms_v2(
    db: ExperimentDB,
    metric: str,
    **hpm_filters: Any
) -> Dict[tuple, pd.DataFrame]:
    """
    Group metric time series by main_hpms (excluding 'seed'), filtered by hpm_filters.
    
    Args:
        db: The ExperimentDB instance.
        metric: The metric name (e.g., 'train_loss', 'val_acc').
        **hpm_filters: Keyword arguments for filtering (e.g., batch_size=128, model__width_mult=1.0).
                      Note: Use double underscore for nested keys (e.g., model__width_mult for model.width_mult)
    
    Returns:
        Dict mapping from tuple of hpm values (excluding 'seed' and 'run_id') to DataFrame with columns:
        - epoch: timestep
        - seed: seed value
        - [metric]: the metric value
    """
    # Get the filtered runs based on hpm filters
    filtered_runs_df = db._runs_df
    
    # Apply filters - handle both dot notation and double underscore notation
    for k, v in hpm_filters.items():
        # Convert double underscore to dot notation if needed
        col_name = k.replace('__', '.')
        
        # Check if column exists
        if col_name in filtered_runs_df.columns:
            filtered_runs_df = filtered_runs_df.filter(pl.col(col_name) == v)
        else:
            print(f"Warning: Column '{col_name}' not found in runs dataframe")
    
    # Get run_ids from filtered runs
    run_ids = filtered_runs_df['run_id'].to_list()
    
    if not run_ids:
        print("No runs found matching the filters")
        return {}
    
    # Get metrics for these runs
    metrics_df = db._metrics_df.filter(
        (pl.col('run_id').is_in(run_ids)) & 
        (pl.col('metric') == metric)
    )
    
    # Join metrics with run info to get hpms
    # Remove 'run_id' from important_hpms to avoid duplicate
    hpms_to_select = [h for h in db.important_hpms if h != 'run_id']
    joined_df = metrics_df.join(
        filtered_runs_df.select(['run_id'] + hpms_to_select),
        on='run_id',
        how='left'
    )
    
    # Group by all important hpms except 'seed' and 'run_id'
    group_keys = [h for h in db.important_hpms if h not in ['seed', 'run_id']]
    
    # Convert to pandas for easier grouping
    joined_pd = joined_df.to_pandas()
    
    # Create result dictionary
    result = {}
    
    # Group by the hpm keys
    for group_values, group_df in joined_pd.groupby(group_keys):
        # Create a clean dataframe with just epoch, seed, and metric value
        metric_df = group_df[['epoch', 'seed', 'value']].copy()
        metric_df.rename(columns={'value': metric}, inplace=True)
        metric_df = metric_df.sort_values(['seed', 'epoch'])
        
        # Use tuple of group values as key
        if isinstance(group_values, tuple):
            key = group_values
        else:
            key = (group_values,)
            
        result[key] = metric_df
    
    return result

def group_metric_by_hpms_v2(
    db: ExperimentDB,
    metric: str,
    **hpm_filters: Any
) -> Dict[tuple, pd.DataFrame]:
    """
    Group metric time series by main_hpms (excluding 'seed'), filtered by hpm_filters.
    
    Args:
        db: The ExperimentDB instance.
        metric: The metric name (e.g., 'train_loss', 'val_acc').
        **hpm_filters: Keyword arguments for filtering (e.g., batch_size=128, model__width_mult=1.0).
                      Note: Use double underscore for nested keys (e.g., model__width_mult for model.width_mult)
    
    Returns:
        Dict mapping from tuple of hpm values (excluding 'seed' and 'run_id') to DataFrame with columns:
        - epoch: timestep
        - seed: seed value
        - [metric]: the metric value
    """
    # Get the filtered runs based on hpm filters
    filtered_runs_df = db._runs_df
    
    # Apply filters - handle both dot notation and double underscore notation
    for k, v in hpm_filters.items():
        # Convert double underscore to dot notation if needed
        col_name = k.replace('__', '.')
        
        # Check if column exists
        if col_name in filtered_runs_df.columns:
            filtered_runs_df = filtered_runs_df.filter(pl.col(col_name) == v)
        else:
            print(f"Warning: Column '{col_name}' not found in runs dataframe")
    
    # Get run_ids from filtered runs
    run_ids = filtered_runs_df['run_id'].to_list()
    
    if not run_ids:
        print("No runs found matching the filters")
        return {}
    
    # Get metrics for these runs
    metrics_df = db._metrics_df.filter(
        (pl.col('run_id').is_in(run_ids)) & 
        (pl.col('metric') == metric)
    )
    
    # Join metrics with run info to get hpms
    # Remove 'run_id' from important_hpms to avoid duplicate
    hpms_to_select = [h for h in db.important_hpms if h != 'run_id']
    joined_df = metrics_df.join(
        filtered_runs_df.select(['run_id'] + hpms_to_select),
        on='run_id',
        how='left'
    )
    
    # Group by all important hpms except 'seed' and 'run_id'
    group_keys = [h for h in db.important_hpms if h not in ['seed', 'run_id']]
    
    # Convert to pandas for easier grouping
    joined_pd = joined_df.to_pandas()
    
    # Create result dictionary
    result = {}
    
    # Group by the hpm keys
    for group_values, group_df in joined_pd.groupby(group_keys):
        # Create a clean dataframe with just epoch, seed, and metric value
        metric_df = group_df[['epoch', 'seed', 'value']].copy()
        metric_df.rename(columns={'value': metric}, inplace=True)
        metric_df = metric_df.sort_values(['seed', 'epoch'])
        
        # Use tuple of group values as key
        if isinstance(group_values, tuple):
            key = group_values
        else:
            key = (group_values,)
            
        result[key] = metric_df
    
    return result

def prepare_data_for_plotting(
    grouped_results: Dict[tuple, pd.DataFrame],
    metric_name: str,
    db: ExperimentDB,
    aggregate_seeds: bool = True
) -> pd.DataFrame:
    """
    Prepare the grouped results for use with plot_training_metrics.
    
    Args:
        grouped_results: Output from group_metric_by_hpms_v2
        metric_name: Name of the metric being plotted
        db: ExperimentDB instance (to access important_hpms)
        aggregate_seeds: If True, average across seeds; if False, include seed as a column
    
    Returns:
        DataFrame formatted for plot_training_metrics with columns:
        epoch, lr, wd, [metric_name], and optionally seed
    """
    all_dfs = []
    
    for group_key, df in grouped_results.items():
        # Extract lr and wd from the group key
        # Find indices for lr and wd in the group key
        group_keys = [h for h in db.important_hpms if h not in ['seed', 'run_id']]
        lr_idx = group_keys.index('optim.lr')
        wd_idx = group_keys.index('optim.weight_decay')
        
        lr = group_key[lr_idx]
        wd = group_key[wd_idx]
        
        # Add lr and wd columns
        df = df.copy()
        df['lr'] = lr
        df['wd'] = wd
        
        if aggregate_seeds:
            # Average across seeds
            agg_df = df.groupby(['epoch', 'lr', 'wd'])[metric_name].mean().reset_index()
            all_dfs.append(agg_df)
        else:
            all_dfs.append(df)
    
    # Combine all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)
    return combined_df

def prep_all_data(
    db: ExperimentDB,
    metrics: List[str],
    hpm_filters: Dict[str, Any],
    aggregate_seeds: bool = True,
) -> None:
    # Prepare data for each metric
    metric_dfs = {}
    for metric in metrics:
        grouped = group_metric_by_hpms_v2(db, metric, **hpm_filters)
        if not grouped:
            print(f"No data found for metric: {metric}")
            continue
        metric_dfs[metric] = prepare_data_for_plotting(grouped, metric, db, aggregate_seeds)
    
    if not metric_dfs:
        print("No data to plot")
        return
    
    # Combine all metric dataframes
    # We need to merge them on epoch, lr, wd
    combined_df = None
    for metric, df in metric_dfs.items():
        if combined_df is None:
            combined_df = df
        else:
            # Merge on common columns
            merge_cols = ['epoch', 'lr', 'wd']
            if 'seed' in df.columns and 'seed' in combined_df.columns:
                merge_cols.append('seed')
            combined_df = combined_df.merge(df, on=merge_cols, how='outer')
    
    return combined_df



def plot_training_metrics(df: pd.DataFrame, xlog: bool = False, ylog: bool = False, yrange_loss: tuple[float, float] = None, yrange_acc: tuple[float, float] = None, figsize: tuple[int, int] = (17, 4.2)) -> None:
    """
    Draw the *four* panels (train‑loss, val‑loss, train‑acc, val‑acc) with
    colour = learning‑rate and linestyle = weight‑decay.  The mapping is derived
    automatically from whatever unique values appear in the dataframe.

    Parameters
    ----------
    df : tidy dataframe with at least the columns produced by
         `make_mock_cifar_df`.
    """
    # ----- discover the hyper‑parameter palette / style -----------------
    lrs = sorted(df['lr'].unique())         # e.g. [0.01, 0.03, 0.1]
    wds = sorted(df['wd'].unique())         # e.g. [1e‑4, 3e‑4, 1e‑3]

    # colours – reuse tab10 but subsample in case there are more than 10 lrs
    cmap = plt.get_cmap('tab10')
    colour_map = {lr: cmap(i % 10) for i, lr in enumerate(lrs)}

    # linestyles – cycle through the classic trio, then fallback to dash‑dot etc.
    style_cycle = ['solid', 'dashed', 'dotted', 'dashdot', (0, (3, 1, 1, 1))]
    style_map = {wd: style_cycle[i % len(style_cycle)] for i, wd in enumerate(wds)}

    # ----- set‑up the 1×4 sub‑plots ------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=figsize, sharex=True)

    panels = [('train_loss', 'Train Loss', ylog),
              ('val_loss',   'Val Loss',   ylog),
              ('train_acc',  'Train Acc',  False),
              ('val_acc',    'Val Acc',    False)]

    for (metric, title, met_ylog), ax in zip(panels, axes):
        for lr in lrs:
            for wd in wds:
                subset = df[(df.lr == lr) & (df.wd == wd)]
                ax.plot(subset['epoch'], subset[metric],
                        label=f'lr={lr}, wd={wd}',
                        color=colour_map[lr],
                        linestyle=style_map[wd],
                        linewidth=1.8)

        ax.set_title(title)
        ax.grid(alpha=.3, which='both', linestyle=':')
        if metric == 'train_loss' or metric == 'val_loss':
            ax.set_ylim(yrange_loss)
        elif metric == 'train_acc' or metric == 'val_acc':
            ax.set_ylim(yrange_acc)
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
    ncol = len(wds) + 1

    leg = fig.legend(handles, labels,
                    ncol=ncol,
                    loc='lower center',
                    bbox_to_anchor=(0.5, -0.20),
                    frameon=False,
                    columnspacing=1.5,
                    handletextpad=0.6)

    # add a bold title and subtitle (two separate lines)
    leg.set_title(f'{title}', prop={'weight': 'bold', 'size': 'medium'})

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.32)   # leave room beneath the plots
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
