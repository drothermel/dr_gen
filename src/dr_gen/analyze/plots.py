"""Simplified plotting module for dr_gen analysis.

Replaces plot_utils.py + common_plots.py with direct matplotlib usage.
Eliminates overengineering while maintaining all core functionality.
"""

import random
from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

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


def merge_defaults(**kwargs: Any) -> dict[str, Any]:
    """Merge user kwargs with defaults."""
    return {**DEFAULT_STYLE, **kwargs}


def sample_data(data: list[Any], n: int | None = None) -> list[Any]:
    """Randomly sample n items from data."""
    if n is None or n >= len(data):
        return data
    return random.sample(data, n)


def plot_lines(
    curves: list[float] | list[list[float]],
    sample: int | None = None,
    ax: plt.Axes | None = None,
    **kwargs: Any,
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
    vals: list[float] | list[list[float]], ax: plt.Axes | None = None, **kwargs: Any
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
    vals1: list[float], vals2: list[float], ax: plt.Axes | None = None, **kwargs: Any
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
    **kwargs: Any,
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
    **kwargs: Any,
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
    **kwargs: Any,
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
            plot_func(data, ax=axes[i], **kwargs)
            axes[i].set_title(f"Plot {i + 1}")

    # Hide unused subplots
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)

    if style.get("suptitle"):
        fig.suptitle(style["suptitle"])

    plt.tight_layout()
    plt.show()

    return fig


def set_plot_defaults(**kwargs: Any) -> None:
    """Update default plotting style."""
    DEFAULT_STYLE.update(kwargs)
