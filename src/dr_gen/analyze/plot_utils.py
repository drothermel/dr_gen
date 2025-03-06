from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import numpy as np

from dr_gen.utils.utils import make_list_of_lists

# -----------------------------------------------------------
#                 Plot Configs
# -----------------------------------------------------------


def get_plt_cfg(**kwargs):
    base_plt_cfg = OmegaConf.create(
        {
            "figsize": (10, 6),
            "legend": False,
            "grid": True,
            "nbins": 10,
            "hist_range": None,
            "title": None,
            "xlabel": None,
            "ylabel": None,
            "xlim": None,
            "ylim": None,
            "density": False,
            "alpha": 1.0,
            "linestyle": None,
            "linewidth": 1,
            "labels": None,
            "colors": None,
            "subplot_shape": (1, None),
        }
    )
    plc = base_plt_cfg.copy()
    plc.hist_range = (80.0, 100.0)
    plc.nbins = 100
    plc.legend = True

    for k, v in kwargs.items():
        if k in plc:
            plc[k] = v
    return plc


def plcv(plc, key, default):
    if key in plc and plc[key] is not None:
        return plc[key]
    return default


def init_plc_lists(plc, list_len):
    plc.labels = plcv(plc, "labels", list(range(list_len)))
    # Set these to a lists of None
    for k in ["colors"]:
        plc[k] = plcv(plc, k, [None for _ in range(list_len)])
    return plc


# -----------------------------------------------------------
#               Grid Enabled Library Attempt
# -----------------------------------------------------------

def format_plot_element(plc, ax):
    if plc.xlabel is not None:
        ax.set_xlabel(plc.xlabel)
    if plc.ylabel is not None:
        ax.set_ylabel(plc.ylabel)
    if plc.title is not None:
        ax.set_title(plc.title)
    if plc.legend:
        ax.legend()
    if plc.xlim is not None:
        ax.set_xlim(plc.xlim)
    if plc.ylim is not None:
        ax.set_ylim(plc.ylim)
    ax.grid(plc.grid)

# ----------------- Add Elements ------------------------

def add_lines_to_plot(plc, ax, data_list, xs=None):
    data_list = make_list_of_lists(data_list)
    plc = init_plc_lists(plc, len(data_list))

    if xs is None:
        xs = range(len(data_list[0]))
    for i, data in enumerate(data_list):
        ax.plot(
            xs,
            data,
            linestyle=plc.linestyle,
            linewidth=plc.linewidth,
            label=plc.labels[i],
        )

# ----------------- Make Plots ------------------------


def make_line_plot(data_lists, ax=None, **kwargs):
    assert len(data_lists) > 0, ">> Empty data lists"
    if 'plc' in kwargs:
        plc = kwargs['plc']
    else:
        plc = get_plt_cfg(**kwargs)

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=plc.figsize)
    add_lines_to_plot(plc, ax, data_lists)
    format_plot_element(plc, ax)
    if fig is not None:
        plt.show()


# ----------------- Make Grids ------------------------

def grid_wrapper(plot_func, data_lists, **kwargs):
    data_lists = make_list_of_lists(data_lists)
    plc = get_plt_cfg(**kwargs)

    # Get grid shape
    n = len(data_lists)
    sp_x, sp_y = plc.subplot_shape
    if sp_x is None:
        sp_x = n
    elif sp_y is None:
        sp_y = n
    assert all([size is not None for size in [sp_x, sp_y]])

    fs_x, fs_y = plc.figsize
    fig, axes = plt.subplots(
        sp_x, sp_y, 
        figsize=(fs_x*sp_x, fs_y*sp_y),
    )
    if n == 1:
        axes = [axes]  # Make it iterable
    for ax, data_list in zip(axes, data_lists):
        plot_func(data_list, ax=ax, plc=plc)

    plt.tight_layout()
    plt.show()
 

# =============================================================
#                         OLD
# =============================================================

# -----------------------------------------------------------
#               Misc & Calc Plot Elements
# -----------------------------------------------------------


def get_multi_curve_summary_stats(data_list):
    data_list = make_list_of_lists(data_list)
    curve_len = len(data_list[0])
    assert all([len(dl) == curve_len for dl in data_list]), (
        ">> All curves must be same length"
    )

    n = len(data_list)
    v_array = np.array(data_list)
    v_mean = np.mean(data_list, axis=0)
    v_std = np.std(v_array, axis=0)
    v_sem = v_std / np.sqrt(n)
    all_stats = {
        "n": n,
        "x_vals": list(range(curve_len)),
        "mean": v_mean,
        "std": v_std,
        "sem": v_sem,
        "std_lower": v_mean - v_std,
        "std_upper": v_mean + v_std,
        "sem_lower": v_mean - v_sem,
        "sem_upper": v_mean + v_sem,
        "min": np.min(v_array, axis=0),
        "max": np.max(v_array, axis=0),
    }
    return {
        k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in all_stats.items()
    }


# -----------------------------------------------------------
#                  Add Elems to Plot
# -----------------------------------------------------------


def format_plot_grid(plc):
    if plc.xlabel is not None:
        plt.xlabel(plc.xlabel)
    if plc.ylabel is not None:
        plt.ylabel(plc.ylabel)
    if plc.title is not None:
        plt.title(plc.title)
    if plc.legend:
        plt.legend()
    if plc.xlim is not None:
        plt.xlim(plc.xlim)
    if plc.ylim is not None:
        plt.ylim(plc.ylim)
    plt.grid(plc.grid)


def add_lines_to_plot_old(plc, data_list, xs=None):
    data_list = make_list_of_lists(data_list)
    plc = init_plc_lists(plc, len(data_list))

    if xs is None:
        xs = range(len(data_list[0]))
    for i, data in enumerate(data_list):
        plt.plot(
            xs,
            data,
            linestyle=plc.linestyle,
            linewidth=plc.linewidth,
            label=plc.labels[i],
        )


def add_cdfs_to_plot(plc, vals, cdfs):
    cdfs = make_list_of_lists(cdfs)
    for i, cdf in enumerate(cdfs):
        (line,) = plt.plot(vals, cdf, linestyle=plc.linestyle, label=plc.labels[i])
        color = line.get_color()
        plt.fill_between(vals, cdf, color=color, alpha=plc.alpha)


def add_histograms_to_plot(plc, vals_list, means=None):
    vals_list = make_list_of_lists(vals_list)
    colors = ["red", "blue", "green"]
    for i, vals in enumerate(vals_list):
        color = colors[i % len(colors)]
        plt.hist(
            vals,
            bins=plc.nbins,
            histtype="stepfilled",
            alpha=plc.alpha,
            edgecolor=color,
            facecolor=color,
            range=plc.hist_range,
            density=plc.density,
        )
        if means is not None:
            plt.axvline(
                means[i],
                color=color,
                linestyle="dashed",
                linewidth=1.5,
                label=plc.labels[i],
            )


# -----------------------------------------------------------
#                 Make Full Plots
# -----------------------------------------------------------


def make_line_plot(data_lists, **kwargs):
    assert len(data_lists) > 0, ">> Empty data lists"
    plc = get_plt_cfg(**kwargs)

    # Make figure add lines
    plt.figure(figsize=plc.figsize)
    add_lines_to_plot(plc, data_lists)
    format_plot_grid(plc)
    plt.show()


# Expected: [list_of_data = [metrics_per_epoch ...] ...]
def make_summary_line_plot(
    run_group_data_lists,
    **kwargs,
):
    # Expected data_list shapes: Num Datasets x Num Runs x Num Epochs
    #    but Num Runs can vary between datasets
    run_group_stats = [get_multi_curve_summary_stats(dl) for dl in run_group_data_lists]

    # Setup PLC
    defaults = {
        "xlabel": "Epoch",
        "ylabel": "Loss",
        "title": "Loss During Training",
        "labels": ["train mean", "val mean", "eval mean"],
    }
    defaults.update(kwargs)
    plc = get_plt_cfg(**defaults)

    # Make plot figure
    plt.figure(figsize=plc.figsize)
    for i, rd_stats in enumerate(run_group_stats):
        x = rd_stats["x_vals"]
        (line_mean,) = plt.plot(x, rd_stats["mean"], linewidth=3, label=plc.labels[i])
        color = line_mean.get_color()

        plt.fill_between(x, rd_stats["min"], rd_stats["max"], color=color, alpha=0.1)
        plt.fill_between(
            x, rd_stats["std_lower"], rd_stats["std_upper"], color=color, alpha=0.3
        )
        plt.plot(
            x,
            rd_stats["sem_lower"],
            linestyle="-",
            linewidth=1,
            color=color,
            alpha=0.5,
        )
        plt.plot(
            x,
            rd_stats["sem_upper"],
            linestyle="-",
            linewidth=1,
            color=color,
            alpha=0.5,
        )

    format_plot_grid(plc)
    plt.show()


def make_cdfs_plot(vals, cdfs, **kwargs):
    cdfs = make_list_of_lists(cdfs)
    defaults = {
        "labels": [f"CDF {i}" for i in range(len(cdfs))],
        "linestyle": "-",
        "alpha": 0.3,
        "xlabel": "accuracy",
        "ylabel": "cdf",
        "title": "CDF" + "s" if len(cdfs) > 1 else "",
    }
    defaults.update(kwargs)
    plc = get_plt_cfg(**defaults)

    plt.figure(figsize=plc.figsize)
    add_cdfs_to_plot(plc, vals, cdfs)
    format_plot_grid(plc)


def make_histogram_plot(vals_list, means=None, **kwargs):
    vals_list = make_list_of_lists(vals_list)
    plc = get_plt_cfg(**kwargs)
    plc = init_plc_lists(plc, len(vals_list))

    plt.figure(figsize=plc.figsize)
    add_histograms_to_plot(plc, vals_list, means=means)
    format_plot_grid(plc)
