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
            "subplot_ylabel": None,
            "subplot_xlabel": None,
            "suptitle": None,
            "suptitle_fs": 16,
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
        "min": np.min(v_array, axis=0),
        "max": np.max(v_array, axis=0),
    }
    return {
        k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in all_stats.items()
    }

def extract_color(ax, total=None, ind=None):
    lines = ax.get_lines()
    if len(lines) == 0:
        return None

    if ind is None or (total is None and len(lines) < ind) or len(lines) < total:
        return lines[-1].get_color()

    return lines[-total:][ind].get_color()

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

def add_min_max_shade_to_plot(
    plc, ax, data_list,
    data_stats=None,
    min_line=False, max_line=False,
):
    data_list = make_list_of_lists(data_list)
    data_n = len(data_list)

    if data_stats is None:
        data_stats = [
            get_multi_curve_summary_stats(dl) for dl in data_list
        ]

    colors = [extract_color(ax, total=data_n, ind=i) for i in  range(data_n)]

    for i, dl in enumerate(data_list):
        dstats = data_stats[i]
        x = dstats['x_vals']
        ax.fill_between(
            x, dstats["min"], dstats["max"], color=colors[i], alpha=0.1
        )
        if min_line:
            ax.plot(x, dstats['min'], color=colors[i], alpha=0.6)
        if max_line:
            ax.plot(x, dstats['max'], color=colors[i], alpha=0.6)

def add_std_shade_to_plot(
    plc, ax, data_list,
    data_stats=None,
    std_min_line=False, std_max_line=False,
):
    data_list = make_list_of_lists(data_list)
    data_n = len(data_list)

    if data_stats is None:
        data_stats = [
            get_multi_curve_summary_stats(dl) for dl in data_list
        ]

    colors = [extract_color(ax, total=data_n, ind=i) for i in  range(data_n)]
    for i, dl in enumerate(data_list):
        dstats = data_stats[i]
        x = dstats['x_vals']
        std_low = [m - s for m, s in zip(dstats['mean'], dstats['std'])]
        std_high = [m + s for m, s in zip(dstats['mean'], dstats['std'])]
        ax.fill_between(
            x,  std_low, std_high,
            color=colors[i], alpha=0.3
        )
        if std_min_line:
            ax.plot(x, std_low, color=colors[i], alpha=0.6)
        if std_max_line:
            ax.plot(x, std_high, color=colors[i], alpha=0.6)

def add_sem_shade_to_plot(
    plc, ax, data_list,
    data_stats=None,
    sem_min_line=False, sem_max_line=False,
):
    data_list = make_list_of_lists(data_list)
    data_n = len(data_list)

    if data_stats is None:
        data_stats = [
            get_multi_curve_summary_stats(dl) for dl in data_list
        ]

    colors = [extract_color(ax, total=data_n, ind=i) for i in  range(data_n)]
    for i, dl in enumerate(data_list):
        dstats = data_stats[i]
        x = dstats['x_vals']
        sem_low = [m - s for m, s in zip(dstats['mean'], dstats['sem'])]
        sem_high = [m + s for m, s in zip(dstats['mean'], dstats['sem'])]
        ax.fill_between(
            x,  sem_low, sem_high,
            color=colors[i], alpha=0.3
        )
        if sem_min_line:
            ax.plot(x, sem_low, color=colors[i], alpha=0.6)
        if sem_max_line:
            ax.plot(x, sem_high, color=colors[i], alpha=0.6)

def add_histograms_to_plot(plc, ax, vals_list, means=None):
    vals_list = make_list_of_lists(vals_list)
    for i, vals in enumerate(vals_list):
        n, bins, patches = ax.hist(
            vals,
            bins=plc.nbins,
            histtype="stepfilled",
            alpha=plc.alpha,
            range=plc.hist_range,
            density=plc.density,
        )
        if means is not None:
            color = patches[0].get_facecolor()
            ax.axvline(
                means[i],
                color=color,
                linestyle="dashed",
                linewidth=1.5,
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

def make_histogram_plot(data_lists, ax=None, **kwargs):
    assert len(data_lists) > 0, ">> Empty data lists"
    if 'plc' in kwargs:
        plc = kwargs['plc']
    else:
        plc = get_plt_cfg(**kwargs)

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=plc.figsize)
    means = [get_multicurve_summary_stats(ds)['mean'] for ds in data_lists]
    add_histograms_to_plot(plc, ax, data_lists, means)
    format_plot_element(plc, ax)
    if fig is not None:
        plt.show()

def make_summary_plot(data_lists, ax=None, **kwargs):
    assert len(data_lists) > 0, ">> Empty data lists"
    n_data = len(data_lists)
    if 'plc' in kwargs:
        plc = kwargs['plc']
    else:
        plc = get_plt_cfg(**kwargs)

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=plc.figsize)

    data_stats = [
        get_multi_curve_summary_stats(dl) for dl in data_lists
    ]

    # Mean Line
    if plc.labels is None or len(plc.labels) != n_data:
        plc.labels = [f"{i}" for i in range(n_data)]
    orig_linewidth = plc.linewidth
    plc.linewidth = kwargs.get("mean_linewidth", 2)
    plc.labels = [
        f"{lb} Mean (#seeds: {len(data_lists[i])})" for i, lb in enumerate(plc.labels)
    ]
    add_lines_to_plot(
        plc, ax, [ds['mean'] for ds in data_stats],
    )
    plc.linewidth = orig_linewidth

    # Min Max Shade
    if kwargs.get("min_max_shade", True):
        plc.labels = [None for _ in range(n_data)]
        add_min_max_shade_to_plot(
            plc, ax, data_lists,
            data_stats=data_stats,
            min_line=kwargs.get('min_line', False),
            max_line=kwargs.get('max_line', False),
        )

    if kwargs.get("std_shade", True):
        plc.labels = [None for _ in range(n_data)]
        add_std_shade_to_plot(
            plc, ax, data_lists,
            data_stats=data_stats,
            std_min_line=kwargs.get('std_min_line', False),
            std_max_line=kwargs.get('std_max_line', False),
        )

    if kwargs.get("sem_shade", True):
        plc.labels = [None for _ in range(n_data)]
        add_sem_shade_to_plot(
            plc, ax, data_lists,
            data_stats=data_stats,
            sem_min_line=kwargs.get('sem_min_line', False),
            sem_max_line=kwargs.get('sem_max_line', False),
        )

    format_plot_element(plc, ax)
    if fig is not None:
        plt.show()


# ----------------- Make Grids ------------------------

def grid_wrapper(plot_func, data_lists, **kwargs):
    data_lists = make_list_of_lists(data_lists)
    data_n = len(data_lists)

    grid_plc = get_plt_cfg(**kwargs)
    plc_list = []
    for i in range(data_n):
        kw_i = {}
        for k, v in kwargs.items():
            if isinstance(v, list):
                kw_i[k] = v[i]
            else:
                kw_i[k] = v
        plc_list.append(get_plt_cfg(**kw_i))
        
    # Get grid shape
    sp_x, sp_y = grid_plc.subplot_shape
    if sp_x is None:
        sp_x = data_n
    elif sp_y is None:
        sp_y = data_n
    assert all([size is not None for size in [sp_x, sp_y]])

    # Make figure and axes
    fs_x, fs_y = grid_plc.figsize
    fig, axes = plt.subplots(
        sp_x, sp_y, 
        figsize=(fs_y*sp_y, fs_x*sp_x),
    )
    axes = np.atleast_2d(axes) # Make indexing easy
    print(axes.shape)

    # Plot data
    for i, (ax, data_list) in enumerate(zip(axes.flatten(), data_lists)):
        plot_func(data_list, ax=ax, plc=plc_list[i], **kwargs)

    # Annotate Plot
    if grid_plc.subplot_ylabel is not None:
        for y in range(axes.shape[0]):
            axes[y, 0].set_ylabel(grid_plc.subplot_ylabel)
    if grid_plc.subplot_xlabel is not None:
        for x in range(axes.shape[1]):
            axes[axes.shape[0]-1, x].set_xlabel(grid_plc.subplot_xlabel)

    fig.suptitle(grid_plc.suptitle, fontsize=grid_plc.suptitle_fs)
    plt.tight_layout()
    plt.show()
 

# =============================================================
#                         OLD
# =============================================================

# -----------------------------------------------------------
#                  Add Elems to Plot
# -----------------------------------------------------------



def add_cdfs_to_plot(plc, vals, cdfs):
    cdfs = make_list_of_lists(cdfs)
    for i, cdf in enumerate(cdfs):
        (line,) = plt.plot(vals, cdf, linestyle=plc.linestyle, label=plc.labels[i])
        color = line.get_color()
        plt.fill_between(vals, cdf, color=color, alpha=plc.alpha)


# -----------------------------------------------------------
#                 Make Full Plots
# -----------------------------------------------------------

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

