import random
import matplotlib.pyplot as plt

from dr_gen.utils.utils import make_list, make_list_of_lists, make_list_of_lols
import dr_gen.analyze.ks_stats as ks
import dr_gen.analyze.plot_utils as pu


def len_to_inds(data_len):
    return list(range(data_len))

def data_to_inds(data):
    return len_to_inds(len(data))


def default_grid_ind_labels(data):
    data = make_list_of_lists(data)
    return [pu.default_ind_labels(dt) for dt in data]

def default_grid_sample_titles(title, orig_data, sampled):
    titles = []
    for orig, samp in zip(orig_data, sampled):
        titles.append(f"{title} | Sampled {len(samp)} / {len(orig)}")
    return titles
             
    
    
# ---------------- Individual Deterministic Plots ------------------

# Handles one or many curves
# Nearly identical to make_line_plot but included to make example
#    pattern clear.
def line_plot(curve, ax=None, **kwargs):
    # [curve_data...] or [curves [curve_data ...]]
    curve = make_list(curve)

    # Set default chart annotations
    if isinstance(curve[0], list):
        kwargs['labels'] = kwargs.get('labels', pu.default_ind_labels(curve))

    # Plot: len(curve) lines
    plt_show, ax = pu.get_subplot_axis(ax, figsize=kwargs.get('figsize', None))
    pu.make_line_plot(curve, ax=ax, **kwargs)
    if plt_show: plt.show()

# Handles one or many vals lists
def histogram_plot(vals, ax=None, **kwargs):
    # [vals...] or [sets [vals ...]]
    vals = make_list(vals)

    # Plot
    plt_show, ax = pu.get_subplot_axis(ax, figsize=kwargs.get('figsize', None))
    pu.make_histogram_plot(vals, ax=ax, **kwargs)
    if plt_show: plt.show()

def cdf_plot(vals1, vals2, ax=None, **kwargs):
    results = ks.calculate_ks_for_run_sets(vals1, vals2)
    vals = results['all_vals']
    cdfs = [results['cdf1'], results['cdf2']]

    kwargs['linestyle'] = kwargs.get('linestyle', '-')
    kwargs['alpha'] = kwargs.get('alpha', 0.3)
    kwargs['xlabel'] = kwargs.get('xlabel', "accuracy")
    kwargs['ylabel'] = kwargs.get('ylabel', "cdf")
    kwargs['title'] = kwargs.get('title', "CDF" + "s" if len(cdfs) > 1 else "")
    kwargs['labels'] = kwargs.get('labels', [f"CDF {i}" for i in range(len(cdfs))])

    plt_show, ax = pu.get_subplot_axis(ax, figsize=kwargs.get('figsize', None))
    pu.make_cdfs_plot(vals, cdfs, ax=ax, **kwargs)
    if plt_show: plt.show()

def cdf_histogram_plot(vals1, vals2, ax=None, **kwargs):
    results = ks.calculate_ks_for_run_sets(vals1, vals2)
    vals = results['all_vals']
    cdfs = [results['cdf1'], results['cdf2']]


    axes = pu.make_grid_figure(
        data_len=2,
        nominal_subplot_shape=(2, 1),
        plot_size=kwargs.get('figsize', pu.DEFAULT_FIGSIZE)
    )

    kwargs['linestyle'] = kwargs.get('linestyle', '-')
    kwargs['alpha'] = kwargs.get('alpha', 0.3)
    kwargs['suptitle'] = f"CDFs and Histograms | KS Stat: {results['ks_stat']:0.2f}"
    pu.make_cdfs_plot(vals, cdfs, ax=axes[0,0], **kwargs)
    pu.make_histogram_plot(
        [vals1, vals2], ax=axes[0,1], **kwargs,
    )
    pu.annotate_grid_figure(axes, pu.get_plt_cfg(**kwargs))
    plt.show()
    
    

# Handles one or many curves per split
def split_plot(
    split_curves,
    ax=None,
    metric_name="Metric",
    x_name="Epoch",
    splits=["train", "val", "eval"],
    **kwargs,
):
    # [split [curves [curve_data ...]]]
    split_curves = make_list_of_lols(split_curves, dim=1)

    # Set default chart annotations
    orig_labels = kwargs.get('labels', pu.default_ind_labels(split_curves[0]))
    kwargs['title'] = kwargs.get('title', f"{metric_name} by Split Per {x_name}")
    kwargs['xlabel'] = kwargs.get('xlabel', x_name)
    kwargs['ylabel'] = kwargs.get('ylabel', metric_name)

    # Plot by split
    plt_show, ax = pu.get_subplot_axis(ax, figsize=kwargs.get('figsize', None))
    n_curves = len(split_curves[0])
    line_styles = {
        "train": "--",
        "val": "-",
        "eval": ":",
    }
    colors = kwargs.get('colors', [None for _ in range(n_curves)])
    for split_i, split_data in enumerate(split_curves):
        if split_i >= len(splits):
            continue
        split = splits[split_i]

        # Set Per-Split Defaults
        kwargs['linestyle'] = line_styles[split] 
        kwargs['labels'] = [f'{il} {pu.first_upper_str(split)}' for il in orig_labels]
        kwargs['colors'] = colors
        plc = pu.get_plt_cfg(**kwargs)

        # Add the lines
        pu.add_lines_to_plot(plc, ax, split_data)
        # Make the line colors for future splits match
        if colors[0] == None:
            colors = pu.extract_colors(ax, n_curves)
    pu.format_plot_element(plc, ax)
    if plt_show: plt.show()

# ---------------- Individual Sampled Plots ------------------
def sample_n(data, n_sample=None):
    n_curves = len(data)
    if n_sample is None or n_sample > n_curves:
        n_sample = n_curves
        sampled = data
        sample_str = ""
    else:
        sampled = random.sample(data, n_sample)
        sample_str = f" | Sample: {n_sample} / {n_curves}"
    return sampled, sample_str

def sample_from_dim(data, n_sample, dim=1):
    n_data = len(data)
    n_curves = len(data[0])

    if dim  == 0:
        # Sample from curves, fixed sample across all data lists
        data_inds = len_to_inds(n_data)
        sampled_inds, sample_str = sample_n(data_inds, n_sample=n_sample)
        # [data [sampled_curves [curve_data ...]]]
        sampled_curves = [data[data_i] for data_i in sampled_inds]
    elif dim == 1:
        # Sample from curves, fixed sample across all data lists
        curve_inds = len_to_inds(n_curves)
        sampled_inds, sample_str = sample_n(curve_inds, n_sample=n_sample)
        # [data [sampled_curves [curve_data ...]]]
        sampled_curves = []
        for data_ind in range(n_data):
            sampled_curves.append([
                data[data_ind][curve_i] for curve_i in sampled_inds
            ])
    return sampled_inds, sampled_curves, sample_str

def multi_line_sample_plot(curves, ax=None, n_sample=None, **kwargs):
    # Initial Curves: [curves [curve_data ...]]
    # Sampled       : [sampled_curves [curve_data ...]]
    curves = make_list_of_lists(curves)
    sampled_inds, sampled_curves, sample_str = sample_from_dim(
        curves, n_sample=n_sample, dim=0, # dim = curves
    )
    kwargs['title'] = kwargs.get("title", f"Multi Line Plot") + sample_str
    line_plot(sampled_curves, ax=ax, **kwargs)


def split_sample_plot(
    split_curves_lists,
    ax=None,
    n_sample=None,
    metric_name="Metric",
    x_name="Epoch",
    splits=["train", "val", "eval"],
    **kwargs,
):
    # Initial Curves Goal: [split [curves [curve_data ...]]]
    #   -> assume missing dimension is "curves" not "split"
    split_curves_lists = make_list_of_lists(split_curves_lists, dim=1)
    # Sampled            : [split [sampled_curves [curve_data ...]]]
    sampled_inds, sampled_split_curves_lists, sample_str = sample_from_dim(
        split_curves_lists, n_sample, dim=1,
    )

    # Set default chart annotations
    kwargs['title'] = kwargs.get(
        'title', f"{metric_name} by Split Per {x_name}"
    ) + sample_str
    labels = kwargs.get('labels', ["" for _ in range(len(split_curves_lists[0]))])
    kwargs['labels'] = [
        f"{labels[si]} Sampled Run Index {si}" for si in sampled_inds
    ]

    split_plot(
        sampled_split_curves_lists, 
        ax=ax,
        metric_name=metric_name,
        x_name=x_name,
        splits=splits,
        **kwargs,
    )

def multi_line_sampled_summary_plot(
    curves_lists, ax=None, n_sample=None, **kwargs
):
    # Initial Curves: [summary_line [curves [curve_data ...]]
    curves_lists = make_list_of_lols(curves_lists, dim=0)
    # Sampled       : [summary_line [sampled_curves [curve_data ...]]
    sampled_inds, sampled_curves_lists, sample_str = sample_from_dim(
        curves_lists, n_sample, dim=1,
    )
    kwargs['title'] = kwargs.get("title", f"Multi Line Summary Plot{sample_str}")

    # n_data lines, summary over curves plotted
    plt_show, ax = pu.get_subplot_axis(ax, figsize=kwargs.get('figsize', None))
    pu.make_summary_plot(sampled_curves_lists, ax, **kwargs)
    if plt_show: plt.show()
    

def split_sampled_summary_plot(
    split_curves_lists,
    ax=None,
    n_sample=None,
    metric_name="Metric",
    x_name="Epoch",
    splits=["train", "val", "eval"],
    **kwargs,
):
    # Initial Curves Goal: [split [curves [curve_data ...]]]
    #   -> assume missing dimension is "curves" not "split"
    split_curves_lists = make_list_of_lists(split_curves_lists, dim=1)
    split_curves_lists = split_curves_lists[:len(splits)]
    # Sampled           : [split [sampled_curves [curve_data ...]]]
    sampled_inds, sampled_split_curves_lists, sample_str = sample_from_dim(
        split_curves_lists, n_sample, dim=1,
    )

    # Set default chart annotations
    kwargs['labels'] = kwargs.get(
        'labels',
        pu.default_split_labels(splits),
    )
    kwargs['title'] = kwargs.get('title', f"{metric_name} by Split Per {x_name}")
    kwargs['xlabel'] = kwargs.get('xlabel', x_name)
    kwargs['ylabel'] = kwargs.get('ylabel', metric_name)

    # n_splits lines, summary over curves plotted
    plt_show, ax = pu.get_subplot_axis(ax, figsize=kwargs.get('figsize', None))
    pu.make_summary_plot(sampled_split_curves_lists, ax, **kwargs)
    if plt_show: plt.show()


# -------------------- Grid Plots: Sample from Set -------------------------

def grid_sample_plot_wrapper(plot_func, curves, n_sample=None, n_grid=4, **kwargs):
    # Setup Grid and Args
    axes = pu.make_grid_figure(
        n_grid,
        nominal_subplot_shape=kwargs.get("subplot_shape", pu.DEFAULT_SUBPLOT_SHAPE),
        plot_size=kwargs.get("figsize", pu.DEFAULT_FIGSIZE),
    )
    # Use this only for non-sampled grid
    #kwargs_list = pu.get_kwargs_lists_for_grid(kwargs, n_grid)

    # Plot Grid
    for grid_ind, ax in enumerate(axes.flatten()):
        plot_func(
            curves, ax=ax, n_sample=n_sample, **kwargs,
        )

    # Annotate and Show
    pu.annotate_grid_figure(axes, pu.get_plt_cfg(**kwargs))
    plt.show()

def multi_line_sample_plot_grid(curves, n_sample=None, n_grid=4, **kwargs):
    grid_sample_plot_wrapper(
        multi_line_sample_plot, curves, n_sample=n_sample, n_grid=n_grid, **kwargs,
    )

def multi_line_sampled_summary_plot_grid(curves, n_sample=None, n_grid=4, **kwargs):
    grid_sample_plot_wrapper(
        multi_line_sampled_summary_plot, curves, n_sample=n_sample, n_grid=n_grid, **kwargs,
    )

def spilt_sample_plot_grid(curves, n_sample=None, n_grid=4, **kwargs):
    grid_sample_plot_wrapper(
        split_sample_plot, curves, n_sample=n_sample, n_grid=n_grid, **kwargs,
    )

def spilt_sampled_summary_plot_grid(curves, n_sample=None, n_grid=4, **kwargs):
    grid_sample_plot_wrapper(
        split_sampled_summary_plot, curves, n_sample=n_sample, n_grid=n_grid, **kwargs,
    )

def grid_seq_plot_wrapper(plot_func, curves, **kwargs):
    n_curves = len(curves)

    # Setup Grid and Args
    axes = pu.make_grid_figure(
        n_curves,
        nominal_subplot_shape=kwargs.get("subplot_shape", pu.DEFAULT_SUBPLOT_SHAPE),
        plot_size=kwargs.get("figsize", pu.DEFAULT_FIGSIZE),
    )
    # Use this only for non-sampled grid
    kwargs_list = pu.get_kwargs_lists_for_grid(kwargs, n_curves)

    # Plot Grid
    for curve_i, ax in enumerate(axes.flatten()):
        plot_func(curves[curve_i], ax=ax, **kwargs_list[curve_i])

    # Annotate and Show
    pu.annotate_grid_figure(axes, pu.get_plt_cfg(**kwargs))
    plt.show()

def histogram_plot_grid(vals, **kwargs):
    grid_seq_plot_wrapper(histogram_plot, vals, **kwargs)

