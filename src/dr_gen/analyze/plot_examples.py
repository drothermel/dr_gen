import random

from dr_gen.utils.utils import make_list_of_lists
import dr_gen.analyze.plot_utils as pu


def len_to_inds(data_len):
    return list(range(data_len))

def data_to_inds(data):
    return len_to_inds(len(data))

def default_ind_labels(data):
    return [f["{i}" for i in range(len(data))]

def default_grid_ind_labels(data):
    data = make_list_of_lists(data)
    return [default_ind_labels(dt) for dt in data]

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
    curves = make_list(curves)

    # Set default chart annotations
    if isinstance(curves[0], list):
        kwargs['labels'] = kwargs.get('labels', default_ind_labels(curves))

    # Plot: len(curves) lines
    plt_show, ax = pu.get_subplot_axis(ax)
    pu.make_line_plot(curves, ax=ax, **kwargs)
    if plt_show: plt.show()
    

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
    split_curves = make_list_of_lols(split_curves, dim=0)

    # Set default chart annotations
    orig_labels = kwargs.get('labels', default_ind_labels(split_curves[0]))
    kwargs['title'] = kwargs.get('title', "{metric_name} by Split Per {x_name}")
    kwargs['xlabel'] = kwargs.get('xlabel', x_name)
    kwargs['ylabel'] = kwargs.get('ylabel', metric_name)

    # Plot by split
    plt_show, ax = pu.get_subplot_axis(ax)
    n_curves = len(split_curves[0])
    line_styles = {
        "train": "--",
        "val": "-",
        "eval": ":",
    }
    colors = kwargs.get('colors', [None for _ in range(n_curves)])
    for split_i, split_data in enumerate(split_curves):
        split = splits[split_i]

        # Set Per-Split Defaults
        kwargs['linestyle'] = line_styles[split] 
        kwargs['labels'] = [f'{il} {pu.first_upper_str(split)}' for il in orig_labels]
        plc = get_plt_cfg(**kwargs)

        # Add the lines
        pu.add_lines_to_plot(plc, ax, split_data)
        # Make the line colors for future splits match
        if colors[0] == None:
            colors = extract_colors(ax, n_curves)
    pu.format_plot_element(plc, ax)
    if plt_show: plt.show()

# ---------------- Individual Sampled Plots ------------------
def sample_n(data, n_sample=None):
    n_in = len(data)
    if n_sample is None or n_sample > n_curves:
        n_sample = n_curves
        sampled = data
        sample_str = ""
    else:
        sampled = random.sample(data, n_sample)
        sample_str = " | Sample: {n_sample} / {n_curves}"
    return sampled, sample_str

def sample_from_dim_one(data, n_sample):
    n_data = len(data)
    n_curves = len(data[0])

    # Sample from curves, fixed sample across all data lists
    curve_inds = len_to_inds(n_curves)
    sampled_inds, sample_str = sample_n(curve_inds, n_sample=n_sample)

    # [data [sampled_curves [curve_data ...]]]
    sampled_curves = []
    for data_ind in range(n_data):
        sampled_curves[data_ind] = [
            data[data_ind][curve_i] for curve_i in sampled_inds
        ]
    return sampled_curves, sample_str

def multi_line_sample_plot(curves, ax=None, n_sample=None, **kwargs):
    # Initial Curves: [curves [curve_data ...]]
    # Sampled       : [sampled_curves [curve_data ...]]
    curves = make_list_of_lists(curves)
    sampled_curves, sample_str = sample_n(curves, n_sample=n_sample)
    kwargs['title'] = kwargs.get("title", "Multi Line Plot{sample_str}")
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
    split_curves_lists = make_list_of_lists(split_curves, dim=1)
    # Sampled            : [split [sampled_curves [curve_data ...]]]
    sampled_split_curves_lists, sample_str = sample_from_dim_one(
        split_curves_lists, n_samples,
    )

    # Set default chart annotations
    kwargs['title'] = kwargs.get(
        'title', "{metric_name} by Split Per {x_name}{sample_str}"
    )
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
    sampled_curves_lists, sample_str = sample_from_dim_one(curves_lists, n_sample)
    kwargs['title'] = kwargs.get("title", "Multi Line Summary Plot{sample_str}")

    # n_data lines, summary over curves plotted
    plt_show, ax = pu.get_subplot_axis(ax)
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
    split_curves_lists = make_list_of_lists(split_curves, dim=1)
    # Sampled           : [split [sampled_curves [curve_data ...]]]
    sampled_split_curves_lists, sample_str = sample_from_dim_one(
        split_curves_lists, n_samples,
    )

    # Set default chart annotations
    kwargs['labels'] = kwargs.get(
        'labels',
        [f"{lb} Mean{sample_str}" for lb in default_split_labels(splits)],
    )
    kwargs['title'] = kwargs.get('title', "{metric_name} by Split Per {x_name}")
    kwargs['xlabel'] = kwargs.get('xlabel', x_name)
    kwargs['ylabel'] = kwargs.get('ylabel', metric_name)

    # n_splits lines, summary over curves plotted
    plt_show, ax = pu.get_subplot_axis(ax)
    pu.make_summary_plot(sampled_split_curves_lists, ax, **kwargs)
    if plt_show: plt.show()


# -------------------- Grid Plots: Sample from Set -------------------------

def multi_line_sample_plot_grid(curves, n_sample=None, n_grid=4, **kwargs):
    #def multi_line_sample_plot(curves, ax=None, n_sample=None, **kwargs):

    # Setup Grid and Args
    axes = make_grid_figure(
        plc_list[0].subplot_shape, plc_list[0].figsize, n_grid,
    )
    kwargs_list = get_kwargs_lists_for_grid(kwargs, n_grid)

    # Plot Grid
    for grid_ind, ax in enumerate(axes.flatten()):
        multi_line_sample_plot(
            curves, ax=ax, n_sample=nsample, **kwargs_list[grid_ind],
        )

    # Annotate and Show
    annotate_grid_figure(axes, get_plt_cfg(**kwargs))
    plt.show()


# -------------------- Grid Plots: Compare Sets -------------------------

    

# -------------------- Comparative Grid Plots -------------------------
    
