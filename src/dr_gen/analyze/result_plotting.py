import numpy as np
import scipy.stats

ESTIM_TYPES = ["point", "std", "sem", "ci_95"]


def expand_to_dim(array, dim, axis=0):
    while len(array.shape) < dim:
        if axis == 0:
            array = array[np.newaxis, :]
        else:
            array = array[:, np.new_axis]
    return array


def runs_metrics_to_ndarray(runs_metrics):
    """Convert list of run metrics to numpy array, trimming to shortest run length."""
    if isinstance(runs_metrics, np.ndarray):
        return runs_metrics
    min_run_len = min([len(rm) for rm in runs_metrics])
    return np.array([rm[:min_run_len] for rm in runs_metrics])


def runs_metrics_dict_to_ndarray_dict(runs_metrics_dict):
    """Convert dictionary of run metrics to dictionary of numpy arrays."""
    if runs_metrics_dict is None:
        return None
    runs_ndarrays = {}
    for hpm, rms in runs_metrics_dict.items():
        runs_ndarrays[hpm] = runs_metrics_to_ndarray(rms)
    return runs_ndarrays  # {hpm: num_runs x T_min}


def select_runs_by_hpms(run_data, hpms):
    """Filter run data to only include specified hyperparameter combinations."""
    if run_data is None:
        return None
    return {h: rm for h, rm in run_data.items() if h in hpms}


def trim_runs_metrics_dict(runs_metrics_dict, nmax, tmax):
    """Trim run metrics to specified number of runs and time steps."""
    if runs_metrics_dict is None:
        return None
    return {h: [m[:tmax] for m in rm[:nmax]] for h, rm in runs_metrics_dict.items()}

def np_to_float(val):
    if isinstance(val, tuple):
        return tuple(np_to_float(vv) for vv in val)
    if isinstance(val, np.ndarray) and val.size == 1:
        return val.item()

    try:
        return float(val)
    except:
        return val


def dist_estim(estim_type, dist):
    assert estim_type in ESTIM_TYPES
    assert isinstance(dist, np.ndarray)
    estim = None
    match estim_type:
        case "point":
            estim = np.mean(dist)
        case "std":
            estim = np.std(dist)
        case "sem":
            estim = np.std(dist) / dist.shape[0]
        case "ci_95":
            estim = (np.percentile(dist, 2.5).item(), np.percentile(dist, 97.5).item())
    assert estim is not None, f"Add {estim_type} to match"
    return np_to_float(estim)


def dist_estims(dist):
    return {estim: dist_estim(estim, dist) for estim in ESTIM_TYPES}


def summary_stats(data):
    """
    Calculate comprehensive summary statistics for bootstrap samples.

    Args:
        data: 2D numpy array (bootstrap_samples x observations)

    Returns:
        Dictionary of statistics or single statistic value
    """
    # Number of observations in each bootstrap sample
    # (here assuming no missing values; otherwise, one could use np.sum(~np.isnan(data), axis=1))
    n = np.full(data.shape[0], data.shape[1])

    # Basic descriptive statistics (using nan-aware functions if needed)
    mean = np.nanmean(data, axis=1)
    median = np.nanmedian(data, axis=1)
    _min = np.nanmin(data, axis=1)
    _max = np.nanmax(data, axis=1)

    # Variance and standard deviation (with Bessel's correction)
    variance = np.nanvar(data, axis=1, ddof=1)
    std = np.nanstd(data, axis=1, ddof=1)

    # Standard Error of the Mean
    sem = std / np.sqrt(n)

    # Quantiles: compute 2.5th, 25th, 75th, and 97.5th percentiles along each row.
    percentiles = np.nanpercentile(data, [2.5, 25, 75, 97.5], axis=1)
    q2p5, q25, q75, q97p5 = (
        percentiles[0],
        percentiles[1],
        percentiles[2],
        percentiles[3],
    )

    # Interquartile range
    iqr = q75 - q25

    # Empirical Cumulative Distribution Function (CDF):
    # Here, we return the sorted values for each bootstrap sample.
    sorted_vals = np.sort(data, axis=1)

    # Prepare the dictionary of statistics. We squeeze the outputs if they are single elements.
    stats = {
        "sorted_vals": sorted_vals,  # still 2D: one row per bootstrap sample, each sorted in ascending order
        "n": np.squeeze(n),
        "mean": np.squeeze(mean),
        "median": np.squeeze(median),
        "min": np.squeeze(_min),
        "max": np.squeeze(_max),
        "variance": np.squeeze(variance),
        "std": np.squeeze(std),
        "sem": np.squeeze(sem),
        "2.5th": np.squeeze(q2p5),
        "25th": np.squeeze(q25),
        "75th": np.squeeze(q75),
        "97.5th": np.squeeze(q97p5),
        "IQR": np.squeeze(iqr),
    }
    stats = {k: np_to_float(v) for k, v in stats.items()}
    return stats


def comparative_stats(estims_a, estims_b):
    """Compare statistics between two sets of bootstrap estimates."""

    dists_key = "dists_of_diff_estims"
    diff_suffix = "_a_minus_b"
    all_stats = {
        dists_key: {},
        **{estim_type: {} for estim_type in ESTIM_TYPES},
    }

    # Calculate Explicit Difference Dist: KS Statistic
    ks_stats = []
    p_vals = []
    vals_a = estims_a["bootstrap_dists"]["sorted_vals"]
    vals_b = estims_b["bootstrap_dists"]["sorted_vals"]
    for vs_a, vs_b in zip(vals_a, vals_b):
        ks, p = scipy.stats.ks_2samp(vs_a, vs_b)
        ks_stats.append(ks)
        p_vals.append(p)
    all_stats[dists_key]["ks_stats"] = np.array(ks_stats)
    all_stats[dists_key]["ks_p_values"] = np.array(p_vals)

    # Calculate Difference in Summary Stats
    for stat in estims_a["bootstrap_dists"]:
        if stat in ["n", "sorted_vals"]:
            continue
        stat_key = f"{stat}{diff_suffix}"
        stat_diff = (
            estims_a["bootstrap_dists"][stat] - estims_b["bootstrap_dists"][stat]
        )
        all_stats[dists_key][stat_key] = np_to_float(stat_diff)

    # Calculate estims of difference summary stats
    for stat, diff_dist in all_stats[dists_key].items():
        diff_summary_stats = dist_estims(diff_dist)
        for estim_type, val in diff_summary_stats.items():
            all_stats[estim_type][stat] = np_to_float(val)

    # Add the Values from Original Data for Difference Estimates
    orig_ks_stat, orig_ks_stat_p_val = scipy.stats.ks_2samp(
        estims_a["original_data"], estims_b["original_data"]
    )
    all_stats["original_data_stats"] = {
        "ks_stats": np_to_float(orig_ks_stat),
        "ks_p_val": np_to_float(orig_ks_stat_p_val),
    }
    for stat in estims_a["original_data_stats"]:
        stat_diff_key = f"{stat}{diff_suffix}"
        all_stats["original_data_stats"][stat_diff_key] = np_to_float(
            estims_a["original_data_stats"][stat]
            - estims_b["original_data_stats"][stat]
        )
    return all_stats


# ============ Bootstrapping ==============

# For all functions
# - Assume n = len(dataset)
# - b = num bootstrap samples


# accepts ndarray or list of lists
def bootstrap_samples(dataset, b=None):
    """
    Generate bootstrap samples from dataset.

    Args:
        dataset: Input data to bootstrap, np array
        b: Number of bootstrap samples (None for single sample)

    Returns:
        Array of bootstrap samples
    """
    # No bootstrap: one sample, the original dataset
    ds = np.squeeze(dataset)
    if b is None:
        return expand_to_dim(ds, 2, axis=0)
    n = ds.shape[-1]
    return np.random.choice(ds, size=(b, n), replace=True)


# accepts ndarray or list of lists
def bootstrap_summary_stats(dataset, num_bootstraps=None):
    """
    Calculate bootstrap summary statistics with uncertainty estimates.

    Args:
        dataset: Input data to analyze
        num_bootstraps: Number of bootstrap samples
        stat: Optional specific statistic to return

    Returns:
        Dictionary containing point estimates, standard errors, and confidence intervals
    """
    dataset = expand_to_dim(dataset, 2, axis=0)
    samples = bootstrap_samples(dataset, num_bootstraps)
    stats_dists = summary_stats(samples)
    b_estims = {
        "num_bootstraps": num_bootstraps,
        "bootstrap_dists": stats_dists,
        **{estim_type: {} for estim_type in ESTIM_TYPES},
    }
    for stat_name, dist in stats_dists.items():
        if len(dist.shape) != 1:
            continue
        # Point, Std, Sem, CI 95
        stat_summary_estims = dist_estims(dist)
        for estim_type, val in stat_summary_estims.items():
            b_estims[estim_type][stat_name] = val

    b_estims["original_data"] = dataset
    b_estims["original_data_stats"] = summary_stats(dataset)
    return b_estims


# runs metrics: [runs [metric_data ...]]
def bootstrap_early_stopping(runs_metrics, num_bootstraps=None):
    """
    Find optimal stopping time using bootstrap estimates.

    Args:
        runs_metrics: List of metric values across time steps
        num_bootstraps: Number of bootstrap samples

    Returns:
        Tuple of (best_time_step, best_metric_value)
    """
    metric_array = runs_metrics_to_ndarray(runs_metrics)

    t_metrics = []
    for t in range(metric_array.shape[-1]):
        t_vals = metric_array[:, t]
        b_estims = bootstrap_summary_stats(t_vals, num_bootstraps)
        mean_point_estim = b_estims["point"]["mean"]
        t_metrics.append(mean_point_estim)
    best_t = np.argmax(t_metrics)
    best_vals_mean = t_metrics[best_t]
    return best_t, best_vals_mean


def bootstrap_select_hpms(
    runs_metrics_by_hpm, early_stopping=False, num_bootstraps=None
):
    """
    Select best hyperparameter combination using bootstrap estimates.

    Args:
        runs_metrics_by_hpm: Dictionary mapping hyperparameters to metric values
        early_stopping: Whether to use early stopping for time step selection
        num_bootstraps: Number of bootstrap samples

    Returns:
        Tuple of (best_hpm, best_time_step)
    """
    hpm_metrics = []
    for hpm, runs_metrics in runs_metrics_by_hpm.items():
        metric_array = runs_metrics_to_ndarray(runs_metrics)
        if early_stopping:
            best_t, best_t_mean = bootstrap_early_stopping(metric_array, num_bootstraps)
        else:
            best_t = metric_array.shape[-1] - 1
            best_t_mean = metric_array[:, best_t].mean()
        hpm_metrics.append((hpm, best_t, best_t_mean))
    hpm, best_t, _ = max(hpm_metrics, key=lambda t: t[-1])
    return hpm, best_t


def bootstrap_best_hpms_stats(
    runs_metrics_for_eval,
    runs_metrics_for_hpm_select=None,
    early_stopping=False,
    num_bootstraps=None,
):
    """
    Calculate statistics for best hyperparameter combination.

    Args:
        runs_metrics_for_eval: Metrics for final evaluation
        runs_metrics_for_hpm_select: Optional metrics for HPM selection
        early_stopping: Whether to use early stopping
        num_bootstraps: Number of bootstrap samples

    Returns:
        Tuple of ((best_hpm, best_time), bootstrap_estimates)
    """
    # -- Hpm Selection -- #
    hpm_select_runs = runs_metrics_for_hpm_select
    # No Hpm Selection: verify only one hpm, turn off early stopping
    if hpm_select_runs is None:
        # No actual hpm selection if no hpm select data provided
        assert len(runs_metrics_for_eval) == 1
        hpm_select_runs = runs_metrics_for_eval
        early_stopping = False
    hpm, best_t = bootstrap_select_hpms(
        hpm_select_runs,
        early_stopping=early_stopping,
        num_bootstraps=num_bootstraps,
    )

    # Make sure selected hpm is available for evaluation
    assert hpm in runs_metrics_for_eval

    # Metric Calculation
    t_vals = runs_metrics_for_eval[hpm][:, best_t]
    b_estims = bootstrap_summary_stats(t_vals, num_bootstraps)
    return (hpm, best_t), b_estims


# =======================


def bootstrap_compare_stats(
    runs_metrics_for_eval,
    hpms_A,
    hpms_B,
    runs_metrics_for_hpm_select=None,
    early_stopping=False,
    num_bootstraps=None,
):
    """
    Compare statistics between two sets of hyperparameter combinations.

    Args:
        runs_metrics_for_eval: Metrics for final evaluation
        hpms_A, hpms_B: Sets of hyperparameters to compare
        runs_metrics_for_hpm_select: Optional metrics for HPM selection
        early_stopping: Whether to use early stopping
        num_bootstraps: Number of bootstrap samples

    Returns:
        Dictionary containing comparison results and statistics
    """
    eval_metrics_arrays = runs_metrics_dict_to_ndarray_dict(runs_metrics_for_eval)
    hpm_metrics_arrays = runs_metrics_dict_to_ndarray_dict(runs_metrics_for_hpm_select)
    (best_hpm_a, best_t_a), estims_a = bootstrap_best_hpms_stats(
        select_runs_by_hpms(eval_metrics_arrays, hpms_A),
        select_runs_by_hpms(hpm_metrics_arrays, hpms_A),
        early_stopping=early_stopping,
        num_bootstraps=num_bootstraps,
    )
    (best_hpm_b, best_t_b), estims_b = bootstrap_best_hpms_stats(
        select_runs_by_hpms(eval_metrics_arrays, hpms_B),
        select_runs_by_hpms(hpm_metrics_arrays, hpms_B),
        early_stopping=early_stopping,
        num_bootstraps=num_bootstraps,
    )
    comp_estims = comparative_stats(estims_a, estims_b)
    return {
        "best_hpm_a": best_hpm_a,
        "best_t_a": best_t_a,
        "runs_shape_a": eval_metrics_arrays[best_hpm_a].shape,
        "best_hpm_b": best_hpm_b,
        "best_t_b": best_t_b,
        "runs_shape_b": eval_metrics_arrays[best_hpm_b].shape,
        "summary_stats_a": estims_a,
        "summary_stats_b": estims_b,
        "comparative_stats": comp_estims,
    }


# ============== Eval Sweep General Helper ==============


def sweep_t_n_compare(
    runs_metrics_for_eval,
    runs_metrics_for_hpm_select,
    hpms_A,
    hpms_B,
    nmax,
    tmax,
    early_stopping=False,
    num_bootstraps=None,
):
    """
    Compare statistics across different numbers of runs and time steps.

    Args:
        runs_metrics_for_eval: Metrics for final evaluation
        runs_metrics_for_hpm_select: Metrics for HPM selection
        hpms_A, hpms_B: Sets of hyperparameters to compare
        nmax: Maximum number of runs to consider
        tmax: Maximum time step to consider
        early_stopping: Whether to use early stopping
        num_bootstraps: Number of bootstrap samples

    Returns:
        Dictionary mapping (n, t) tuples to comparison results
    """
    all_stats = {}
    for n in range(nmax):
        for t in range(tmax):
            nt = (n, t)
            all_stats[nt] = bootstrap_compare_stats(
                trim_runs_metrics_dict(runs_metrics_for_eval, n, t),
                trim_runs_metrics_dict(runs_metrics_for_hpm_select, n, t),
                hpms_A,
                hpms_B,
                early_stopping=early_stopping,
                num_bootstraps=num_bootstraps,
            )
    return all_stats


# ================  Specific Eval Setups & Helpers  ===================


def make_hpm_specs(
    lr=0.1,
    wd=1e-4,
    epochs=270,
):
    """
    Create hyperparameter specifications dictionary.

    Args:
        lr: Learning rate
        wd: Weight decay
        epochs: Number of epochs

    Returns:
        Dictionary of hyperparameter specifications
    """
    return {
        "optim.lr": lr,
        "optim.weight_decay": wd,
        "epochs": epochs,
    }


def get_pretrained_vs_random_init_runs(
    rg,
    hpm_specs,
    split,
    metric="acc1",
    one_per=True,
):
    """
    Get runs comparing pretrained vs random initialization.

    Args:
        rg: Run group containing experiment data
        hpm_specs: Hyperparameter specifications
        split: Data split to use
        metric: Metric to compare
        one_per: Whether to select one run per initialization

    Returns:
        Tuple of (pretrained_runs, random_runs)
    """
    # {hpm: runs_metrics}
    all_hpms = rg.select_run_split_metrics_by_hpms(
        metric,
        split,
        **hpm_specs,
    )
    hpms_pre = {
        hpm: d for hpm, d in all_hpms.items() if hpm["model.weights"] == "DEFAULT"
    }
    hpms_rand = {
        hpm: d for hpm, d in all_hpms.items() if hpm["model.weights"] != "DEFAULT"
    }
    if one_per:
        assert len(hpms_pre) == len(hpms_rand) == 1
    return hpms_pre, hpms_rand


def one_tn_no_hpm_select_compare_weight_init(
    rg,
    hpm_select_dict,
    t,
    n,
    num_bootstraps=1000,
    split="val",
):
    hpms_pre, hpms_rand = get_pretrained_vs_random_init_runs(
        rg,
        hpm_select_dict,
        split,
        one_per=True,
    )
    all_runs_data = {**hpms_pre, **hpms_rand}
    return bootstrap_compare_stats(
        runs_metrics_for_eval=trim_runs_metrics_dict(all_runs_data, n, t),
        runs_metrics_for_hpm_select=None,
        hpms_A=hpms_pre,
        hpms_B=hpms_rand,
        early_stopping=False,
        num_bootstraps=num_bootstraps,
    )


def one_tn_hpm_compare_weight_init(
    rg,
    hpm_select_dict,
    t,
    n,
    early_stopping=True,
    num_bootstraps=1000,
    split="val",
):
    hpms_val_pre, hpms_val_rand = get_pretrained_vs_random_init_runs(
        rg,
        hpm_select_dict,
        "val",
        one_per=False,
    )
    hpms_eval_pre, hpms_eval_rand = get_pretrained_vs_random_init_runs(
        rg,
        hpm_select_dict,
        "eval",
        one_per=False,
    )
    all_runs_val_data = {**hpms_val_pre, **hpms_val_rand}
    all_runs_eval_data = {**hpms_eval_pre, **hpms_eval_rand}
    runs_metrics_for_eval = trim_runs_metrics_dict(all_runs_eval_data, n, t)
    runs_metrics_for_hpm_select = trim_runs_metrics_dict(all_runs_val_data, n, t)
    return bootstrap_compare_stats(
        runs_metrics_for_eval=runs_metrics_for_eval,
        runs_metrics_for_hpm_select=runs_metrics_for_hpm_select,
        hpms_A=hpms_eval_pre,
        hpms_B=hpms_eval_rand,
        early_stopping=early_stopping,
        num_bootstraps=num_bootstraps,
    )


# Compare a single set of hpms across inits.
#  - metric: "acc1"
#
# Defaults Explanation
#  - use val split if no hpm select
#  - arbitrary num bootstraps
#  - tmax = num epochs for main run sets
#  - nmax = minimum num runs from largest hpm set
def sweep_tn_no_hpm_select_compare_weight_init(
    rg,
    hpm_select_dict,
    Tmax=270,
    Nmax=99,
    num_bootstraps=1000,
    split="val",
):
    """
    Compare weight initializations without hyperparameter selection.

    Args:
        rg: Run group containing experiment data
        hpm_select_dict: Hyperparameter specifications
        Tmax: Maximum time step
        Nmax: Maximum number of runs
        num_bootstraps: Number of bootstrap samples
        split: Data split to use

    Returns:
        Dictionary of comparison results across different n and t values
    """
    hpms_pre, hpms_rand = get_pretrained_vs_random_init_runs(
        rg,
        hpm_select_dict,
        split,
        one_per=True,
    )
    all_runs_data = {**hpms_pre, **hpms_rand}
    return sweep_t_n_compare(
        runs_metrics_for_eval=all_runs_data,
        runs_metrics_for_hpm_select=None,  # hpm select runs data
        hpms_A=hpms_pre,
        hpms_B=hpms_rand,
        nmax=Nmax,
        tmax=Tmax,
        b=num_bootstraps,
        early_stopping=False,  # no hpm select
    )


# Compare weight inits via hpm selelction from a sweep.
#  - metric: "acc1"
#  - use val split for hpm select, eval for metric calculation
#
# Defaults Explanation
#  - arbitrary num bootstraps
#  - tmax = num epochs for main run sets
#  - nmax = minimum num runs from most hpm sets
#  - early_stoppng = true by default, but controllable
def sweep_tn_hpm_compare_weight_init(
    rg,
    hpm_select_dict,
    Tmax=270,
    Nmax=20,
    num_bootstraps=1000,
    early_stopping=True,
):
    """
    Compare weight initializations with hyperparameter selection.

    Args:
        rg: Run group containing experiment data
        hpm_select_dict: Hyperparameter specifications
        Tmax: Maximum time step
        Nmax: Maximum number of runs
        num_bootstraps: Number of bootstrap samples
        early_stopping: Whether to use early stopping

    Returns:
        Dictionary of comparison results across different n and t values
    """
    hpms_val_pre, hpms_val_rand = get_pretrained_vs_random_init_runs(
        rg,
        hpm_select_dict,
        "val",
        one_per=False,
    )
    hpms_eval_pre, hpms_eval_rand = get_pretrained_vs_random_init_runs(
        rg,
        hpm_select_dict,
        "eval",
        one_per=False,
    )
    all_runs_val_data = {**hpms_val_pre, **hpms_val_rand}
    all_runs_eval_data = {**hpms_eval_pre, **hpms_eval_rand}
    return sweep_t_n_compare(
        runs_metrics_for_eval=all_runs_eval_data,
        runs_metrics_for_hpm_select=all_runs_val_data,
        hpms_A=hpms_val_pre,
        hpms_B=hpms_val_rand,
        nmax=Nmax,
        tmax=Tmax,
        b=num_bootstraps,
        early_stopping=early_stopping,
    )


def print_comparative_summary_stats(all_stats):
    best_hpm_a = all_stats["best_hpm_a"]
    shape_a = all_stats["runs_shape_a"]
    best_t_a = all_stats["best_t_a"]
    best_hpm_b = all_stats["best_hpm_b"]
    shape_b = all_stats["runs_shape_b"]
    best_t_b = all_stats["best_t_b"]

    estims_a = all_stats["summary_stats_a"]
    estims_b = all_stats["summary_stats_b"]
    comp = all_stats["comparative_stats"]

    num_bootstraps = estims_a["num_bootstraps"]

    print(">> :: Per Dist Summary Stats ::\n")
    print(f"Compare using {num_bootstraps} bootstraps:")
    print(f"  - [{shape_a} | best: {best_t_a}] {best_hpm_a}")
    print(f"  - [{shape_b} | best: {best_t_b}] {best_hpm_b}")
    print()
    for k, va in estims_a.items():
        if k in ["bootstrap_dists", "num_bootstraps", "original_data"]:
            continue
        print(k)
        if isinstance(va, dict):
            for kk, vva in va.items():
                if kk in ["sorted_vals"]:
                    continue
                vvb = estims_b[k][kk]
                if isinstance(vvb, tuple):
                    print(
                        f"   {kk:10} | ({vva[0]:.8f}, {vva[1]:.8f}) | ({vvb[0]:.8f}, {vvb[1]:.8f})"
                    )
                else:
                    print(f"   {kk:10} | {vva:.8f} | {vvb:.8f}")
        else:
            print("  ", va, estims_b[k])
    print()

    print(">> :: Comparative Summary Stats ::\n")
    print(" A Minus B: Negative means B is bigger")
    print(f"  - A: {best_hpm_a}")
    print(f"  - B: {best_hpm_b}")

    for k, vc in comp.items():
        if k in ['dists_of_diff_estims', 'original_data_stats']:
            continue
        print(k, type(vc))
        for kk, vv in vc.items():
            if isinstance(vv, tuple):
                print(f"  {kk:20} ({vv[0]:0.8f}, {vv[1]:0.8f})")
            else:
                print(f"  {kk:20} {vv:0.8f}")

