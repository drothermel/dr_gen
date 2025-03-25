import numpy as np
import random
import dr_gen.analyze.ks_stat as ks


def runs_metrics_to_ndarray(runs_metrics):
    if isinstance(runs_metrics, np.ndarray):
        return runs_metrics
    min_run_len = min([len(rm) for rm in runs_metrics])
    metric_array = np.array([rm[:min_run_len] for rm in runs_metrics])
    return metric_array # num_runs x T_min


# TODO
def summary_stats(data, stat=None):
    # [D] => [[D]] if needed
    # TODO: grab from old fxn, add new, only stat if not none, return {name: float or nd.array}
    # n, mean, median, min, max, variance, std, sem, 2.5th, 25th, 75th, 97.5th, IQR, CDF
    # float only if squeeze produces a single elem
    return {}

# TODO
def comparative_stats(estims_a, estims_b):
    return {}

# ============ Bootstrapping ==============

# For all functions
# n - sample size
# b - num bootstrap samples

# accepts ndarray or list of lists
def bootstrap_samples(dataset, n, b):
    return np.random.choice(dataset, size=(b, n), replace=True)

# accepts ndarray or list of lists
def bootstrap_summary_stats(dataset, n=None, b=None, stat=None):
    if b is None and n is None:
        samples = dataset
    elif b is None:
        samples = dataset[:n]
    else:
        samples = bootstrap_samples(dataset, n, b)
    stats_dists = summary_stats(samples, stat=stat)
    b_estims = {
        "dist": stats_dists,
        "point": {},
        "std": {},
        "sem": {},
        "ci_95": {},
    }
    for stat, dist in stats_dists.items():
        if dist.dim  != 1:
            continue

        b_estims['point'][stat] = np.mean(dist)
        b_estims['std'][stat] = np.stddev(dist)
        b_estims['sem'][stat] = b_estims['std'][std] / dist.shape[0]
        b_estims['ci_95'][stat] = (np.percentile(2.5, dist), np.percentile(97.5, dist))
    return b_estims

# runs metrics: [runs [metric_data ...]]
def bootstrap_early_stopping(runs_metrics, n=None, b=None):
    metric_array = runs_metrics_to_ndarray(runs_metrics)

    t_metrics = []
    for t in range(metric_array.shape[-1]):
        t_vals = metric_array[:, t]
        b_estims = bootstrap_summary_stats(t_vals, n, b, stat="mean")
        t_metrics.append(b_estims['point']['mean'])
    best_t = np.argmax(t_metrics)
    best_vals_mean = t_metrics[best_t]
    return best_t, best_vals_mean

def bootstrap_select_hpms(runs_metrics_by_hpm, early_stopping=False, n=None, b=None):
    hpm_metrics = []
    for hpm, runs_metrics in runs_metrics_by_hpm.items():
        metric_array = runs_metrics_to_ndarray(runs_metrics)
        if early_stopping:
            best_t, best_t_mean = bootstrap_early_stopping(metric_array, n, b)
        else:
            best_t = metric_array.shape[-1]
            best_t_vals = metric_array[:, best_t] if n is None else metric_array[:n, best_t]
            best_t_mean = best_t_vals.mean()
        hpm_metrics.append((hpm, best_t, best_t_mean))
    hpm, best_t, _ = max(hpm_metrics, key=lambda t: t[-1])
    return hpm, best_t

def bootstrap_best_hpms_stats(
    runs_metrics_for_eval,
    runs_metrics_for_hpm_select=None,
    early_stopping=False,
    n=None,
    b=None,
):
    # Hpm Selection
    if runs_metrics_for_hpm_select is None:
        # No actual hpm selection if no hpm select data provided
        assert len(runs_metrics_for_eval) == 1
        hpm, best_t = bootstrap_select_hpms(
            runs_metrics_for_eval, early_stopping=False, n=None, b=None,
        )
    else:
        hpm, best_t = bootstrap_select_hpms(
            runs_metrics_for_hpm_select,
            early_stopping=early_stopping,
            n=n,
            b=b,
        )

    # Make sure selected hpm is available for evaluation
    assert hpm in runs_metrics_for_eval

    t_vals = metric_array[:, best_t]
    b_estims = bootstrap_summary_stats(t_vals, n, b)
    b_estims['vals'] = t_vals
    b_estims['n'] = n
    b_estims['b'] = b
    return (hpm, best_t), b_estims


# =======================

def select_runs_by_hpms(run_data, hpms):
    if run_data is None:
        return None
    return {h: rm for h, rm in run_data.items() if h in hpms}

def trim_runs_metrics_dict(runs_metrics_dict, nmax, tmax):
    if runs_metrics_dict is None:
        return None
    return {h: [m[:tmax] for m in rm[:nmax]] for h, rm in run_metrics_dict.items()}

## Next: Use bootstrap_best_hpmps_stats to get ((hpm, best_t), summary_stats)
#           for all the values of (n, t) in the sweep
#        Take the estimated values to calculate individual hpm set stats and
#           then calculate comparative hpm set stats

def boostrap_compare_stats(
    runs_metrics_for_eval,
    hpms_A,
    hpms_B,
    runs_metrics_for_hpm_select=None,
    early_stopping=False,
    n=None,
    b=None,
):
    (best_hpm_a, best_t_a), estims_a = bootstrap_best_hpms_stats(
        select_runs_by_hpms(runs_metrics_for_eval, hpms_A),
        select_runs_by_hpms(runs_metrics_for_hpm_select, hpms_A),
        early_stopping=early_stopping, n=n, b=b,
    )
    (best_hpm_b, best_t_b), estims_b = bootstrap_best_hpms_stats(
        select_runs_by_hpms(runs_metrics_for_eval, hpms_B),
        select_runs_by_hpms(runs_metrics_for_hpm_select, hpms_B),
        early_stopping=early_stopping, n=n, b=b,
    )
    comp_estims = comparative_stats(estims_a, estims_b)
    return {
        'best_hpm_a': best_hpm_a,
        'best_t_a': best_t_a,
        'best_hpm_b': best_hpm_b,
        'best_t_b': best_t_b,
        'summary_stats_a': estims_a,
        'summary_stats_b': estims_b,
        'comparative_stats': comp_estims,
    }


# =======================

def sweep_t_n_compare(
    runs_metrics_for_eval,
    runs_metrics_for_hpm_select,
    hpms_A,
    hpms_B,
    nmax,
    tmax,
    early_stopping=False,
    b=None,
):
    all_stats = {}
    for n in range(nmax):
        for t in range(tmax):
            nt = (n, t)
            all_stats[nt] = bootstrap_compare_stats(
                trim_runs_metrics_dict(runs_metrics_for_eval, nmax, tmax),
                trim_runs_metrics_dict(runs_metrics_for_hpm_select, nmax, tmax),
                hpms_A,
                hpms_B,
                early_stopping=early_stopping, n=n, b=b,
            )
    return all_stats

# Default Tmax and Nmax set to magic vals
def sweep_t_n_compare_acc_by_init_default_hpms(
    rg, Tmax=270, Nmax=99, lr=0.1, wd=1e-4, b=None,
):
    hpm_select_dict = {
        "optim.lr": lr,
        "optim.weight_decay": wd,
        'epochs': Tmax,
    }

    # No hpm selection via sweep
    # So use val split for metric calculation
    #   { hpm: [runs [metric_data ...]]}
    hpm_select_runs_metrics_by_hpm = None
    metric_calculation_runs_metric_by_hpm = rg.select_run_split_metrics_by_hpms(
        "acc1",
        "val",
        **hpm_select_dict,
    )
    assert len(metric_calculation_runs_metric_by_hpm) == 2
    hpmA, hpmB = list(metric_calculation_runs_metric_by_hpm.keys())

    sample_inds_by_n = sweep_n_sample_inds(Nmax)
    hpms_A = [hpmA]
    hpms_B = [hpmB]

    return sweep_t_n_compare_metrics(
        hpm_select_runs_metrics_by_hpm,
        metric_calculation_runs_metric_by_hpm,
        hpms_A,
        hpms_B,
        Tmax,
        Nmax,
        early_stopping=False,
        b=b,
    )


def sweep_t_n_compare_acc_by_init_hpm_select(
    rg, Tmax=270, Nmax=99,
    LRs=[0.04, 0.06, 0.1, 0.16, 0.25],
    WDs=[0.00016, 4e-05, 0.00025, 6.3e-0.5, 1e-05, 0.0001],
    b=None,
):
    hpm_select_dict = {
        "optim.lr": LRs,
        "optim.weight_decay": WDs,
        'epochs': Tmax,
    }

    # Val split for hpm selection
    #   { hpm: [runs [metric_data ...]]}
    hpm_select_runs_metrics_by_hpm = rg.select_run_split_metrics_by_hpms(
        "acc1",
        "val",
        **hpm_select_dict,
    )

    # Eval split for metric calculation
    #   { hpm: [runs [metric_data ...]]}
    metric_calculation_runs_metric_by_hpm = rg.select_run_split_metrics_by_hpms(
        "acc1",
        "eval",
        **hpm_select_dict,
    )

    all_hpms = [hpm_select_runs_metrics_by_hpm.keys()]
    hpms_A = [hpm for hpm in all_hpms if hpm['model.weights'] == 'DEFAULT']
    hpms_B = [hpm for hpm in all_hpms if hpm['model.weights'] != 'DEFAULT']

    return sweep_t_n_compare_metrics(
        hpm_select_runs_metrics_by_hpm,
        metric_calculation_runs_metric_by_hpm,
        hpms_A,
        hpms_B,
        Tmax,
        Nmax,
        early_stopping=True,
        b=b,
    )
    
