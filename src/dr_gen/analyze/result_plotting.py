import random
import dr_gen.analyze.ks_stat as ks

def get_bootstrap_sample_inds(Nmax, n, k):
    all_inds = list(range(len(Nmax)))
    sample_inds = []
    for i in range(k):
        sample_inds.append(random.select(all_inds, n))
    return sample_inds

# TODO: figure this out
def bootstrap_k(Nmax, n):
    print(">> WARNING: this isn't correctly impld")
    diff = Nmax - n
    if diff < 2:
        return 1
    return 1 if diff <= 2 else diff - 2

# Returns: [k_samples [n_run_inds ...]]
def sweep_n_sample_inds(Nmax):
    all_n_sample_inds = {}
    for n in range(1, Nmax):
        k = bootstrap_k(Nmax, n)
        all_n_smaple_inds[n] = get_bootstrap_sample_inds(Nmax, n, k)
    return all_n_sample_inds


# runs_metrics:     [runs [metric_data ...]]
# sampled_run_inds: [sampled_sets [run_inds ...]]
# Returns:          [sampled_sets [runs [metric_data ...]]
def select_sample_runs_metrics(runs_metrics, sampled_run_inds):
    runs_metrics_samples = []
    for run_inds_list in sampled_run_inds:
        runs_metrics_samples.append(
            [runs_metrics[ri] for ri in run_inds_set]
        )
    return runs_metrics_samples


# runs_metrics_samples: [sampled_sets [runs [metric_data ...]]
# Returns:              [sampled_sets [run_metric_at_t ...]]
def select_epoch_from_sampled_metrics(runs_metrics_samples, t):
    epoch_sampled_metrics = []
    for runs_metrics in runs_metrics_samples:
        epoch_sampled_metrics.append(
            [rm[t] for rm in runs_metrics]
        )
    return epoch_sampled_metrics

# runs_metrics_by_hpm:  { hpm: [runs [metric_data ...]] }
# sample_inds_by_n:  { n: [sampled_sets [run_inds ...]] }
# Returns:              [sampled_sets [run_metric_at_t ...]]
def get_hpm_t_n_sampled_final_val(
    runs_metrics_by_hpm,
    sample_inds_by_n,
    hpm,
    t_eval,
    n,
):
    hpm_sampled_runs = select_sample_runs_metrics(
        runs_metrics_by_hpm[hpm], sample_inds_by_n[n],
    )
    return select_epoch_from_sampled_metrics(hpm_sampled_runs, t_eval)


# TODO: figure this out
# Takes:    [sampled_sets [runs_metric_at_t ...]]
# Returns:  value_estim, [vals_used_for_estim ....]
def sampled_mean(runs_final_metrics):
    print(">> WARN: unimpld")
    return 0, [0 for _ in range(len(runs_final_metrics))]

# TODO: figure this out
# Takes:    [sampled_sets [runs_metric_at_t ...]]
# Returns:  value_estim, [vals_used_for_estim ....]
def sampled_std(runs_final_metrics):
    print(">> WARN: unimpld")
    return 0, [0 for _ in range(len(runs_final_metrics))]

# TODO: figure this out
# Takes:    [sampled_sets [runs_metric_at_t ...]]
# Returns:  value_estim, [vals_used_for_estim ....]
def sampled_gaussian_error(runs_final_metrics):
    print(">> WARN: unimpld")
    return 0, [0 for _ in range(len(runs_final_metrics))]

# TODO: figure this out
# Takes:    [sampled_sets [runs_metric_at_t ...]] for 2 run types
# Returns:  value_estim, [vals_used_for_estim ....]
def sampled_ks_stats(runs_A_final_metrics, runs_B_final_metrics):
    print(">> WARN: unimpld")
    return 0, [0 for _ in range(len(runs_A_final_metrics))]

# Max sample size to consider is minimum of hpm provided sample sizes
def get_max_n_from_hpm_runs(hpm_runs):
    hpm_avail_num_runs = [
        len(runs) for runs in hpm_runs.values()
    ]
    Nmax = min(hpm_avail_num_runs)
    return Nmax

# Takes:    { hpms: [sampled_sets [runs [metric_data ...]]] }, Tmax
def hpm_select(sampled_runs_metrics_by_hpm, Tmax):
    hpm_t_to_metric = {}
    best_hpm_t = None
    for hpm, sampled_runs in sampled_runs_metrics_by_hpm.items():
        for t in range(1, Tmax):
            hpm_t_final_vals = select_epoch_from_sampled_metrics(sampled_runs, t)
            acc_hpm_t, _ = sample_mean(hpm_t_final_vals)
            hpm_t_to_metric[(hpm, t)] = acc_hpm_t
            if best_hpm_t is None or best_hpm_t[1] < acc_hpm_t:
                best_hpm_t = ((hpm, t), acc_hpm_t)
    return best_hpm_t, hpm_t_to_metric

    

# Takes:    [sampled_sets [runs_metric_at_t ...]]
def calc_run_metrics(runs_final_metrics):
    mean_estim, means = sample_mean(runs_final_metrics)
    std_estim, stds = sample_std(runs_final_metrics)
    gerror_estim, gaussian_errors = sample_gaussian_errors(runs_final_metrics)
    return {
        'mean_estimate': mean_estim,
        'means': means,
        'std_estimate': std,
        'stds': stds,
        'gaussian_error_estimate': gerror_estim,
        'guassian_errors': gaussian_errors,
    }

def calc_compare_metrics(final_metrics_and_results_A, final_metrics_and_results_B):
    final_metrics_A, results_A = final_metrics_and_results_A
    final_metrics_B, results_B = final_metrcis_and_results_B

    compare_metrics = {}
    compare_metrics['ks_stats_diff'] = sampled_ks_stats(final_metrics_A, final_metrics_B)
    compare_metrics['mean_diff'] = results_A['mean'] - results_B['mean']
    return compare_metrics
    
# *_runs_metrics_by_hpm:    { hpm: [runs [metric_data ...]] }
# Returns:                  { t: { n: results } }
def sweep_t_n_compare_metrics(
    hpm_select_runs_metrics_by_hpm,
    metric_caclulation_runs_metric_by_hpm,
    sample_inds_by_n,
    hpms_A,
    hpms_B,
    Tmax,
    Nmax,
):
    all_results = {t: {} for t in range(1, Tmax)}

    hpms = [hpms_A, hpms_B]
    for t in range(1, Tmax):
        for n in range(1, Nmax):
            htn_vals_and_metrics = []
            best_hpms_and_t_evals = []
            for hpms in [hpms_A, hpms_B]:
                # Select HPM and Evaluation Epoch (hpm selection if dataset provided)
                if hpm_select_runs_metrics_by_hpm is None:
                    best_hpm = hpms[0]
                    best_t_eval = Tmax
                    best_hpms_and_t_evals.append((best_hpm, best_t_eval))
                else:
                    hpm_select_tn_sampled_metrics = {
                        hpm: select_sample_run_metrics(
                            hpm_select_runs_metrics_by_hpm[hpm],
                            sample_inds_by_n[n],
                        ) for hpm in hpms
                    }
                    (best_hpm_t, ),, _ = hpm_select(
                        hpm_select_tn_sampled_metrics, Tmax,
                    )
                    best_hpm, best_t_eval = best_hpm_t
                    best_hpms_and_t_evals.append(best_hpm_t)

                # Use best_hpm and best_t_eval to cacluate metrics
                htn_final_vals = get_hpm_t_n_final_val_samples(
                    metric_calculation_runs_metrics_by_hpm,
                    sample_inds_by_n,
                    best_hpm,
                    n,
                    best_t_eval,
                )
                htn_metrics = calc_run_metrics(htn_final_vals)
                htn_vals_and_metrics.append((htn_final_vals, htn_metrics))

            # Calculate compare metrics
            htn_compare_metrics = calc_compare_metrics(*htn_vals_and_metrics)
            all_results[t][n] = {
                'n': n,
                't': t,
                'best_hpms_and_t_evals': best_hpms_and_t_evals,
                'htn_vals_and_metrics': htn_vals_and_metrics,
                'htn_compare_metrics': htn_compare_metrics,
            }
    return all_results, hpms

# Default Tmax and Nmax set to magic vals
def sweep_t_n_compare_acc_by_init_default_hpms(
    rg, Tmax=270, Nmax=99, lr=0.1, wd=1e-4,
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
    )


def sweep_t_n_compare_acc_by_init_hpm_select(
    rg, Tmax=270, Nmax=99,
    LRs=[0.04, 0.06, 0.1, 0.16, 0.25],
    WDs=[0.00016, 4e-05, 0.00025, 6.3e-0.5, 1e-05, 0.0001],
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
    )
    
