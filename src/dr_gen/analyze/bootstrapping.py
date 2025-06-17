import math

import numpy as np
from scipy.stats import ks_2samp

# Constants for dimensional validation and statistical analysis
EXPECTED_2D_DIMENSIONS = 2
EXPECTED_3D_DIMENSIONS = 3
ALPHA_SIGNIFICANCE_LEVEL = 0.05
PERCENTILE_2_5 = 2.5
PERCENTILE_97_5 = 97.5
PERCENTILE_25 = 25
PERCENTILE_75 = 75
DEFAULT_MAX_LIST_PRINT_LEN = 3

# Modern NumPy random generator
_RNG = np.random.default_rng()

# --- Helpers for prepping data for bootstrapping ---


def get_min_2d_data_shape(data_dict):
    """Takes { exp_name: list_of_lists }.

    Finds the minimum lengths of outer (R) and inner (T) lists.
    Returns (min_R, min_T) or None if empty dict/lists found.
    """
    if not data_dict:
        return None
    min_first = math.inf  # Will hold min_R
    min_second = math.inf  # Will hold min_T
    for outer_list in data_dict.values():
        if not outer_list:
            return None  # Check runs list
        min_first = min(min_first, len(outer_list))
        for inner_list in outer_list:
            if not inner_list:
                return None  # Check timesteps list
            min_second = min(min_second, len(inner_list))
    # Safeguard: Ensure minimums were actually found
    if math.inf in (min_first, min_second):
        return None
    return (int(min_first), int(min_second))  # (min_R, min_T)


def make_uniform_2d_data_arrays_pair(data_a, data_b):
    """Make numpy arrays, determine min common dimensions and crop."""
    arr_a, arr_b = np.array(data_a, dtype=float), np.array(data_b, dtype=float)
    assert arr_a.ndim == EXPECTED_2D_DIMENSIONS
    assert arr_b.ndim == EXPECTED_2D_DIMENSIONS
    assert arr_a.size != 0
    assert arr_b.size != 0
    R = min(arr_a.shape[0], arr_b.shape[0])  # noqa: N806
    T = min(arr_a.shape[1], arr_b.shape[1])  # noqa: N806
    return arr_a[:R, :T], arr_b[:R, :T], R, T


def make_uniform_2d_data_arrays_dict(data_dict):
    """Converts dict of lists of lists to dict of numpy arrays with uniform shape.

    Returns None if input is empty or contains empty lists.
    Shape will be (min_R, min_T).
    """
    min_dims = get_min_2d_data_shape(data_dict)
    if min_dims is None:
        return None

    R, L = min_dims  # Num runs, Length Runs  # noqa: N806
    result_arrays = {}
    for key, outer_list in data_dict.items():
        truncated_data = [inner[:L] for inner in outer_list[:R]]
        result_arrays[key] = np.array(truncated_data, dtype=float)
    return result_arrays


# --- Bootstrapping Functions ---


def bootstrap_samples_batched(dataset, b=None):
    """Generates bootstrap samples for each row of a 2D array (the batch dimension).

    Samples elements with replacement from the second dimension (the data dimension).
    Input shape: (batch_dim, data_dim)
    Output shape: (batch_dim, num_samples_b, data_dim).
    """
    assert isinstance(dataset, np.ndarray)
    if len(dataset.shape) == 1:
        dataset = dataset[np.newaxis, :]
    if b is None:
        return dataset[:, np.newaxis]  # Shape (B, 1, L)

    B, L = dataset.shape  # Batch Size, Data Length  # noqa: N806
    indices = _RNG.integers(0, L, size=(B, b, L))
    batch_indices = np.arange(B)[:, np.newaxis, np.newaxis]
    return dataset[batch_indices, indices]


def bootstrap_experiment_timesteps(data_dict, num_bootstraps=None):
    """Performs batched bootstrapping on timestep data across runs for multiple experiments.

    "

    Standardizes experiments to (min_R, min_T), transposes to (min_T, min_R),
    stacks experiments, performs batched bootstrapping on the runs (R) for each
    timestep (T) across all experiments simultaneously, then unpacks results.

    Args:
        data_dict (dict): Dict mapping experiment names (str) to lists of lists
                          (R runs x T timesteps).
                          Example: {'exp1': [[r1t1, r1t2], [r2t1, r2t2]]} (R=2, T=2)
        num_bootstraps (int): The number of bootstrap samples (B) to generate
                              for each timestep's distribution across runs.

    Returns:
        dict or None: A dictionary mapping experiment names to 3D NumPy arrays
                      of shape (T, B, R) - representing (min_timesteps, num_bootstraps,
                      min_runs), containing the bootstrapped metric distributions for
                      each timestep. Returns None if the initial data processing fails
                      (e.g., empty lists).
    """
    uniform_arrays = make_uniform_2d_data_arrays_dict(data_dict)
    if uniform_arrays is None or not uniform_arrays:
        return None
    R, T = next(iter(uniform_arrays.values())).shape  # noqa: N806

    # === Reshape to make one batched bootstrap samples call ===
    experiment_names = list(uniform_arrays.keys())
    E = len(experiment_names)  # Number of experiments  # noqa: N806

    # Consider that we have R sampled metrics for each timestep T
    #   Transpose each array from (min_R, min_T) to (min_T, min_R)
    transposed_arrays = [uniform_arrays[name].T for name in experiment_names]

    stacked_data = np.stack(transposed_arrays, axis=0)

    reshaped_data = stacked_data.reshape(-1, R)

    # == Perform Batched Bootstrapping ==
    # Input shape: (batch=E*T, data=R), Output shape: (batch=E*T, B, data=R)
    bootstrapped_combined = bootstrap_samples_batched(reshaped_data, b=num_bootstraps)

    # == Separate Bootstrapped Data Back by Experiment ==
    # Reshape back to separate experiments along the first dimension
    # Shape: (E, T, B, R) where B = num_bootstraps
    reshaped_bootstrapped = bootstrapped_combined.reshape(E, T, num_bootstraps, R)
    result_dict = {}
    for i, name in enumerate(experiment_names):
        result_dict[name] = reshaped_bootstrapped[i]
    return result_dict


# --- Statistics Calculation Functions ---


def calc_stats_across_bootstrap_samples(timestep_data):
    """Calculates multiple base statistics for each bootstrap sample (vectorized).

    Takes a np array for a single timestep shape (B, R).
    Returns dict mapping stat names to 1D arrays (shape B,).
    """
    B, R = timestep_data.shape  # Num bootstrap samples, Num Runs  # noqa: N806
    stats = {}
    # Use errstate to handle potential warnings (e.g., std dev of single value if R=1)
    std = np.std(timestep_data, axis=1, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        all_percentiles = np.percentile(
            timestep_data,
            [PERCENTILE_2_5, PERCENTILE_25, PERCENTILE_75, PERCENTILE_97_5],
            axis=1,
        )
        stats["mean"] = np.mean(timestep_data, axis=1)
        stats["median"] = np.median(timestep_data, axis=1)
        stats["min"] = np.min(timestep_data, axis=1)
        stats["max"] = np.max(timestep_data, axis=1)
        stats["std_dev"] = std
        stats["sem_of_mean"] = std / np.sqrt(R) if R > 0 else np.full(B, np.nan)
        stats["percentile_2.5"] = all_percentiles[0, :]
        stats["percentile_25"] = all_percentiles[1, :]
        stats["percentile_75"] = all_percentiles[2, :]
        stats["percentile_97.5"] = all_percentiles[3, :]
        stats["iqr"] = all_percentiles[2, :] - all_percentiles[1, :]
    return stats


def summarize_distribution(dist):
    """Calculates summary stats (mean, std, CI) for a 1D bootstrap distribution.

    Takes: 1D array of data values
    Returns dict: 'point_estimate', 'spread', 'ci_95_lower', 'ci_95_upper'.
    """
    B = len(dist)  # noqa: N806
    # Handle cases where stats are undefined or unreliable
    if B <= 1:
        summary = {
            "point_estimate": np.nan,
            "spread": np.nan,
            "ci_95_lower": np.nan,
            "ci_95_upper": np.nan,
        }
        if B == 1:
            # Mean is defined for B=1, others are not.
            summary["point_estimate"] = dist[0]
        return summary

    # Use ddof=0 for population std dev
    ci_lower, ci_upper = np.percentile(dist, [PERCENTILE_2_5, PERCENTILE_97_5])
    return {
        "point_estimate": np.mean(dist),  # Mean of the B values
        "spread": np.std(dist, ddof=0),  # Std Dev, ddof=0 for pop std dev
        "ci_95_lower": ci_lower,  # 2.5th percentile of B values
        "ci_95_upper": ci_upper,  # 97.5th percentile of B values
    }


def calc_multi_stat_bootstrap_summary(bootstrapped_data):
    """Calculates bootstrap summary statistics for multiple stats.

    Per experiment and timestep.

    Takes: dict { exp_name: (T, B, R) numpy array }, timestep,
    bootstrap samples, replicas
    Returns:
      {
          'timestep': t,
          'mean_point_estimate': ...,
          'mean_spread': ...,
          'mean_ci_95_lower': ...,
          'mean_ci_95_upper': ...,
          'mean_distribution': np.array([...]), # Shape (B,)
          'median_point_estimate': ...,
          'median_spread': ...,
          # ... etc. for all base statistics ...
      }.
    """
    final_results = {}
    for exp_name, exp_data in bootstrapped_data.items():
        # Validate the data structure for the current experiment
        if (
            not isinstance(exp_data, np.ndarray)
            or exp_data.ndim != EXPECTED_3D_DIMENSIONS
        ):
            continue
        T, B, R = (  # noqa: N806
            exp_data.shape
        )  # Timesteps, Bootstrap samples, Runs/Replicates
        if B <= 1 or R == 0:
            continue

        timestep_results_list = []
        for t in range(T):
            timestep_data = exp_data[t, :, :]  # (B, R)

            # --- Calculate base statistics distributions ---
            # Returns dict: {'mean': array(B,), 'median': array(B,), ...}
            stats_dists = calc_stats_across_bootstrap_samples(timestep_data)

            # --- Summarize each distribution ---
            t_result = {"timestep": t}
            for stat_name, dist in stats_dists.items():
                summary = summarize_distribution(dist)
                t_result[f"{stat_name}_distribution"] = dist
                t_result[f"{stat_name}_point_estimate"] = summary["point_estimate"]
                t_result[f"{stat_name}_spread"] = summary["spread"]
                t_result[f"{stat_name}_ci_95_lower"] = summary["ci_95_lower"]
                t_result[f"{stat_name}_ci_95_upper"] = summary["ci_95_upper"]
            timestep_results_list.append(t_result)
        final_results[exp_name] = timestep_results_list
    return final_results


def select_best_hpms(summary_stats_data):
    """Selects the best hyperparameter set (experiment) and the best timestep.

    By maximizing the 'mean_point_estimate'.
    Takes: the output of calc_multi_stat_bootstrap_summary
    Returns: (best_experiment_name (str), best_timestep (int)).
    """
    overall_best_score = -math.inf
    best_experiment_name = None
    best_timestep_for_best_exp = None

    # Iterate through each experiment (hyperparameter set)
    for exp_name, timestep_data_list in summary_stats_data.items():
        # Find the timestep summary with the maximum 'mean_point_estimate' for
        # this experiment
        best_ts_summary_for_exp = max(
            timestep_data_list, key=lambda ts: ts["mean_point_estimate"]
        )

        # Get the score and timestep for this experiment's best
        current_best_score_for_exp = best_ts_summary_for_exp["mean_point_estimate"]
        current_best_timestep_for_exp = best_ts_summary_for_exp["timestep"]

        # Check if this experiment's best score is the overall best so far
        if current_best_score_for_exp > overall_best_score:
            overall_best_score = current_best_score_for_exp
            best_experiment_name = exp_name
            best_timestep_for_best_exp = current_best_timestep_for_exp

    return (best_experiment_name, best_timestep_for_best_exp)


# Calculates point differences and bootstrap CIs for the differences.
def calc_diff_stats_and_ci(summary_stats_a, summary_stats_b):
    """Calculates point differences and bootstrap CIs for the difference between statistics.

    The inputs are lists of timestep summary dicts for each exp.
    """
    stat_names = [
        k.replace("_point_estimate", "")
        for k in summary_stats_a
        if k.endswith("_point_estimate")
    ]
    diff = {}
    for stat_name in stat_names:
        # Get point estimates, default to NaN if key missing
        pa = summary_stats_a.get(f"{stat_name}_point_estimate", np.nan)
        pb = summary_stats_b.get(f"{stat_name}_point_estimate", np.nan)
        diff[f"{stat_name}_diff_point_estimate"] = pa - pb

        # Get distributions for CI calculation, default to None if key missing
        dist_a = summary_stats_a.get(f"{stat_name}_distribution")
        dist_b = summary_stats_b.get(f"{stat_name}_distribution")

        # Calculate difference dist, confidence interval
        diff_dist = dist_a - dist_b
        ci_lower, ci_upper = np.percentile(diff_dist, [PERCENTILE_2_5, PERCENTILE_97_5])
        diff[f"{stat_name}_diff_ci_95"] = (ci_lower, ci_upper)

        # Null hypothesis: difference is zero. Reject if 0 is outside CI.
        diff[f"{stat_name}_diff_reject_null_ci_95"] = not (ci_lower <= 0 <= ci_upper)
    return diff


def calc_ks_stat_and_summary(bdata_a, bdata_b, num_bootstraps):
    """Calculates bootstrap confidence interval for the KS statistic between matched samples.

    Takes: bootstrapped data for one timestep, shape (B, R).
    Returns: List of results per timestep.
    """
    # Calculate KS stat for each matched bootstrap sample pair
    ks_dist = np.empty(num_bootstraps)
    for b in range(num_bootstraps):
        ks_dist[b], _ = ks_2samp(bdata_a[b, :], bdata_b[b, :])

    # Get the dist summary
    summary = summarize_distribution(ks_dist)
    return {
        "ks_stat_distribution": ks_dist,
        "ks_stat_point_estimate": summary["point_estimate"],
        "ks_stat_spread": summary["spread"],
        "ks_stat_ci_95_lower": summary["ci_95_lower"],
        "ks_stat_ci_95_upper": summary["ci_95_upper"],
    }


def perform_ks_permutation_test(arr_a, arr_b, R, num_permutations):  # noqa: N803
    """Performs a bootstrap permutation test using the KS statistic.

    Args:
        arr_a (np.ndarray): Standardized original data for exp A, shape (R).
        arr_b (np.ndarray): Standardized original data for exp B, shape (R).
        R (int): Number of runs.
        num_permutations (int): Number of permutations to generate the null
            distribution.
        alpha (float): Significance level for p-value comparison.

    Returns:
        dict: Containing the observed KS, p-value, and null hypothesis rejection
            decision.
    """
    # Calculate the observed KS statistic on the original data
    observed_ks, _ = ks_2samp(arr_a, arr_b)

    # Combine original data for permutation and generate null dist
    combined_data = np.concatenate((arr_a, arr_b))  # Shape (2R,)
    null_ks_dist = np.empty(num_permutations)

    # Shuffle, split and calc ks stat on null dist
    shuffled = np.copy(combined_data)
    for p in range(num_permutations):
        _RNG.shuffle(shuffled)
        null_ks_dist[p], _ = ks_2samp(shuffled[:R], shuffled[R:])

    # Calculate p-value: proportion of null KS stats >= observed KS stat
    p_value = np.mean(null_ks_dist >= observed_ks)
    reject_null = p_value < ALPHA_SIGNIFICANCE_LEVEL
    return {
        "observed_ks": observed_ks,
        "p_value": p_value,
        "reject_null": reject_null,
    }


def compare_experiments_bootstrap(
    data_a_raw,
    data_b_raw,
    hpm_a,
    hpm_b,
    timestep_a,
    timestep_b,
    num_bootstraps=1000,
    num_permutations=1000,
):
    # Make data uniform and select best timestep
    arr_a, arr_b, R, T = make_uniform_2d_data_arrays_pair(data_a_raw, data_b_raw)  # noqa: N806
    arr_a = arr_a[:, timestep_a : timestep_a + 1]
    arr_b = arr_b[:, timestep_b : timestep_b + 1]

    # Bootstrap samples
    bdata_a = bootstrap_samples_batched(arr_a.T, b=num_bootstraps)  # Shape (T, B, R)
    bdata_b = bootstrap_samples_batched(arr_b.T, b=num_bootstraps)  # Shape (T, B, R)

    # Calculate statistics
    summary_a_list = calc_multi_stat_bootstrap_summary({"exp_A": bdata_a})["exp_A"]
    summary_b_list = calc_multi_stat_bootstrap_summary({"exp_B": bdata_b})["exp_B"]
    assert len(summary_a_list) == 1, (
        f"Expected 1 summary for A, got {len(summary_a_list)}"
    )
    assert len(summary_b_list) == 1, (
        f"Expected 1 summary for B, got {len(summary_b_list)}"
    )
    summary_a = summary_a_list[0]
    summary_b = summary_b_list[0]

    exp_a_name = f"{hpm_a}_t{timestep_a}"
    exp_b_name = f"{hpm_b}_t{timestep_b}"
    print_bootstrap_summary_exp_results(
        exp_a_name,
        summary_a,
    )
    print_bootstrap_summary_exp_results(
        exp_b_name,
        summary_b,
    )

    # Calculate difference results
    diff_results = calc_diff_stats_and_ci(summary_a, summary_b)
    ks_stat_results = calc_ks_stat_and_summary(
        bdata_a.squeeze(),
        bdata_b.squeeze(),
        num_bootstraps=num_bootstraps,
    )
    ks_permutation_test_results = perform_ks_permutation_test(
        arr_a.squeeze(), arr_b.squeeze(), R=R, num_permutations=num_permutations
    )

    return {
        "hpm_A": hpm_a,
        "timestep_A": timestep_a,
        "original_A": arr_a.squeeze(),
        "bootstraps_A": bdata_a.squeeze(),
        "summary_A": summary_a,
        "hpm_B": hpm_b,
        "timestep_B": timestep_b,
        "original_B": arr_b.squeeze(),
        "bootstraps_B": bdata_b.squeeze(),
        "summary_B": summary_b,
        "difference_stats": diff_results,
        "ks_ci_test": ks_stat_results,
        "ks_permutation_test": ks_permutation_test_results,
    }


# --- Utility Functions ---
def print_bootstrap_summary_exp_results(
    _exp_name,  # Unused but kept for API compatibility
    exp_result,
    _max_list_print_len=DEFAULT_MAX_LIST_PRINT_LEN,  # Unused but kept for API compatibility
):
    for key, value in exp_result.items():
        if key == "timestep":  # Already printed
            continue

        if isinstance(value, np.ndarray | list):
            continue
        if isinstance(value, tuple):
            pass
        elif isinstance(value, int | float | bool | str) or value is None:
            # Print primitive types directly
            if isinstance(value, float):
                pass  # Format floats
            else:
                pass
