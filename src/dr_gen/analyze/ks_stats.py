import numpy as np
import scipy.stats as stats


def find_max_diff_point(values1, values2):
    v1 = np.sort(np.array(values1))
    v2 = np.sort(np.array(values2))

    # Combine both arrays and get unique sorted values.
    all_vals = np.sort(np.unique(np.concatenate([v1, v2])))

    results = {}
    # For each unique value, compute the proportion of values in each sample that are <= that value.
    results["all_vals"] = all_vals
    results["cdf1"] = np.searchsorted(v1, all_vals, side="right") / len(v1)
    results["cdf2"] = np.searchsorted(v2, all_vals, side="right") / len(v2)

    # Compute absolute differences between the two CDFs.
    differences = np.abs(results["cdf1"] - results["cdf2"])
    results["max_idx"] = np.argmax(differences)
    results["max_diff_value"] = all_vals[results["max_idx"]]
    results["ks_stat"] = differences[results["max_idx"]]
    return results


def calculate_ks_for_run_sets(vals_1, vals_2):
    # Compute the KS statistic and p-value.
    ks_stat, p_value = stats.ks_2samp(vals_1, vals_2)

    # Identify contributing samples
    results = find_max_diff_point(vals_1, vals_2)
    results["p_value"] = p_value
    results["seeds_group_1"] = len(vals_1)
    results["seeds_group_2"] = len(vals_2)

    # Print Main Stats
    # cdf1_val = results["cdf1"][results["max_idx"]]
    # cdf2_val = results["cdf2"][results["max_idx"]]
    # print(
    #    f"ks_stat: {ks_stat:0.4f}, p_value: {p_value:0.4e} | max_val: {results['max_diff_value']:0.2f}, cdf1_val: {cdf1_val:0.4f}, cdf2_val: {cdf2_val:0.4f}"
    # )
    return results
