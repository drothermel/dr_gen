import re
from collections import defaultdict

import numpy as np
import pandas as pd

# Constants for validation
EXPECTED_MATCHING_HPMS = 2
EXPECTED_KEY_VALUE_PARTS = 2

import dr_gen.analyze.bootstrapping as bu

# === Helpers to Select Relevant Run Groups === #


def make_hpm_specs(
    lr=0.1,
    wd=1e-4,
    epochs=270,
):
    """Create hyperparameter specifications dictionary.

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
    """Get runs comparing pretrained vs random initialization.

    Args:
        rg: Run group containing experiment data
        hpm_specs: Hyperparameter specifications
        split: Data split to use
        metric: Metric to compare
        one_per: Whether to select one run per initialization

    Returns:
        Tuple of (pretrained_runs, random_runs)
    """
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
        assert len(hpms_pre) == 1, f"Expected 1 pretrained HPM, got {len(hpms_pre)}"
        assert len(hpms_rand) == 1, f"Expected 1 random HPM, got {len(hpms_rand)}"
    return hpms_pre, hpms_rand


def select_matching_hpms(
    hpms_a,
    hpms_b,
    hpm_whitelist=None,
    ignore_key="model.weights",
):
    hpms_to_use = set()

    # Group hpms by hash
    hpms_by_hash = defaultdict(list)
    for hpm in [*hpms_a, *hpms_b]:
        # Verify hpm in whitelist
        if hpm_whitelist is not None and hpm not in hpm_whitelist:
            continue

        # Get the hash which ignores key if specified
        new_hash = hash(hpm)
        if ignore_key is not None:
            new_important_vals = {
                k: v for k, v in hpm.important_values.items() if k != ignore_key
            }
            new_hash = hash(hpm.as_tupledict(important_vals=new_important_vals))

        hpms_by_hash[new_hash].append(hpm)

    # Only use hpms where the hash has a value for group A and B
    for hash_hpms in hpms_by_hash.values():
        if len(hash_hpms) == EXPECTED_MATCHING_HPMS:
            hpms_to_use.update(hash_hpms)

    return hpms_to_use


def get_compare_runs_pretrain_vs_random(
    rg,
    hpm_select_dict,
):
    # Get the runs by hpm from val for hpm select and eval for metric calc
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

    # Filter to only hpms that exist both splits of both settings
    hpms_to_use = select_matching_hpms(
        hpms_a=hpms_val_pre.keys(),
        hpms_b=hpms_val_rand.keys(),
        hpm_whitelist=[*hpms_eval_pre.keys(), *hpms_eval_rand.keys()],
        ignore_key="model.weights",  # the hpm we're comparing between
    )
    hpms_val_pre = {str(h): d for h, d in hpms_val_pre.items() if h in hpms_to_use}
    hpms_eval_pre = {str(h): d for h, d in hpms_eval_pre.items() if h in hpms_to_use}
    hpms_val_rand = {str(h): d for h, d in hpms_val_rand.items() if h in hpms_to_use}
    hpms_eval_rand = {str(h): d for h, d in hpms_eval_rand.items() if h in hpms_to_use}
    return (hpms_val_pre, hpms_eval_pre), (hpms_val_rand, hpms_eval_rand)


def find_best_hpm_for_group(group_val_data, group_name, num_bootstraps):
    """Finds the best hyperparameter set and timestep for a given group
    based on validation data.
    tuple: (best_experiment_name, best_timestep) or (None, None) if failed.
    """
    if not group_val_data:
        return None, None

    bootstrapped_val = bu.bootstrap_experiment_timesteps(
        group_val_data, num_bootstraps=num_bootstraps
    )
    summary_stats_val = bu.calc_multi_stat_bootstrap_summary(bootstrapped_val)
    best_exp_name, best_timestep = bu.select_best_hpms(summary_stats_val)
    return best_exp_name, best_timestep


# TODO: this is what I want to call in the script
# After using get_compare_runs_pretrain_vs_random() to get the group data
def run_comparison_eval(
    group_a_data,
    group_b_data,
    group_a_name,
    group_b_name,
    val_num_bootstraps=1000,
    eval_num_bootstraps=1000,
    num_permutations=1000,
):
    val_a, eval_a = group_a_data
    val_b, eval_b = group_b_data
    best_hpm_a, best_ts_a = find_best_hpm_for_group(
        val_a, group_a_name, val_num_bootstraps
    )
    best_hpm_b, best_ts_b = find_best_hpm_for_group(
        val_b, group_b_name, val_num_bootstraps
    )

    # Check if best HPMs were found
    if best_hpm_a is None or best_hpm_b is None:
        # Return indicating failure but providing partial results if available
        return (best_hpm_a, best_ts_a, best_hpm_b, best_ts_b, None)

    # --- Perform comparison on evaluation data using ONLY the best HPMs found ---
    # Get the evaluation data for the *specific* best HPMs and timesteps
    eval_data_a = eval_a.get(best_hpm_a)
    eval_data_b = eval_b.get(best_hpm_b)

    # Ensure we have data for the best HPMs in the evaluation set
    if eval_data_a is None or eval_data_b is None:
        return (best_hpm_a, best_ts_a, best_hpm_b, best_ts_b, None)

    comparison_results = bu.compare_experiments_bootstrap(
        eval_data_a,  # just the best hpm data, all timesteps
        eval_data_b,  # just the best hpm data, all timesteps
        hpm_a=best_hpm_a,  # for naming
        hpm_b=best_hpm_b,  # for naming
        timestep_a=best_ts_a,  # to select best timestep from eval data
        timestep_b=best_ts_b,  # to select best timestep from eval data
        num_bootstraps=eval_num_bootstraps,
        num_permutations=num_permutations,
    )
    return (best_hpm_a, best_ts_a, best_hpm_b, best_ts_b, comparison_results)


def print_results_report(
    best_hpm_a, best_ts_a, best_hpm_b, best_ts_b, comparison_results
):
    """Prints a formatted report summarizing the analysis results.

    Args:
        best_hpm_a (str): Name of the best HPM for Group A.
        best_ts_a (int): Best validation timestep for Group A.
        best_hpm_b (str): Name of the best HPM for Group B.
        best_ts_b (int): Best validation timestep for Group B.
        comparison_results (dict): Results from the evaluation comparison.
    """
    # Report Difference Statistics
    final_diff_stats = comparison_results["difference_stats"]
    final_diff_stats["mean_diff_point_estimate"]
    final_diff_stats["mean_diff_ci_95"]
    final_diff_stats["mean_diff_reject_null_ci_95"]

    # Report KS Permutation Test Results
    final_ks_perm_stats = comparison_results["ks_permutation_test"]
    final_ks_perm_stats.get("observed_ks", "N/A")
    final_ks_perm_stats.get("p_value", "N/A")
    final_ks_perm_stats.get("reject_null", None)  # Default to None


# ================  CSV Export  ===================


# --- Helper Functions for HPM Parsing ---


def parse_hpm_value(value_str):
    """Attempts to convert a string value to float, int, bool, or None."""
    try:
        # Attempt integer conversion first
        return int(value_str)
    except ValueError:
        try:
            # Attempt float conversion
            return float(value_str)
        except ValueError:
            # Check for boolean/None strings (case-insensitive)
            lower_val = value_str.lower()
            if lower_val == "none":
                return "NONE"
            if lower_val == "true":
                return True
            if lower_val == "false":
                return False
            # Return the original string if no conversion applies
            return value_str


def parse_group_name(group_name_str):
    """Parses a group name string (e.g., "key1=val1 key2=val2") into a
    dictionary of hyperparameters.
    """
    hpm_dict = {}
    # Split by space, handling potential multiple spaces and stripping whitespace
    parts = re.split(r"\s+", group_name_str.strip())
    for part in parts:
        if not part:  # Skip empty strings resulting from multiple spaces
            continue
        # Split by '=', ensuring only the first '=' is used
        key_value = part.split("=", 1)
        if len(key_value) == EXPECTED_KEY_VALUE_PARTS:
            key, value_str = key_value
            hpm_dict[key] = parse_hpm_value(value_str)
        else:
            # Log a warning if a part cannot be parsed as key=value
            pass
    return hpm_dict


# --- Convert run data into dataframes ---


# Helper
def add_hpm_columns(df):
    """Parses the 'group_name' column and adds HPMs as separate columns.

    Args:
        df (pd.DataFrame): The initial DataFrame with a 'group_name' column.

    Returns:
        pd.DataFrame: DataFrame with HPM columns added/merged.
    """
    if df.empty or "group_name" not in df.columns:
        return df  # Return original df if empty or missing column

    # Apply the parsing function to get a Series of dictionaries
    hpm_series = df["group_name"].apply(parse_group_name)

    # Convert the Series of dictionaries into a DataFrame
    hpm_df = pd.DataFrame(hpm_series.tolist(), index=df.index)

    # Join the new HPM columns to the original DataFrame
    # Use suffixes if there's an unlikely column name collision, though joining
    # on index should be fine
    return df.join(hpm_df)


def run_groups_to_df(valid_groups, min_steps):
    """Builds intermediate NumPy arrays for columns based on valid groups and min_steps,
    then concatenates them and creates the initial Pandas DataFrame.

    Args:
        valid_groups (dict): Dictionary of valid group names to 2D NumPy arrays.
        min_steps (int): The number of steps/columns to trim each array to.

    Returns:
        pd.DataFrame: The initial DataFrame with raw data ('group_name',
                      'run_index', 'step_index', 'metric_value'). Returns
                      an empty DataFrame if valid_groups is empty.
    """
    if not valid_groups:
        return pd.DataFrame()

    all_group_names, all_run_indices, all_step_indices, all_metric_values = (
        [],
        [],
        [],
        [],
    )

    # Build lists of NumPy arrays for each column
    for group_name, run_metrics_array in valid_groups.items():
        num_runs = run_metrics_array.shape[0]
        trimmed_array = run_metrics_array[:, :min_steps]
        num_data_points = num_runs * min_steps  # Total points for this group

        # Generate column data using repeat/tile
        group_col = np.repeat(group_name, num_data_points)
        run_idx_col = np.repeat(np.arange(num_runs), min_steps)
        step_idx_col = np.tile(np.arange(min_steps), num_runs)
        metric_col = trimmed_array.flatten()  # Flatten trimmed array

        # Append arrays for this group to the main lists
        all_group_names.append(group_col)
        all_run_indices.append(run_idx_col)
        all_step_indices.append(step_idx_col)
        all_metric_values.append(metric_col)

    # Check if any data was actually processed before concatenating
    if not all_group_names:
        return pd.DataFrame()

    # Concatenate the data from all groups
    final_group_names = np.concatenate(all_group_names)
    final_run_indices = np.concatenate(all_run_indices)
    final_step_indices = np.concatenate(all_step_indices)
    final_metric_values = np.concatenate(all_metric_values)

    # Create the initial DataFrame
    df = pd.DataFrame(
        {
            "group_name": final_group_names,
            "run_index": final_run_indices,
            "step_index": final_step_indices,
            "metric_value": final_metric_values,
        }
    )
    return add_hpm_columns(df)
