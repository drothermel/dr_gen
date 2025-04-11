import math
import numpy as np
import pandas as pd
import re
import dr_gen.analyze.bootstrapping as bu

# === Helpers to Select Relevant Run Groups === #

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

def select_matching_hpms(
    hpms_A, hpms_B, hpm_whitelist=None, ignore_key='model.weights',
):
    hpms_to_use = set()

    # Group hpms by hash
    hpms_by_hash = defaultdict(list)
    for hpm in [*hpms_A, *hpms_B]:
        # Verify hpm in whitelist
        if hpm_whitelist is not None and hpm not in hpm_whitelist:
            print(hpm, "not in whitelist")
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
        if len(hash_hpms) == 2:
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
        hpms_A=hpms_val_pre.keys(),
        hpms_B=hpms_val_rand.keys(),
        hpm_whitelist=[*hpms_eval_pre.keys(), *hpms_eval_rand.keys()],
        ignore_key='model.weights', # the hpm we're comparing between
    )
    hpms_val_pre = {str(h): d for h, d in hpms_val_pre.items() if h in hpms_to_use}
    hpms_eval_pre = {str(h): d for h, d in hpms_eval_pre.items() if h in hpms_to_use}
    hpms_val_rand = {str(h): d for h, d in hpms_val_rand.items() if h in hpms_to_use}
    hpms_eval_rand = {str(h): d for h, d in hpms_eval_rand.items() if h in hpms_to_use}
    return (hpms_val_pre, hpms_eval_pre), (hpms_val_rand, hpms_eval_rand)

def find_best_hpm_for_group(group_val_data, group_name, num_bootstraps):
    """
    Finds the best hyperparameter set and timestep for a given group
    based on validation data.
    tuple: (best_experiment_name, best_timestep) or (None, None) if failed.
    """
    if not group_val_data:
        return None, None

    bootstrapped_val = bu.bootstrap_experiment_timesteps(group_val_data, num_bootstraps=num_bootstraps)
    summary_stats_val = bu.calc_multi_stat_bootstrap_summary(bootstrapped_val)
    best_exp_name, best_timestep = bu.select_best_hpms(summary_stats_val)
    print(f">> Best HPM for {group_name}: {best_exp_name} (based on validation timestep {best_timestep})")
    return best_exp_name, best_timestep

# TODO: this is what I want to call in the script
# After using get_compare_runs_pretrain_vs_random() to get the gorup data
def run_comparison_eval(
    group_a_data, group_b_data,
    group_a_name, group_b_name,
    val_num_bootstraps=1000, eval_num_bootstraps=1000, num_permutations=1000,
):
    val_a, eval_a = group_a_data
    val_b, eval_b = group_b_data
    best_hpm_A, best_ts_A = find_best_hpm_for_group(val_a, group_a_name, val_num_bootstraps)
    best_hpm_B, best_ts_B = find_best_hpm_for_group(val_b, group_b_name, val_num_bootstraps)
    comparison_results = bu.compare_experiments_bootstrap(
        eval_a,
        eval_b,
        num_bootstraps=eval_num_bootstraps,
        num_permutations=num_permutations,
    )
    return (
        best_hpm_A,
        best_ts_A,
        best_hpm_B,
        best_ts_B,
        comparison_results
    )

def print_results_report(best_hpm_A, best_ts_A, best_hpm_B, best_ts_B, comparison_results):
    """
    Prints a formatted report summarizing the analysis results.
    Args:
        best_hpm_A (str): Name of the best HPM for Group A.
        best_ts_A (int): Best validation timestep for Group A.
        best_hpm_B (str): Name of the best HPM for Group B.
        best_ts_B (int): Best validation timestep for Group B.
        comparison_results (dict): Results from the evaluation comparison.
    """
    print("\n--- Results Report ---")
    print(f"Best Hyperparameters (selected using Validation data):")
    print(f"  Group A: {best_hpm_A} (Best performance observed at validation timestep {best_ts_A})")
    print(f"  Group B: {best_hpm_B} (Best performance observed at validation timestep {best_ts_B})")

    print("\nComparison on Evaluation Data (using selected HPMs):")

    # Report Difference Statistics
    final_diff_stats = comparison_results['difference_stats'][0] # each hpm uses best ts
    mean_diff = final_diff_stats['mean_diff_point_estimate']
    mean_diff_ci = final_diff_stats['mean_diff_ci_95']
    mean_reject_null = final_diff_stats['mean_diff_reject_null_ci_95']

    print(f"\nPerformance Difference (Group A - Group B) at Final Evaluation Timestep:")
    print(f"  Mean Metric Difference: {mean_diff:.4f}")
    print(f"  95% CI for Mean Difference: ({mean_diff_ci[0]:.4f}, {mean_diff_ci[1]:.4f})")
    print(f"  Statistically Significant Difference (via CI)? {'Yes' if mean_reject_null else 'No'}")

    # Report KS Permutation Test Results
    final_ks_perm_stats = comparison_results['ks_permutation_test'][0]
    ks_observed = final_ks_perm_stats.get('observed_ks', 'N/A')
    ks_p_value = final_ks_perm_stats.get('p_value', 'N/A')
    ks_reject_null = final_ks_perm_stats.get('reject_null', None) # Default to None

    print(f"\nKS Permutation Test (Comparing Distributions) at Final Evaluation Timestep:")
    print(f"  Observed KS Statistic: {ks_observed:.4f}")
    print(f"  P-value: {ks_p_value:.4f}")
    print(f"  Statistically Significant Difference (p < 0.05)? {'Yes' if ks_reject_null else 'No'}")


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
            if lower_val == 'none':
                return "NONE"
            if lower_val == 'true':
                return True
            if lower_val == 'false':
                return False
            # Return the original string if no conversion applies
            return value_str

def parse_group_name(group_name_str):
    """Parses a group name string (e.g., "key1=val1 key2=val2") into a dictionary of hyperparameters."""
    hpm_dict = {}
    # Split by space, handling potential multiple spaces and stripping whitespace
    parts = re.split(r'\s+', group_name_str.strip())
    for part in parts:
        if not part: # Skip empty strings resulting from multiple spaces
            continue
        # Split by '=', ensuring only the first '=' is used
        key_value = part.split('=', 1)
        if len(key_value) == 2:
            key, value_str = key_value
            hpm_dict[key] = parse_hpm_value(value_str)
        else:
            # Log a warning if a part cannot be parsed as key=value
            print(f"Warning: Could not parse part '{part}' as key=value in group name '{group_name_str}'")
            # Optionally assign a default value or skip:
            # hpm_dict[part] = None # Example: Assign None if parsing fails
    return hpm_dict


# --- Convert run data into dataframes --- 

# Helper
def add_hpm_columns(df):
    """
    Parses the 'group_name' column and adds HPMs as separate columns.

    Args:
        df (pd.DataFrame): The initial DataFrame with a 'group_name' column.

    Returns:
        pd.DataFrame: DataFrame with HPM columns added/merged.
    """
    if df.empty or 'group_name' not in df.columns:
        return df # Return original df if empty or missing column

    try:
        # Apply the parsing function to get a Series of dictionaries
        hpm_series = df['group_name'].apply(parse_group_name)

        # Convert the Series of dictionaries into a DataFrame
        hpm_df = pd.DataFrame(hpm_series.tolist(), index=df.index)

        # Join the new HPM columns to the original DataFrame
        # Use suffixes if there's an unlikely column name collision, though joining on index should be fine
        df_with_hpms = df.join(hpm_df)
        return df_with_hpms

    except Exception as e:
        print(f"Error during hyperparameter parsing: {e}. Returning DataFrame without HPM columns.")
        # Optionally drop the group_name column even if parsing failed
        if not keep_group_name_col and 'group_name' in df.columns:
             return df.drop(columns=['group_name'])
        return df # Return the original DataFrame in case of error


def run_groups_to_df(valid_groups, min_steps):
    """
    Builds intermediate NumPy arrays for columns based on valid groups and min_steps,
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

    all_group_names, all_run_indices, all_step_indices, all_metric_values = [], [], [], []

# Build lists of NumPy arrays for each column
    for group_name, run_metrics_array in valid_groups.items():
        num_runs = run_metrics_array.shape[0]
        trimmed_array = run_metrics_array[:, :min_steps]
        num_data_points = num_runs * min_steps # Total points for this group

        # Generate column data using repeat/tile
        group_col = np.repeat(group_name, num_data_points)
        run_idx_col = np.repeat(np.arange(num_runs), min_steps)
        step_idx_col = np.tile(np.arange(min_steps), num_runs)
        metric_col = trimmed_array.flatten() # Flatten trimmed array

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
    df = pd.DataFrame({
        'group_name': final_group_names,
        'run_index': final_run_indices,
        'step_index': final_step_indices,
        'metric_value': final_metric_values
    })
    hpm_df = add_hpm_columns(df)
    return hpm_df
