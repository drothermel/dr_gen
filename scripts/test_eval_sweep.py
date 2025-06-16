import pickle as pkl
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

import dr_gen.analyze.result_plotting as rplt
from dr_gen.analyze.run_group import RunGroup

# === Configuration ===
RUN_DATA_PATH = "/Users/daniellerothermel/drotherm/data/dr_gen/cifar10/cluster_runs/lr_wd_init_v0"
# T_STEP = 50  # Step size for sweeping through timesteps - REMOVED
# N_STEP = 5  # Step size for sweeping through runs - REMOVED
N_START = 3
T_START = 1
NUM_N_POINTS = 17 # Number of points to test along the N axis (runs)
NUM_T_POINTS = 270 # Number of points to test along the T axis (timesteps)
OUTPUT_CSV_PATH = f"/Users/daniellerothermel/Desktop/comparison_sweep_summary_v0_n{NUM_N_POINTS}_t{NUM_T_POINTS}.csv"
OUTPUT_PKL_PATH = f"/Users/daniellerothermel/Desktop/comparison_sweep_summary_v0_n{NUM_N_POINTS}_t{NUM_T_POINTS}.pkl"

# HPMs for selecting runs (same as test_eval.py)
hpm_specs_hpm_select = {
    "optim.lr": [0.04, 0.06, 0.1, 0.16, 0.25],
    "optim.weight_decay": [1e-05, 4e-05, 6.3e-05, 0.0001, 0.00016, 0.00025],
    "epochs": 270,
}

# Bootstrap/Permutation settings (can be kept lower for faster sweep)
VAL_NUM_BOOTSTRAPS = 1000
EVAL_NUM_BOOTSTRAPS = 1000
NUM_PERMUTATIONS = 1000

# === Helper Function ===

def trim_hpm_data(hpm_data_dict, n_runs, t_length):
    """Trims the data within a dictionary {hpm_name: [[run1_metrics], [run2_metrics], ...]}
    to the specified number of runs and timesteps.

    Args:
        hpm_data_dict (dict): The input dictionary of HPM data.
        n_runs (int): The maximum number of runs to keep.
        t_length (int): The maximum number of timesteps to keep per run.

    Returns:
        dict: A new dictionary with the trimmed data. Returns empty dict if input is empty.
    """
    if not hpm_data_dict:
        return {}

    trimmed_dict = {}
    for hpm_name, runs_data in hpm_data_dict.items():
        if not runs_data: # Skip if no runs for this hpm
            continue

        # Ensure runs_data is list of lists/arrays
        if not isinstance(runs_data, (list, np.ndarray)) or not all(isinstance(run, (list, np.ndarray)) for run in runs_data):
             print(f"Warning: Skipping HPM '{hpm_name}' due to unexpected data format: {type(runs_data)}")
             continue

        # Trim runs (outer list)
        actual_n = min(n_runs, len(runs_data))
        trimmed_runs = deepcopy(runs_data[:actual_n]) # Deep copy to avoid modifying original

        # Trim timesteps (inner lists)
        trimmed_hpm_data = []
        for run in trimmed_runs:
            if not isinstance(run, (list, np.ndarray)): # Should not happen based on check above, but safety first
                continue
            actual_t = min(t_length, len(run))
            trimmed_hpm_data.append(run[:actual_t])

        # Only add if there's valid data after trimming
        if trimmed_hpm_data and all(trimmed_hpm_data): # Ensure no empty inner lists remained
            trimmed_dict[hpm_name] = trimmed_hpm_data

    return trimmed_dict


# === Main Script Logic ===

# --- Load Data ---
print(f"Loading run data from: {RUN_DATA_PATH}")
rg = RunGroup()
rg.load_runs_from_base_dir(RUN_DATA_PATH)
print("Loaded runs.")

# --- Select Run Groups (Initial Selection) ---
print("Selecting runs for pretrained vs. random initialization comparison...")
(hpms_val_pre_full, hpms_eval_pre_full), (hpms_val_rand_full, hpms_eval_rand_full) = rplt.get_compare_runs_pretrain_vs_random(
    rg,
    hpm_select_dict=hpm_specs_hpm_select,
)

print(f"Found {len(hpms_val_pre_full)} matching HPM sets for Pretrained.")
print(f"Found {len(hpms_val_rand_full)} matching HPM sets for Random Init.")

if not hpms_val_pre_full or not hpms_val_rand_full:
    print("Error: Could not find matching hyperparameter sets for comparison. Exiting.")
    exit()

# --- Determine Sweep Limits ---
# Find min runs across all HPMs in both groups (use validation set as proxy)
min_runs_pre = min(len(runs) for runs in hpms_val_pre_full.values()) if hpms_val_pre_full else 0
min_runs_rand = min(len(runs) for runs in hpms_val_rand_full.values()) if hpms_val_rand_full else 0
max_n = min(min_runs_pre, min_runs_rand)

# Find min timesteps across all runs for all HPMs (use validation set as proxy)
min_ts_pre = min(len(ts) for runs in hpms_val_pre_full.values() for ts in runs) if hpms_val_pre_full else 0
min_ts_rand = min(len(ts) for runs in hpms_val_rand_full.values() for ts in runs) if hpms_val_rand_full else 0
max_t = min(min_ts_pre, min_ts_rand)

print(f"Determined sweep limits: max_n = {max_n}, max_t = {max_t}")

if max_n < 2 or max_t < 2: # Need at least 2 points to define a range
    print(f"Error: Insufficient data dimensions (max_n={max_n}, max_t={max_t}) for sweep with >1 points. Exiting.")
    exit()

# --- Generate Sweep Points ---
# Ensure at least 2 points if possible, otherwise use max_n/max_t
n_points_actual = min(NUM_N_POINTS, max_n-N_START+1)
t_points_actual = min(NUM_T_POINTS, max_t-T_START+1)

# Generate approximately evenly spaced integer points including 1 and max_n/max_t
if n_points_actual <= 1:
    n_values = np.array([max_n]) # Or maybe [1] if max_n > 0? Let's stick to max_n
else:
    n_values = np.unique(np.linspace(N_START, max_n, n_points_actual, dtype=int))

if t_points_actual <= 1:
    t_values = np.array([max_t])
else:
    t_values = np.unique(np.linspace(T_START, max_t, t_points_actual, dtype=int))

print(f"Will test N values: {n_values}")
print(f"Will test T values: {t_values}")

# --- Run Sweep ---
all_results = []
print(f"Starting sweep over {len(n_values)} n points and {len(t_values)} t points...")

# for n in range(1, max_n + 1, N_STEP): # OLD LOOP
for n in n_values:
    # Determine valid 't' range for this 'n' - trim function handles lengths
    # t_values = range(1, max_t + 1, T_STEP) # OLD T VALUES
    # if max_t not in t_values: # Ensure the max_t is included
    #     t_values = list(t_values) + [max_t]

    for t in t_values:
        print(f"  Running analysis for n={n}, t={t}...")

        # Trim data for this n, t combination
        hpms_val_pre = trim_hpm_data(hpms_val_pre_full, n, t)
        hpms_eval_pre = trim_hpm_data(hpms_eval_pre_full, n, t)
        hpms_val_rand = trim_hpm_data(hpms_val_rand_full, n, t)
        hpms_eval_rand = trim_hpm_data(hpms_eval_rand_full, n, t)

        # Check if trimming resulted in empty data for any group needed for comparison
        if not hpms_val_pre or not hpms_eval_pre or not hpms_val_rand or not hpms_eval_rand:
             print(f"    Skipping n={n}, t={t} due to insufficient data after trimming.")
             continue # Skip this iteration if any group is empty

        # Combine val/eval data correctly for run_comparison_eval
        group_a_data = (hpms_val_pre, hpms_eval_pre)
        group_b_data = (hpms_val_rand, hpms_eval_rand)

        try:
            # Run the comparison
            (
                best_hpm_A, best_ts_A, best_hpm_B, best_ts_B, comparison_results
            ) = rplt.run_comparison_eval(
                group_a_data=group_a_data,
                group_b_data=group_b_data,
                group_a_name="Pretrained",
                group_b_name="Random Init",
                val_num_bootstraps=VAL_NUM_BOOTSTRAPS,
                eval_num_bootstraps=EVAL_NUM_BOOTSTRAPS,
                num_permutations=NUM_PERMUTATIONS,
            )

            # --- Collect Results ---
            if comparison_results is None:
                print(f"    Warning: Comparison failed for n={n}, t={t}. Skipping result.")
                continue # Skip if comparison failed internally

            # Prepare data row for CSV (adapted from test_eval.py)
            csv_data = {
                "sweep_n": n,
                "sweep_t": t,
                "group_A_name": "Pretrained",
                "group_B_name": "Random Init",
                "best_hpm_A": best_hpm_A,
                "best_val_timestep_A": best_ts_A,
                "best_hpm_B": best_hpm_B,
                "best_val_timestep_B": best_ts_B,
            }

            # Flatten summary, difference, and test statistics
            stats_to_flatten = {
                "summary_A_": comparison_results.get("summary_A", {}),
                "summary_B_": comparison_results.get("summary_B", {}),
                "diff_": comparison_results.get("difference_stats", {}),
                "ks_ci_": comparison_results.get("ks_ci_test", {}),
                "ks_perm_": comparison_results.get("ks_permutation_test", {})
            }

            for prefix, stats_dict in stats_to_flatten.items():
                for key, value in stats_dict.items():
                    if isinstance(value, np.ndarray):
                        csv_data[f"{prefix}{key}"] = str(value.tolist()) # Stringify arrays
                    elif isinstance(value, tuple):
                        csv_data[f"{prefix}{key}_lower"] = value[0] if len(value) > 0 else np.nan
                        csv_data[f"{prefix}{key}_upper"] = value[1] if len(value) > 1 else np.nan
                    else:
                        csv_data[f"{prefix}{key}"] = value

            ## Add original and bootstrapped samples (converting to string)
            #for key_suffix in ['original', 'bootstraps']:
                #for group_suffix in ['A', 'B']:
                    #full_key = f'{key_suffix}_{group_suffix}'
                    #if full_key in comparison_results:
                        #data_array = comparison_results[full_key]
                        #if isinstance(data_array, np.ndarray):
                             #csv_data[full_key] = str(data_array.tolist())
                        #else:
                             #csv_data[full_key] = str(data_array) # Handle non-array cases if any

            all_results.append(csv_data)

        except (ValueError, KeyError, TypeError, AttributeError) as e:
            print(f"    Error during analysis for n={n}, t={t}: {e}")
            # Optionally add a row indicating error or just skip
            # error_row = {'sweep_n': n, 'sweep_t': t, 'error': str(e)}
            # all_results.append(error_row)
            continue # Continue to next iteration

print("Sweep complete.")

# --- Save Results to CSV ---
if all_results:
    print(f"Saving results summary for {len(all_results)} successful runs to {OUTPUT_CSV_PATH}...")
    results_df = pd.DataFrame(all_results)

    # Ensure directory exists
    Path(OUTPUT_CSV_PATH).parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    results_df.to_csv(OUTPUT_CSV_PATH, index=False)
    pkl.dump(all_results, Path(OUTPUT_PKL_PATH).open("wb"))
    print(f"Results saved successfully to {OUTPUT_CSV_PATH}")
else:
    print("No results generated during the sweep.")
