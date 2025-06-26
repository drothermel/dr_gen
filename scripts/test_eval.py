from pathlib import Path

import numpy as np  # Needed for potential NaN comparisons if runs fail
import pandas as pd  # For CSV export

import dr_gen.analyze.result_plotting as rplt
from dr_gen.analyze.run_group import RunGroup

# Define the path to the run data
RUN_DATA_PATH = (
    "/Users/daniellerothermel/drotherm/data/dr_gen/cifar10/cluster_runs/lr_wd_init_v0"
)
# Define the output path for the results CSV
OUTPUT_CSV_PATH = "/Users/daniellerothermel/Desktop/comparison_summary_v0.csv"

# Define HPMs to select runs for comparison
# Use lists for hyperparameters where multiple values should be considered
hpm_specs_hpm_select = {
    "optim.lr": [0.04, 0.06, 0.1, 0.16, 0.25],
    "optim.weight_decay": [1e-05, 4e-05, 6.3e-05, 0.0001, 0.00016, 0.00025],
    "epochs": 270,
}

# --- Load Data ---
print(f"Loading run data from: {RUN_DATA_PATH}")
rg = RunGroup()
rg.load_runs_from_base_dir(RUN_DATA_PATH)
print("Loaded runs.")

# --- Select Run Groups ---
print("Selecting runs for pretrained vs. random initialization comparison...")
(hpms_val_pre, hpms_eval_pre), (hpms_val_rand, hpms_eval_rand) = (
    rplt.get_compare_runs_pretrain_vs_random(
        rg,
        hpm_select_dict=hpm_specs_hpm_select,
    )
)

print(f"Found {len(hpms_val_pre)} matching HPM sets for Pretrained.")
print(f"Found {len(hpms_val_rand)} matching HPM sets for Random Init.")

if not hpms_val_pre or not hpms_val_rand:
    print("Error: Could not find matching hyperparameter sets for comparison.")
    exit()

# --- Run Comparison ---
print("Running comparison analysis...")
# Note: We assume get_compare_runs_pretrain_vs_random ensures matching HPMs
# between val and eval sets for each group (pre/rand).

# Combine val/eval data correctly for run_comparison_eval
group_a_data = (hpms_val_pre, hpms_eval_pre)
group_b_data = (hpms_val_rand, hpms_eval_rand)

(
    best_hpm_a,
    best_ts_a,
    best_hpm_b,
    best_ts_b,
    comparison_results,
) = rplt.run_comparison_eval(
    group_a_data=group_a_data,
    group_b_data=group_b_data,
    group_a_name="Pretrained",
    group_b_name="Random Init",
    val_num_bootstraps=500,  # Reduced for faster testing
    eval_num_bootstraps=500,  # Reduced for faster testing
    num_permutations=500,  # Reduced for faster testing
)

# --- Print Report ---
print("Comparison analysis complete.")
rplt.print_results_report(
    best_hpm_A=best_hpm_a,
    best_ts_A=best_ts_a,
    best_hpm_B=best_hpm_b,
    best_ts_B=best_ts_b,
    comparison_results=comparison_results,
)

# --- Save Results to CSV ---
print(f"\nSaving results summary to {OUTPUT_CSV_PATH}...")
# Prepare data row for CSV
csv_data = {
    "group_A_name": "Pretrained",
    "group_B_name": "Random Init",
    "best_hpm_A": best_hpm_a,
    "best_val_timestep_A": best_ts_a,
    "best_hpm_B": best_hpm_b,
    "best_val_timestep_B": best_ts_b,
    "bootstraps_A": comparison_results["bootstraps_A"],
    "bootstraps_B": comparison_results["bootstraps_B"],
    "original_A": comparison_results["original_A"],
    "original_B": comparison_results["original_B"],
}

# Flatten summary, difference, and test statistics, excluding distributions
stats_to_flatten = {
    "summary_A_": comparison_results.get("summary_A", {}),
    "summary_B_": comparison_results.get("summary_B", {}),
    "diff_": comparison_results.get("difference_stats", {}),
    "ks_ci_": comparison_results.get("ks_ci_test", {}),
    "ks_perm_": comparison_results.get("ks_permutation_test", {}),
}

for prefix, stats_dict in stats_to_flatten.items():
    for key, value in stats_dict.items():
        # Convert NumPy arrays to string representation for CSV
        if isinstance(value, np.ndarray):
            # Convert array to list, then to string
            csv_data[f"{prefix}{key}"] = str(value.tolist())
        elif isinstance(value, tuple):
            # Handle tuples (like CIs) by splitting into lower/upper
            csv_data[f"{prefix}{key}_lower"] = value[0] if len(value) > 0 else np.nan
            csv_data[f"{prefix}{key}_upper"] = value[1] if len(value) > 1 else np.nan
        else:
            # Store other primitive types directly
            csv_data[f"{prefix}{key}"] = value

for key in ["bootstraps", "original"]:
    b_a = comparison_results[f"{key}_A"]
    if isinstance(b_a, np.ndarray):
        csv_data[f"{key}_A"] = str(b_a.tolist())
    b_b = comparison_results[f"{key}_B"]
    if isinstance(b_b, np.ndarray):
        csv_data[f"{key}_B"] = str(b_b.tolist())

# Create DataFrame
results_df = pd.DataFrame([csv_data])

# Ensure directory exists
Path(OUTPUT_CSV_PATH).parent.mkdir(parents=True, exist_ok=True)

# Save to CSV
results_df.to_csv(OUTPUT_CSV_PATH, index=False)
print(f"Results saved successfully to {OUTPUT_CSV_PATH}")
