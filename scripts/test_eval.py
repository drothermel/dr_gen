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

# --- Bootstrap comparison functionality has been removed ---
print("Bootstrap comparison analysis has been removed from the codebase.")
print("Use basic statistical analysis instead.")

# Bootstrap comparison functionality has been removed

# --- Print Report ---
print("Comparison analysis complete.")
# print_results_report function has been removed with bootstrap functionality

# --- CSV export functionality has been removed ---
print("CSV export functionality removed with bootstrap analysis.")
# The following variables are no longer available due to bootstrap removal:
# best_hpm_a, best_ts_a, best_hpm_b, best_ts_b, comparison_results

# CSV export code removed with bootstrap functionality
