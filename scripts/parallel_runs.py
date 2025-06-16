# launcher_hydra_external_parallel.py
import argparse
import itertools  # For generating parameter combinations
import os
import subprocess
import time
from pathlib import Path

# --- Default Configuration for the Launcher ---
DEFAULT_MAX_PARALLEL_JOBS = 7
DEFAULT_SEEDS_TO_RUN = [42, 123] # Shortened for brevity in help text
DEFAULT_START_SEED = 1

# --- Hydra Parameter Defaults (can be overridden by CLI args) ---
DEFAULT_VAL_BS = "64" # String, to be parsed
DEFAULT_PROJ_DIR_NAME = "cifar10_test"
DEFAULT_EPOCHS = "20" # String
DEFAULT_BATCH_SIZE = "500" # String
DEFAULT_LR = "0.01" # String
DEFAULT_WEIGHT_DECAY = "2.5e-4" # String
DEFAULT_WEIGHT_TYPE = "scratch"

# --- Static Configuration ---
PYTHON_EXECUTABLE = "python"
# IMPORTANT: Update this path!
TRAINING_SCRIPT_PATH = "/scratch/ddr8143/repos/dr_gen/scripts/train.py"

AVAILABLE_GPUS = None # Example: [0, 1] for specific GPU assignment
_current_gpu_idx = 0

LAUNCHER_LOG_DIR = "launcher_run_logs_combinations"

def print_flush(in_val: object) -> None:
    print(in_val, flush=True)

def parse_value_list(value_str: object, target_type: type = str) -> list:
    """Parses a command-line string. If it contains commas, splits it into a list.

    Converts elements to the target_type.
    If already a list (e.g., for seeds), returns it after type conversion.
    """
    if isinstance(value_str, list):
        return [target_type(item) for item in value_str]

    # Ensure value_str is a string before calling split
    if not isinstance(value_str, str):
        value_str = str(value_str)

    items = [s.strip() for s in value_str.split(",")]

    # Handle boolean case explicitly if needed, e.g., "true" -> True
    if target_type is bool:
        return [s.lower() == "true" for s in items]
    return [target_type(item) for item in items]

def setup_launcher_logging() -> None:
    """Creates or cleans the launcher log directory."""
    log_dir = Path(LAUNCHER_LOG_DIR)
    if log_dir.exists():
        print_flush(f"Launcher log directory '{LAUNCHER_LOG_DIR}' already exists.")
    log_dir.mkdir(parents=True, exist_ok=True)
    print_flush(f"Launcher stdout/stderr logs will be stored in: {log_dir.resolve()}")

def get_next_gpu_id() -> int | None:
    """Cycles through available GPUs if specified."""
    global _current_gpu_idx  # noqa: PLW0603
    if AVAILABLE_GPUS and len(AVAILABLE_GPUS) > 0:
        gpu_id = AVAILABLE_GPUS[_current_gpu_idx % len(AVAILABLE_GPUS)]
        _current_gpu_idx += 1
        return gpu_id
    return None

def create_unique_job_name(param_dict: dict) -> str:
    """Creates a unique and descriptive name from parameter dictionary.

    Used for logging and directories.
    """
    name_parts = []
    # Sort items for consistent naming
    for key, value in sorted(param_dict.items()):
        # Sanitize key for directory/file names: replace '.' with '_' or remove
        # entirely if problematic
        sanitized_key = (
            key.replace(".", "_").replace("paths_", "").replace("optim_", "")
        )
        name_parts.append(f"{sanitized_key}_{value}")
    return "-".join(name_parts)

def start_training_run(
    param_combination_dict: dict, job_name: str
) -> tuple[object, str] | tuple[None, str]:
    """Constructs the command to run the Hydra script with parameters.

    Starts it as a subprocess.
    Returns a tuple (subprocess.Popen object, job_name).
    """
    command = [PYTHON_EXECUTABLE, TRAINING_SCRIPT_PATH]

    # Add Hydra overrides from the parameter combination dictionary
    for hydra_path, value in param_combination_dict.items():
        # Special handling for val_bs to apply to two Hydra paths
        if hydra_path == "val_bs_shared": # Internal key used for val_batch_size arg
            command.append(f"val.batch_size={value}")
            command.append(f"eval.batch_size={value}")
        else:
            command.append(f"{hydra_path}={value}")

    # --- Environment Setup (e.g., for specific GPU) ---
    current_env = os.environ.copy()
    current_env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    assigned_gpu_id = get_next_gpu_id()
    launch_message_gpu = (
        f"on GPU {assigned_gpu_id}"
        if assigned_gpu_id is not None
        else "(default GPU)"
    )
    print_flush(f"Attempting to launch job '{job_name}' {launch_message_gpu}...")
    if assigned_gpu_id is not None:
        current_env["CUDA_VISIBLE_DEVICES"] = str(assigned_gpu_id)

    print_flush(f"  Command: {' '.join(command)}")

    # --- Define unique stdout and stderr log files for this subprocess ---
    stdout_log_path = Path(LAUNCHER_LOG_DIR) / f"{job_name}_stdout.log"
    stderr_log_path = Path(LAUNCHER_LOG_DIR) / f"{job_name}_stderr.log"

    # --- Start the Subprocess ---
    try:
        with (
            stdout_log_path.open("wb") as stdout_file,
            stderr_log_path.open("wb") as stderr_file,
        ):
            # Command is constructed from controlled constants and validated parameters
            process = subprocess.Popen(  # noqa: S603
                command, env=current_env, stdout=stdout_file, stderr=stderr_file
            )
    except FileNotFoundError:
        print_flush(
            f"[ERROR] Could not find Python executable '{PYTHON_EXECUTABLE}' or "
            f"script '{TRAINING_SCRIPT_PATH}'. Please check paths."
        )
        error_log_path = Path(LAUNCHER_LOG_DIR) / "launcher_critical_errors.log"
        with error_log_path.open("a") as f_err:
            f_err.write(
                f"FileNotFoundError for job '{job_name}': "
                f"Python='{PYTHON_EXECUTABLE}', Script='{TRAINING_SCRIPT_PATH}'\n"
            )
        return None, job_name
    except (OSError, subprocess.SubprocessError) as e:
        print_flush(f"[ERROR] Failed to launch job '{job_name}' due to: {e}")
        error_log_path = Path(LAUNCHER_LOG_DIR) / "launcher_critical_errors.log"
        with error_log_path.open("a") as f_err:
            f_err.write(
                f"Launch exception for job '{job_name}': {e}\n"
                f"Command: {' '.join(command)}\n"
            )
        return None, job_name
    else:
        print_flush(
            f"  Successfully launched PID: {process.pid} for job '{job_name}'. "
            f"Logs: {stdout_log_path}"
        )
        return process, job_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch multiple Hydra training runs by sweeping through "
        "combinations of parameters."
    )

    # Core launcher args
    parser.add_argument(
        "-p", "--max_parallel_jobs",
        type=int,
        default=DEFAULT_MAX_PARALLEL_JOBS,
        help=f"Max parallel training jobs. Default: {DEFAULT_MAX_PARALLEL_JOBS}"
    )
    parser.add_argument("-g", "--avail_gpus", type=str, default="")
    parser.add_argument(
        "-s", "--start_seed",
        type=int,
        default=DEFAULT_START_SEED,
        help=f"Start seed for sequence (if --max_seed is used). Default: {DEFAULT_START_SEED}"
    )
    parser.add_argument(
        "-e", "--max_seed",
        type=int,
        default=None,
        help="Max seed for sequence. Overrides default seeds list."
    )
    parser.add_argument(
        "--seeds_list",
        type=str,
        default=",".join(map(str, DEFAULT_SEEDS_TO_RUN)),
        help=f"Comma-separated list of seeds if not using start/max_seed. Default: {','.join(map(str, DEFAULT_SEEDS_TO_RUN))}"
    )

    # Hydra parameter sweep args
    parser.add_argument(
        "--val_bs",
        type=str,
        default=DEFAULT_VAL_BS,
        help=f"Validation/Evaluation batch size(s), comma-separated. Default: {DEFAULT_VAL_BS}"
    )
    parser.add_argument(
        "--proj_name",
        type=str,
        default=DEFAULT_PROJ_DIR_NAME,
        help=f"Project directory name(s) for 'paths.proj_dir_name', comma-separated. Default: {DEFAULT_PROJ_DIR_NAME}"
    )
    parser.add_argument(
        "--epochs",
        type=str,
        default=DEFAULT_EPOCHS,
        help=f"Number of epoch(s), comma-separated. Default: {DEFAULT_EPOCHS}"
    )
    parser.add_argument(
        "--bs",
        type=str,
        default=DEFAULT_BATCH_SIZE,
        help=f"Training batch size(s), comma-separated. Default: {DEFAULT_BATCH_SIZE}"
    )
    parser.add_argument(
        "--lr",
        type=str,
        default=DEFAULT_LR,
        help=f"Learning rate(s) for 'optim.lr', comma-separated. Default: {DEFAULT_LR}"
    )
    parser.add_argument(
        "--wd",
        type=str,
        default=DEFAULT_WEIGHT_DECAY,
        help=f"Weight decay(s) for 'optim.weight_decay', comma-separated. Default: {DEFAULT_WEIGHT_DECAY}"
    )
    parser.add_argument(
        "--wtype",
        type=str,
        default=DEFAULT_WEIGHT_TYPE,
        help=f"Weight type(s) for 'weight_type', comma-separated. Default: {DEFAULT_WEIGHT_TYPE}"
    )
    parser.add_argument("--wn", type=str, default="DEFAULT")
    parser.add_argument("--ws", type=str, default="torchvision")
    parser.add_argument("--xft", type=str, default="pycil")
    parser.add_argument("--use_percent", type=str, default="1.0")

    args = parser.parse_args()
    AVAILABLE_GPUS = (
        None if args.avail_gpus == ""
        else parse_value_list(args.avail_gpus, target_type=int)
    )
    MAX_PARALLEL_JOBS = args.max_parallel_jobs
    WEIGHT_NAME = args.wn

    # Determine seeds to run
    if args.max_seed is not None:
        if args.max_seed < args.start_seed:
            print_flush(f"[ERROR] --max_seed ({args.max_seed}) must be >= --start_seed ({args.start_seed}). Exiting.")
            exit(1)
        actual_seeds_to_run = list(range(args.start_seed, args.max_seed + 1))
        print_flush(f"Using generated seeds from {args.start_seed} to {args.max_seed}.")
    else:
        actual_seeds_to_run = parse_value_list(args.seeds_list, int)
        print_flush(f"Using provided seeds list: {actual_seeds_to_run}")

    if not Path(TRAINING_SCRIPT_PATH).is_file():
        print_flush(f"[ERROR] Training script not found at '{TRAINING_SCRIPT_PATH}'. Please check the path. Exiting.")
        exit(1)

    # Prepare lists of parameter values for itertools.product
    param_map = {
        # Argparse dest name : (Hydra path / internal key, type converter)
        "val_bs_shared": ("val_bs_shared", int), # Special key for val_bs
        "paths.proj_dir_name": (args.proj_name, str),
        "epochs": (args.epochs, int),
        "train.batch_size": (args.bs, int),
        "optim.lr": (args.lr, float),
        "optim.weight_decay": (args.wd, float),
        "weight_type": (args.wtype, str),
        "model.weights": (args.wn, str),
        "model.source": (args.ws, str),
        "data.transform_type": (args.xft, str),
        "data.train.use_percent": (args.use_percent, float),
    }

    # Always include 'seed' as the first parameter to sweep
    param_names_for_product = ["seed"]
    value_lists_for_product = [actual_seeds_to_run]

    # Add other sweepable parameters
    # Order matters for zip later, so build param_names_for_product and value_lists_for_product in sync
    hydra_param_cli_map = {
        # Hydra Path : (argparse_value_string, target_type)
        "val_bs_shared": (args.val_bs, int), # Use internal key
        "paths.proj_dir_name": (args.proj_name, str),
        "epochs": (args.epochs, int),
        "train.batch_size": (args.bs, int),
        "optim.lr": (args.lr, float),
        "optim.weight_decay": (args.wd, float),
        "weight_type": (args.wtype, str),
        "model.weights": (args.wn, str),
        "model.source": (args.ws, str),
        "data.transform_type": (args.xft, str),
        "data.train.use_percent": (args.use_percent, float),
    }

    for hydra_path_key, (cli_value_str, target_type) in hydra_param_cli_map.items():
        param_names_for_product.append(hydra_path_key)
        value_lists_for_product.append(parse_value_list(cli_value_str, target_type))

    # Generate all combinations
    all_combinations_values = list(itertools.product(*value_lists_for_product))

    all_param_dicts_to_run = [
        dict(zip(param_names_for_product, combo_values, strict=False))
        for combo_values in all_combinations_values
    ]

    total_jobs = len(all_param_dicts_to_run)
    print_flush(f"\nTotal number of unique parameter combinations to run: {total_jobs}")
    if total_jobs == 0:
        print_flush("No jobs to run. Exiting.")
        exit(0)

    setup_launcher_logging()
    active_processes = []  # List of (process_object, job_name)
    launched_job_names = set()

    for i, current_run_param_dict in enumerate(all_param_dicts_to_run):
        # Create a unique name for this specific job combination
        current_job_name = create_unique_job_name(current_run_param_dict)

        if current_job_name in launched_job_names:
            print_flush(f"Job '{current_job_name}' seems to be a duplicate (already launched). Skipping.")
            continue

        while len(active_processes) >= MAX_PARALLEL_JOBS:
            print_flush(f"Max parallel jobs ({MAX_PARALLEL_JOBS}) reached. Waiting ({len(active_processes)} active)...")
            for proc_info in active_processes[:]:
                process, p_job_name = proc_info
                if process.poll() is not None:
                    rc = process.returncode
                    status = "successfully" if rc == 0 else f"with error code {rc}"
                    print_flush(f"  Job '{p_job_name}' (PID: {process.pid}) completed {status}.")
                    if rc != 0:
                        print_flush(f"    Check logs: {LAUNCHER_LOG_DIR}/{p_job_name}_stderr.log")
                    active_processes.remove(proc_info)
            if len(active_processes) >= MAX_PARALLEL_JOBS:
                time.sleep(15)

        print_flush(f"\n[{i+1}/{total_jobs}] Launching job with params: {current_run_param_dict}")
        process_obj, launched_job_name = start_training_run(current_run_param_dict, current_job_name)

        if process_obj:
            active_processes.append((process_obj, launched_job_name))
            launched_job_names.add(launched_job_name)
            print_flush(f"  Active jobs: {len(active_processes)}/{MAX_PARALLEL_JOBS}")
        else:
            print_flush(f"[WARNING] Failed to launch job '{current_job_name}'. It will be skipped.")

        time.sleep(2) # Brief pause between launches

    print_flush("\nAll jobs launched. Waiting for remaining to complete...")
    while active_processes:
        print_flush(f"Waiting for {len(active_processes)} remaining job(s)...")
        for proc_info in active_processes[:]:
            process, p_job_name = proc_info
            if process.poll() is not None:
                rc = process.returncode
                status = "successfully" if rc == 0 else f"with error code {rc}"
                print_flush(f"  Job '{p_job_name}' (PID: {process.pid}) completed {status} (final check).")
                if rc != 0: print_flush(f"    Check logs: {LAUNCHER_LOG_DIR}/{p_job_name}_stderr.log")
                active_processes.remove(proc_info)
        if active_processes: time.sleep(20)

    print_flush("\n--- All training runs initiated by the launcher have completed. ---")
    print_flush(f"Launcher's per-job stdout/stderr logs are in: {os.path.abspath(LAUNCHER_LOG_DIR)}")
    print_flush("--- Launcher script finished. ---")

