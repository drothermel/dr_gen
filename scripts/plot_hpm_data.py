import argparse
import json  # Added missing import
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_data(pkl_filepath):
    """Loads the data from the specified .pkl file."""
    pkl_path = Path(pkl_filepath)
    if not pkl_path.exists():
        print(f"Error: File not found: {pkl_filepath}")
        return None
    try:
        with Path(pkl_filepath).open("rb") as f:
            # Security note: Loading trusted pickle files from our own framework
            data = pickle.load(f)  # noqa: S301
        if not isinstance(data, dict):
            print(
                f"Error: Expected a dictionary in {pkl_filepath}, "
                f"but found {type(data)}."
            )
            return None
        # Check if all top-level values are dicts and contain 'hpms' and 'metrics'
        if not data or not all(
            isinstance(v, dict) and "hpms" in v and "metrics" in v
            for v in data.values()
        ):
            print(
                f"Error: Data in {pkl_filepath} does not have the expected "
                f"structure ('hpms' and 'metrics' keys in sub-dictionaries)."
            )
        else:
            return data
    except pickle.UnpicklingError:
        print(
            f"Error: Could not unpickle data from {pkl_filepath}. "
            f"File might be corrupted or not a pickle file."
        )
        return None
    except (OSError, ValueError, TypeError) as e:
        print(f"An unexpected error occurred while loading {pkl_filepath}: {e}")
        return None


def get_user_choice(prompt, options, *, allow_skip=False, allow_done=False):
    """Generic function to get a numbered choice from the user."""
    print(prompt)
    # Display options. If an option is a list or dict, use json.dumps for
    # cleaner display.
    display_options = []
    for opt in options:
        if isinstance(opt, list | dict):
            try:
                display_options.append(json.dumps(opt, sort_keys=True))
            except TypeError:
                display_options.append(str(opt))  # Fallback
        else:
            display_options.append(str(opt))

    for i, option_str in enumerate(display_options):
        print(f"  {i + 1}. {option_str}")

    extra_options = []
    if allow_skip:
        extra_options.append("'s' to skip")
    if allow_done:
        extra_options.append("'d' for done")

    if extra_options:
        print(f"  ({', '.join(extra_options)})")

    while True:
        try:
            choice_str = input("Your choice: ").strip().lower()
            if allow_skip and choice_str == "s":
                return "skip"
            if allow_done and choice_str == "d":
                return "done"

            choice_idx = int(choice_str) - 1
            if 0 <= choice_idx < len(options):
                return options[choice_idx]  # Return the original option object
            extra_cmd_text = " or special commands" if extra_options else ""
            print(
                f"Invalid choice. Please enter a number between 1 and "
                f"{len(options)}{extra_cmd_text}."
            )
        except ValueError:
            extra_cmd_text = " or special commands" if extra_options else ""
            print(f"Invalid input. Please enter a number{extra_cmd_text}.")
        except EOFError:  # Handle Ctrl+D or unexpected end of input
            print("\nSelection aborted.")
            return None


def _initialize_hpm_selection(
    all_runs_data,
) -> tuple[list[str] | None, list[str] | None]:
    """Initialize HPM selection by extracting all HPM keys and setting up candidates.

    Returns tuple of (sorted_hpm_keys, candidate_run_names) or (None, None) if failed.
    """
    if not all_runs_data:
        print("No run data loaded.")
        return None, None

    all_hpm_keys_overall = set()
    for run_details in all_runs_data.values():
        if isinstance(run_details.get("hpms"), dict):
            all_hpm_keys_overall.update(run_details["hpms"].keys())

    sorted_hpm_keys_to_consider = sorted(all_hpm_keys_overall)
    candidate_run_names = list(all_runs_data.keys())

    return sorted_hpm_keys_to_consider, candidate_run_names


def _auto_fill_single_candidate_hpm(
    all_runs_data, candidate_run_names, hpm_key, user_selected_hpms
) -> None:
    """Auto-fill HPM value if only one candidate remains and HPM not yet selected."""
    if len(candidate_run_names) == 1 and hpm_key not in user_selected_hpms:
        remaining_hpms_from_candidate = all_runs_data[candidate_run_names[0]]["hpms"]
        if hpm_key in remaining_hpms_from_candidate:
            user_selected_hpms[hpm_key] = remaining_hpms_from_candidate[hpm_key]


def _get_hpm_options_for_candidates(
    all_runs_data, candidate_run_names, hpm_key
) -> tuple[dict, list]:
    """Determine available unique values for the given HPM key among current candidates.

    Returns tuple of (options_map, actual_options_for_selection).
    """
    options_map = {}  # Maps string representation to actual value object
    for name in candidate_run_names:
        run_hpms = all_runs_data[name]["hpms"]
        if hpm_key in run_hpms:
            actual_val = run_hpms[hpm_key]
            try:
                val_str_representation = json.dumps(actual_val, sort_keys=True)
            except TypeError:
                val_str_representation = str(actual_val)

            if val_str_representation not in options_map:
                options_map[val_str_representation] = actual_val

    sorted_option_strings = sorted(options_map.keys())
    actual_options_for_selection = [options_map[s] for s in sorted_option_strings]

    return options_map, actual_options_for_selection


def _handle_single_option_hpm(
    hpm_key, actual_options_for_selection, user_selected_hpms
) -> bool:
    """Handle case where HPM has only one option among current candidates.

    Returns True if auto-selected, False otherwise.
    """
    if len(actual_options_for_selection) == 1:
        if hpm_key not in user_selected_hpms:
            user_selected_hpms[hpm_key] = actual_options_for_selection[0]
            option_val = actual_options_for_selection[0]
            option_display = (
                json.dumps(option_val)
                if isinstance(option_val, list | dict)
                else option_val
            )
            print(
                f"For {hpm_key}, only option is: {option_display} "
                f"(fixed for current candidates)."
            )
        return True
    return False


def _filter_candidates_by_hpm(
    all_runs_data, candidate_run_names, hpm_key, hpm_value
) -> list[str]:
    """Filter candidate run names by the given HPM key-value pair."""
    return [
        name
        for name in candidate_run_names
        if all_runs_data[name]["hpms"].get(hpm_key) == hpm_value
    ]


def _process_user_hpm_choice(
    hpm_key, candidate_run_names, actual_options_for_selection
) -> str | None:
    """Handle user interaction for HPM selection.

    Returns the user's choice value or special string ('done', 'skip', None).
    """
    prompt_message = (
        f"\nSelect value for '{hpm_key}' "
        f"(currently {len(candidate_run_names)} matching groups):"
    )
    return get_user_choice(
        prompt_message,
        actual_options_for_selection,
        allow_skip=True,
        allow_done=True,
    )


def _display_final_selection(all_runs_data, selected_name, user_selected_hpms) -> None:
    """Display the final selected run group and its HPMs."""
    print(f"\nSelected run group: {selected_name}")
    print("HPMs for this group (matching your criteria):")
    final_group_hpms = all_runs_data[selected_name]["hpms"]
    for k, v_actual in final_group_hpms.items():
        v_display = (
            json.dumps(v_actual) if isinstance(v_actual, list | dict) else v_actual
        )
        if k in user_selected_hpms:  # If user made a choice or it was auto-selected
            print(f"  {k}: {v_display} (Selected/Fixed)")
        else:  # HPMs not part of selection criteria but present in the chosen group
            print(f"  {k}: {v_display}")


def select_run_group_interactively(all_runs_data):
    """Allows the user to interactively select HPM values to narrow down to a run group.

    Returns the name of the selected run group, or None if aborted/failed.
    """
    sorted_hpm_keys_to_consider, candidate_run_names = _initialize_hpm_selection(
        all_runs_data
    )
    if sorted_hpm_keys_to_consider is None:
        return None

    user_selected_hpms = {}
    print("\n--- Select Hyperparameters for the Run Group ---")

    for hpm_key in sorted_hpm_keys_to_consider:
        if not candidate_run_names:
            print(
                "No runs match the current selection criteria. Aborting HPM selection."
            )
            return None

        _auto_fill_single_candidate_hpm(
            all_runs_data, candidate_run_names, hpm_key, user_selected_hpms
        )

        options_map, actual_options_for_selection = _get_hpm_options_for_candidates(
            all_runs_data, candidate_run_names, hpm_key
        )

        if not options_map:  # This HPM key is not in any of the current candidates
            continue

        if _handle_single_option_hpm(
            hpm_key, actual_options_for_selection, user_selected_hpms
        ):
            # Filter candidates by this auto-selected/fixed HPM value
            current_selection_for_key = user_selected_hpms[hpm_key]
            candidate_run_names = _filter_candidates_by_hpm(
                all_runs_data, candidate_run_names, hpm_key, current_selection_for_key
            )

            if not candidate_run_names:
                print(
                    "Error: Auto-selection or fixed HPM led to no candidates. "
                    "This is unexpected."
                )
                return None
            continue

        # Prompt user for this HPM
        user_choice_val = _process_user_hpm_choice(
            hpm_key, candidate_run_names, actual_options_for_selection
        )

        if user_choice_val is None:
            return None  # User aborted
        if user_choice_val == "done":
            print("Proceeding with current HPM selections.")
            break
        if user_choice_val == "skip":
            print(f"Skipping HPM '{hpm_key}'.")
            continue

        user_selected_hpms[hpm_key] = user_choice_val

        # Filter candidate_run_names based on the new selection
        candidate_run_names = _filter_candidates_by_hpm(
            all_runs_data, candidate_run_names, hpm_key, user_selected_hpms[hpm_key]
        )

        if not candidate_run_names:
            print(
                "No run groups match your latest selection. "
                "Please try again or broaden criteria."
            )
            return None

    final_matching_names = candidate_run_names

    if not final_matching_names:
        print("\nNo run group found matching all your HPM selections.")
        return None

    if len(final_matching_names) == 1:
        selected_name = final_matching_names[0]
        _display_final_selection(all_runs_data, selected_name, user_selected_hpms)
        return selected_name

    # Multiple groups remain - let user choose
    print("\nMultiple run groups match your selections:")
    return get_user_choice(
        "Please choose one of the final matching groups:", final_matching_names
    )


def plot_metrics(run_name, run_details, split_name, metric_name) -> None:
    """Plots the specified metric for the given run."""
    if (
        split_name not in run_details["metrics"]
        or metric_name not in run_details["metrics"][split_name]
    ):
        print(
            f"Error: Metric '{metric_name}' for split '{split_name}' "
            f"not found in run '{run_name}'."
        )
        return

    metric_data_sxe = run_details["metrics"][split_name][metric_name]  # S x E array

    expected_ndim = 2
    if (
        not isinstance(metric_data_sxe, np.ndarray)
        or metric_data_sxe.ndim != expected_ndim
    ):
        print(
            f"Error: Metric data for {metric_name} in {split_name} "
            f"is not a 2D NumPy array."
        )
        return
    if metric_data_sxe.size == 0:  # Empty array
        print(
            f"Warning: Metric data for {metric_name} in {split_name} "
            f"is empty. Cannot plot."
        )
        return

    num_seeds, num_epochs = metric_data_sxe.shape
    if num_epochs == 0:
        print(
            f"Warning: Metric data for {metric_name} in {split_name} "
            f"has 0 epochs. Cannot plot."
        )
        return

    epochs_axis = np.arange(num_epochs)

    plt.figure(figsize=(12, 7))
    for i in range(num_seeds):
        plt.plot(epochs_axis, metric_data_sxe[i, :], label=f"Seed {i + 1}")

    plt.xlabel("Epoch")
    plt.ylabel(metric_name.capitalize())

    max_title_len = 70
    display_run_name = (
        run_name
        if len(run_name) <= max_title_len
        else run_name[: max_title_len - 3] + "..."
    )
    plt.title(
        f"{metric_name.capitalize()} for {split_name.capitalize()} Split\n"
        f"Group: {display_run_name}"
    )

    plt.legend()
    plt.grid()
    plt.tight_layout()
    print("Displaying plot... Close the plot window to continue.")
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactively plot metrics from aggregated log data."
    )
    parser.add_argument(
        "--pkl_file",
        default=".",
        type=str,
        help="Path to the .pkl file containing aggregated run data.",
    )
    args = parser.parse_args()
    args.pkl_file = (
        "/Users/daniellerothermel/drotherm/data/dr_gen/run_data_v1/lr_wd_init_v0_t2.pkl"
    )

    all_runs_data = load_data(args.pkl_file)
    if not all_runs_data:
        return

    while True:
        selected_run_name = select_run_group_interactively(all_runs_data)
        if not selected_run_name:
            print("No run group selected or selection aborted.")
            if (
                input("\nTry selecting another run group? (y/n): ").strip().lower()
                != "y"
            ):
                break
            continue

        selected_run_details = all_runs_data[selected_run_name]

        available_splits = sorted(selected_run_details["metrics"].keys())
        if not available_splits:
            print(f"No metric splits found for run group '{selected_run_name}'.")
            if (
                input("\nTry selecting another run group? (y/n): ").strip().lower()
                != "y"
            ):
                break
            continue

        print(f"\n--- Select Data Split for '{selected_run_name}' ---")
        selected_split = get_user_choice("Available splits:", available_splits)
        if not selected_split:
            print("\nSplit selection aborted.")
            if (
                input("\nTry selecting another run group? (y/n): ").strip().lower()
                != "y"
            ):
                break
            continue

        available_metrics = sorted(
            selected_run_details["metrics"].get(selected_split, {}).keys()
        )
        if not available_metrics:
            print(
                f"No metrics found for split '{selected_split}' "
                f"in run group '{selected_run_name}'."
            )
            if (
                input("\nTry selecting another run group? (y/n): ").strip().lower()
                != "y"
            ):
                break
            continue

        print(f"\n--- Select Metric for Split '{selected_split}' ---")
        selected_metric = get_user_choice("Available metrics:", available_metrics)
        if not selected_metric:
            print("\nMetric selection aborted.")
            if (
                input("\nTry selecting another run group? (y/n): ").strip().lower()
                != "y"
            ):
                break
            continue

        print(
            f"\nPlotting {selected_metric} for {selected_split} split "
            f"of run group '{selected_run_name}'..."
        )
        plot_metrics(
            selected_run_name, selected_run_details, selected_split, selected_metric
        )

        if (
            input("\nPlot another metric (for the same or different group)? (y/n): ")
            .strip()
            .lower()
            != "y"
        ):
            break

    print("\nExiting interactive plotter.")


if __name__ == "__main__":
    main()
