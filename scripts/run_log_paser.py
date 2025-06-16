import argparse
import json
import os
import pickle
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np

# A unique object to represent missing keys in configs for comparison during grouping
_NotPresent = object()

# --- Configuration Key Blacklist ---
# Keys containing any of these substrings will be ignored for varying HPM
# checks and in the final HPM output.
CONFIG_KEY_BLACKLIST = [
    "paths",  # Will match "paths.root", "data.paths.something", etc.
    "write_checkpoint",
    "weight_type",
    # Add other key substrings to blacklist here
]


def is_key_blacklisted(key, blacklist):
    """Checks if a key contains any of the substrings in the blacklist."""
    if key is None:
        return False
    for pattern in blacklist:
        if pattern in key:
            return True
    return False


def flatten_dict(d, parent_key="", sep="."):
    """Flattens a nested dictionary.

    E.g., {"a": 1, "b": {"c": 2}} becomes {"a": 1, "b.c": 2}.
    Preserves order if input is OrderedDict.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(
            v, dict
        ):  # Changed from collections.abc.MutableMapping to just dict for simplicity
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    # If original dict was OrderedDict, preserve order, else normal dict
    return OrderedDict(items) if isinstance(d, OrderedDict) else dict(items)


def parse_log_file(filepath):
    """Parses a single .jsonl log file.

    The first line is expected to be the JSON config.
    It handles potentially nested configs like
    {"type": "dict_config", "value": {...actual_config...}}.
    The parsed config is then flattened.
    Subsequent lines contain metric data.

    Args:
        filepath (str): Path to the .jsonl file.

    Returns:
        tuple: (config_dict, metrics_data_dict)
               config_dict is the actual configuration dictionary, FLATTENED.
               metrics_data_dict is a dict like:
               {'train': {'loss': [val1, val2, ...], 'acc1': [val1, val2, ...]}, ...}
               Returns (None, empty_metrics_dict) if parsing fails critically.
    """
    raw_config = None
    metrics_data = defaultdict(lambda: defaultdict(list))

    try:
        with Path(filepath).open() as f:
            # First line is config
            try:
                first_line = f.readline()
                if not first_line.strip():
                    print(f"Warning: File {filepath} is empty or first line is blank.")
                    return None, metrics_data

                parsed_json_config = json.loads(first_line)

                if (
                    isinstance(parsed_json_config, dict)
                    and parsed_json_config.get("type") == "dict_config"
                    and "value" in parsed_json_config
                    and isinstance(parsed_json_config["value"], dict)
                ):
                    raw_config = parsed_json_config["value"]
                else:
                    raw_config = parsed_json_config

            except json.JSONDecodeError as e:
                print(
                    f"Warning: Could not parse config from first line of "
                    f"{filepath}: {e}. Skipping this file."
                )
                return None, metrics_data
            except (AttributeError, KeyError, TypeError) as e:
                print(
                    f"Warning: Error processing config from {filepath}: "
                    f"{e}. Skipping this file."
                )
                return None, metrics_data

            if not isinstance(
                raw_config, dict
            ):  # Ensure raw_config is a dict before flattening
                print(
                    f"Warning: Config from {filepath} is not a dictionary "
                    f"after initial parsing. Skipping file."
                )
                return None, metrics_data

            # Flatten the configuration
            try:
                config = flatten_dict(raw_config)
            except (AttributeError, TypeError, ValueError) as e:
                print(
                    f"Warning: Could not flatten config for {filepath}: "
                    f"{e}. Skipping this file."
                )
                return None, metrics_data

            # Process rest of the lines for metrics
            for line_num, line_content in enumerate(f, start=2):
                if not line_content.strip():
                    continue
                try:
                    log_entry = json.loads(line_content)
                    if (
                        log_entry.get("title")
                        and log_entry.get("data_name")
                        and isinstance(log_entry.get("agg_stats"), dict)
                    ):
                        split_name = log_entry["data_name"]
                        agg_stats = log_entry["agg_stats"]

                        if "loss" in agg_stats:
                            metrics_data[split_name]["loss"].append(agg_stats["loss"])
                        if "acc1" in agg_stats:
                            metrics_data[split_name]["acc1"].append(agg_stats["acc1"])

                except json.JSONDecodeError:
                    pass
                except (KeyError, TypeError, ValueError):
                    pass

    except OSError as e:
        print(f"Error: Could not read file {filepath}: {e}")
        return None, metrics_data

    return config, metrics_data


def group_runs_and_identify_varying_keys(all_runs_data, key_blacklist):
    """Identifies varying FLATTENED config parameters (excluding 'seed' and
    blacklisted keys) and groups runs.

    Args:
        all_runs_data (list): A list of dictionaries, where each dict has
                              {'config': FLATTENED_actual_config_dict,
                               'metrics': metrics_dict,
                               'filepath': str}.
        key_blacklist (list): List of substrings. Keys containing these will be ignored.

    Returns:
        tuple: (grouped_runs_dict, varying_keys_list, group_to_hpms_map_dict)
               grouped_runs_dict maps run_group_name to a list of run_data dicts.
               varying_keys_list contains the names of FLATTENED config keys
               that vary (and are not blacklisted).
               group_to_hpms_map_dict maps run_group_name to its
               representative FLATTENED config (excluding blacklisted).
    """
    if not all_runs_data:
        return defaultdict(list), [], {}

    valid_runs_data = [
        run
        for run in all_runs_data
        if run["config"] is not None and isinstance(run["config"], dict)
    ]
    if not valid_runs_data:
        return defaultdict(list), [], {}

    all_flattened_configs = [run["config"] for run in valid_runs_data]

    super_keyset = set()
    for cfg in all_flattened_configs:
        super_keyset.update(cfg.keys())

    varying_config_keys = []
    # Sort super_keyset for consistent order of varying_config_keys and group names
    for key in sorted(super_keyset):
        # Exclude seed, and any blacklisted keys
        if (
            key == "seed"
            or key.endswith(".seed")
            or is_key_blacklisted(key, key_blacklist)
        ):
            continue

        values_for_key = set()
        for config_item in all_flattened_configs:
            value = config_item.get(key, _NotPresent)
            if value is _NotPresent:
                values_for_key.add(str(_NotPresent))
            else:
                # Use JSON dumps for complex types to ensure hashability and
                # correct comparison
                try:
                    values_for_key.add(json.dumps(value, sort_keys=True))
                except TypeError:  # Fallback for non-JSON serializable simple types
                    values_for_key.add(str(value))

        if len(values_for_key) > 1:
            varying_config_keys.append(key)

    grouped_runs = defaultdict(list)
    group_to_hpms_map = {}

    for run_data in valid_runs_data:
        config = run_data["config"]  # This is the flattened config
        group_name_parts = []
        # Use the already sorted varying_config_keys for consistent group naming
        # These keys are already filtered against the blacklist.
        for key in varying_config_keys:
            value = config.get(key)
            if value is None and key not in config:
                value_str = "MISSING"
            else:
                try:
                    value_str = json.dumps(value, sort_keys=True, separators=(",", ":"))
                except TypeError:
                    value_str = str(value)

            # Sanitize key and value_str for file/group naming
            sane_key = (
                key.replace("/", "_")
                .replace("\\", "_")
                .replace('"', "")
                .replace("'", "")
            )
            sane_value_str = (
                value_str.replace("/", "_")
                .replace("\\", "_")
                .replace('"', "")
                .replace("'", "")
            )
            group_name_parts.append(f"{sane_key}={sane_value_str}")

        run_group_name = (
            "_".join(group_name_parts) if group_name_parts else "default_group"
        )
        grouped_runs[run_group_name].append(run_data)

        if run_group_name not in group_to_hpms_map:
            # Store the (flattened) config of the first run encountered for this group
            # Exclude 'seed' and blacklisted keys from the stored HPMs for the group
            hpms_for_group = {
                k: v
                for k, v in config.items()
                if k != "seed"
                and not k.endswith(".seed")
                and not is_key_blacklisted(k, key_blacklist)
            }
            group_to_hpms_map[run_group_name] = hpms_for_group

    return grouped_runs, varying_config_keys, group_to_hpms_map


def aggregate_metrics_for_groups(grouped_runs):
    """Aggregates metrics for each run group into S x E NumPy arrays.
    This function's core logic for metrics aggregation remains the same.

    Args:
        grouped_runs (dict): Maps run_group_name to a list of run_data dicts.
                             Each run_data['config'] is a FLATTENED config.

    Returns:
        dict: group_metrics_results of the structure:
              {run_group_name: {split: {metric: np.array(shape=(S, E))}}}
    """
    group_metrics_results = {}
    target_splits = ["train", "val", "eval"]
    target_metrics = ["loss", "acc1"]

    for run_group_name, runs_in_group in grouped_runs.items():
        current_group_split_metric_results = defaultdict(dict)

        for split in target_splits:
            for metric in target_metrics:
                metrics_by_seed_for_current_setting = []
                num_epochs_per_seed = []

                # Sort runs by seed to ensure consistent ordering in the S dimension
                # Access 'seed' from the flattened config.
                sorted_runs_in_group = sorted(
                    runs_in_group, key=lambda r: r["config"].get("seed", float("inf"))
                )

                for run_data in sorted_runs_in_group:
                    seed_metric_values = run_data["metrics"][split][metric]
                    if seed_metric_values:
                        metrics_by_seed_for_current_setting.append(seed_metric_values)
                        num_epochs_per_seed.append(len(seed_metric_values))

                if not metrics_by_seed_for_current_setting:
                    continue

                min_epochs = min(num_epochs_per_seed) if num_epochs_per_seed else 0
                if min_epochs == 0:
                    continue

                truncated_metrics_by_seed = [
                    series[:min_epochs]
                    for series in metrics_by_seed_for_current_setting
                    if len(series) >= min_epochs
                ]

                if not truncated_metrics_by_seed:
                    continue

                try:
                    aggregated_array = np.array(truncated_metrics_by_seed)
                    if (
                        aggregated_array.ndim == 2
                        and aggregated_array.shape[1] == min_epochs
                    ):
                        current_group_split_metric_results[split][metric] = (
                            aggregated_array
                        )
                    else:
                        print(
                            f"Warning: Could not form valid S x E array for "
                            f"{run_group_name}/{split}/{metric}. "
                            f"Expected E={min_epochs}, got shape "
                            f"{aggregated_array.shape}. Skipping."
                        )
                except (ValueError, TypeError, AttributeError) as e:
                    print(
                        f"Error: Failed to convert to NumPy array for "
                        f"{run_group_name}/{split}/{metric}: {e}. Skipping."
                    )

        if current_group_split_metric_results:
            # Clean up empty metric dicts within splits for this group
            for split_key in list(current_group_split_metric_results.keys()):
                if not current_group_split_metric_results[split_key]:
                    del current_group_split_metric_results[split_key]
            if (
                current_group_split_metric_results
            ):  # if group still has data after cleaning splits
                group_metrics_results[run_group_name] = (
                    current_group_split_metric_results
                )

    return group_metrics_results


# Main execution block
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process .jsonl log files from a directory to aggregate and group metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=".",
        help="Directory containing .jsonl log files.",
    )
    parser.add_argument(
        "--output_pkl",
        type=str,
        default=".",
        help="Path to save the output .pkl file (e.g., results.pkl).",
    )

    args = parser.parse_args()
    args.log_dir = "/Users/daniellerothermel/drotherm/data/dr_gen/cifar10/cluster_runs/lr_wd_init_v0_t2/"
    args.output_pkl = (
        "/Users/daniellerothermel/drotherm/data/dr_gen/run_data_v1/lr_wd_init_v0_t2.pkl"
    )

    log_dir_path = Path(args.log_dir)
    if not log_dir_path.is_dir():
        print(f"Error: Directory not found: {args.log_dir}")
        exit(1)

    all_runs_data_with_flattened_configs = []
    print(f"Scanning directory: {args.log_dir}")
    for filename in os.listdir(args.log_dir):
        if filename.endswith(".jsonl"):
            filepath = log_dir_path / filename
            # parse_log_file now returns flattened_config
            flattened_config, metrics = parse_log_file(filepath)
            if flattened_config is not None:
                all_runs_data_with_flattened_configs.append(
                    {
                        "config": flattened_config,
                        "metrics": metrics,
                        "filepath": filepath,
                    }
                )

    if not all_runs_data_with_flattened_configs:
        print(
            "No valid log files with parsable and flattenable configurations found. Exiting."
        )
        exit(0)
    print(
        f"Successfully parsed and flattened configs for {len(all_runs_data_with_flattened_configs)} files."
    )

    print(
        "\nGrouping runs and identifying varying parameters (using flattened keys)..."
    )
    # Pass CONFIG_KEY_BLACKLIST to the grouping function
    grouped_runs, varying_keys, group_to_hpms = group_runs_and_identify_varying_keys(
        all_runs_data_with_flattened_configs, CONFIG_KEY_BLACKLIST
    )

    if not grouped_runs:
        print("No run groups could be formed.")
        if not any(all_runs_data_with_flattened_configs):
            print("No run data was loaded initially.")
        exit(0)

    print(
        f"Identified {len(varying_keys)} varying config parameters (excluding 'seed', blacklisted keys, dot-separated): {varying_keys if varying_keys else 'None'}"
    )
    print(f"Formed {len(grouped_runs)} run groups: {list(grouped_runs.keys())}")

    print("\nAggregating metrics for each group...")
    # This function returns {group_name: {split: {metric: np_array}}}
    aggregated_metrics_per_group = aggregate_metrics_for_groups(grouped_runs)

    # Construct the final output dictionary with "metrics" and "hpms" keys
    final_output_to_pickle = {}
    if not aggregated_metrics_per_group and group_to_hpms:
        print(
            "No metrics could be aggregated, but HPM groups were formed. Output will contain HPMs only for groups that were identified."
        )
        # Still create entries if only HPMs are available for a group
        for group_name, hpms in group_to_hpms.items():
            final_output_to_pickle[group_name] = {
                "metrics": {},  # No metrics data
                "hpms": hpms,
            }

    elif not aggregated_metrics_per_group and not group_to_hpms:
        print(
            "No metrics could be aggregated and no HPM groups formed. Output .pkl will be empty."
        )

    else:
        print(
            f"Successfully aggregated metrics for {len(aggregated_metrics_per_group)} groups."
        )
        for group_name, metrics_data in aggregated_metrics_per_group.items():
            if group_name in group_to_hpms:
                final_output_to_pickle[group_name] = {
                    "metrics": metrics_data,
                    "hpms": group_to_hpms[
                        group_name
                    ],  # HPMs are already flattened, seed & blacklisted excluded
                }
            else:
                # This case should ideally not happen if group_name comes from aggregated_metrics_per_group
                # which is derived from grouped_runs, which also populates group_to_hpms.
                print(
                    f"Warning: HPMs not found for metric group '{group_name}'. This group will be incomplete in output."
                )
                final_output_to_pickle[group_name] = {
                    "metrics": metrics_data,
                    "hpms": {},  # Empty HPMs as a fallback
                }
        # Check for groups that had HPMs but no metrics
        for group_name, hpms in group_to_hpms.items():
            if group_name not in final_output_to_pickle:
                print(
                    f"Info: Group '{group_name}' had HPMs but no aggregated metrics. Adding with empty metrics."
                )
                final_output_to_pickle[group_name] = {"metrics": {}, "hpms": hpms}

    print(f"\nSaving aggregated results to: {args.output_pkl}")
    try:
        with Path(args.output_pkl).open("wb") as f_out:
            pickle.dump(final_output_to_pickle, f_out)
        print("Successfully saved results.")
    except OSError as e:
        print(f"Error: Could not write to output file {args.output_pkl}: {e}")
    except pickle.PicklingError as e:
        print(f"Error: Could not pickle the results: {e}")
    except (ValueError, TypeError, AttributeError) as e:
        print(f"An unexpected error occurred during saving: {e}")

    print("\nScript finished.")
