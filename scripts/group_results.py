import json
import re
import shutil
from pathlib import Path

# Constants for hyperparameter segment positions
XFT_SEGMENT_INDEX = 4  # 5th hyperparameter segment (0-indexed)
INIT_TYPE_SEGMENT_INDEX = 6  # 7th hyperparameter segment (0-indexed)
INIT_NAME_SEGMENT_INDEX = 7  # 8th hyperparameter segment (0-indexed)


def parse_sweep2_path_hyperparameters(path_string: str):
    normalized_path = path_string.strip("/")
    parts = normalized_path.split("/")
    hpm_dict = {}
    run_label_parts = []
    dataset_marker = "cifar10"
    hpm_start_idx_in_parts = parts.index(dataset_marker) + 1
    assert len(parts) >= hpm_start_idx_in_parts + 1

    hpm_segments = parts[hpm_start_idx_in_parts:-2] # Ignores datetime and filename
    datetime_str = parts[-2] # Second to last part is datetime
    hpm_dict["datetime"] = datetime_str
    key_num_regex = re.compile(r"^([a-zA-Z_]+)((?:[-+]?\d*\.\d+|[-+]?\d+)(?:[eE][-+]?\d+)?)$")
    for idx, segment in enumerate(hpm_segments):
        processed_as_key_num = False
        is_seed_segment = False
        match = key_num_regex.match(segment)
        if match:
            key_part = match.group(1)
            num_part_str = match.group(2)
            try:
                num_val = float(num_part_str)
                if num_val.is_integer():
                    num_val = int(num_val)

                hpm_dict[key_part] = num_val
                processed_as_key_num = True
                if key_part == "s": # Check if it's a seed parameter (e.g., "s0", "s1")
                    is_seed_segment = True
            except ValueError:
                # If num_part_str is not a valid number (e.g. "resnet8.a1_in1k" where "8.a1_in1k" is not numeric)
                # This segment does not fit the key-numeric pattern. processed_as_key_num remains False.
                pass
        if not processed_as_key_num:
            if idx == XFT_SEGMENT_INDEX:
                hpm_dict["xft"] = segment
            elif idx == INIT_TYPE_SEGMENT_INDEX:
                hpm_dict["init_type"] = segment
            elif idx == INIT_NAME_SEGMENT_INDEX:
                hpm_dict["init_name"] = segment
        if not is_seed_segment:
            run_label_parts.append(segment)
    run_label = "_".join(run_label_parts)
    return hpm_dict, run_label


def file_path_to_name(fpath):
    """/scratch/ddr8143/logs/cifar10/bs500/lr0.25/wd0.0001/s18/pretrained/2025-03-01-17-39-1740868772/json_out.jsonl
    fl = fpath.parts
    bs = fl[5]
    lr = fl[6]
    wd = fl[7]
    seed = fl[8]
    winit = fl[9]
    datetime = fl[10]
    date_time_hash = hash_string_to_length(datetime, 6)
    return f"cifar10_{winit}__{bs}_{lr}_{wd}_{seed}__{date_time_hash}.jsonl"
    """
    # /scratch/ddr8143/logs/cifar10/bs500/lr0.01/wd0.001/tup0.5/xftpycil/s0/pretrained/resnet8.a1_in1k/2025-03-01-17-39-1747039595/json_out.jsonl
    hpm_dict, run_label = parse_sweep2_path_hyperparameters(str(fpath))
    return hpm_dict, run_label



sweep_name = "resnetstrikes_t0"
root_dir = Path("/scratch/ddr8143/logs/cifar10/")
dest_dir = Path(f"/scratch/ddr8143/data/run_results/cifar_sweeps/{sweep_name}")
all_json = []
for file in root_dir.rglob("*.jsonl"):
    if file.is_file():
        fpath = file.resolve()
        hpm_dict, fname = file_path_to_name(fpath)
        all_json.append((fpath, {"run_name": fname, "seed": hpm_dict["s"], "hpms": hpm_dict}))
dest_dir.mkdir(parents=True, exist_ok=True)

# Dump the metadata into a file
metadata_path = dest_dir / "metadata.json"
with metadata_path.open("w") as f:
    json.dump(
        {str(v): str(k) for k, v in all_json},
        f,
        indent=4,
    )

# Copy files to the target destination
print(">> Copying files to destination")
for source_path, new_info_dict in all_json:
    src_path = Path(source_path)
    run_name = new_info_dict["run_name"]
    seed = new_info_dict["seed"]
    hpm_dest_dir = dest_dir / run_name
    hpm_dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = hpm_dest_dir / f"s{seed}.jsonl"
    shutil.copy(src_path, dest_path)

print(f">> All copied to: {dest_dir.resolve()}")
