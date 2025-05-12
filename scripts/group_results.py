from pathlib import Path
import json
import shutil

from dr_gen.utils.utils import hash_string_to_length

def file_path_to_name(fpath):
    # /scratch/ddr8143/logs/cifar10/bs500/lr0.25/wd0.0001/s18/pretrained/2025-03-01-17-39-1740868772/json_out.jsonl

    fl = fpath.parts
    bs = fl[5]
    lr = fl[6]
    wd = fl[7]
    seed = fl[8]
    winit = fl[9]
    datetime = fl[10]
    date_time_hash = hash_string_to_length(datetime, 6)
    return f"cifar10_{winit}__{bs}_{lr}_{wd}_{seed}__{date_time_hash}.jsonl"


sweep_name = "lr_wd_init_v0_t2"
root_dir = Path("/scratch/ddr8143/logs/cifar10/")
dest_dir = Path(f"/scratch/ddr8143/data/run_results/cifar_sweeps/{sweep_name}")
all_json = []
for file in root_dir.rglob("*.jsonl"):
    if file.is_file():
        fpath = file.resolve()
        fname = file_path_to_name(fpath)
        all_json.append((fpath, fname))
dest_dir.mkdir(parents=True, exist_ok=True)

metadata_path = dest_dir / "metadata.json"
with metadata_path.open("w") as f:
    json.dump(
        {str(v): str(k) for k, v in all_json},
        f,
        indent=4,
    )

print(">> Copying files to destination")
for source_path, dest_name in all_json:
    src = Path(source_path)
    dest_file = dest_dir / dest_name
    shutil.copy(src, dest_file)

print(f">> All copied to: {dest_dir.resolve()}")
