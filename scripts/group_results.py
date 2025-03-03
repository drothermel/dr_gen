from pathlib import Path
import json
import hashlib
import shutil

def hash_string_to_length(s, length):
    # Encode the string to bytes and compute the SHA-256 hash
    hash_obj = hashlib.sha256(s.encode('utf-8'))
    # Get the hexadecimal digest of the hash
    hex_digest = hash_obj.hexdigest()
    # Return the hash truncated to the specified length
    return hex_digest[:length]

def file_path_to_name(fpath, init):
    #/scratch/ddr8143/logs/cifar10_scratch/bs500/lr0.25/wd0.0001/s18/2025-03-01/17-39-1740868772/json_out.jsonl

    fl = fpath.parts
    bs = fl[5]
    lr = fl[6]
    wd = fl[7]
    seed = fl[8]
    date = fl[9]
    time = fl[10]
    date_time_hash = hash_string_to_length(f"{date} {time}", 6)
    return f"cifar10_{init}__{bs}_{lr}_{wd}_{seed}__{date_time_hash}.jsonl"
    

sweep_name = "lr_wd_init_v0"
root_dir = Path("/scratch/ddr8143/logs/cifar10_scratch/")
dest_dir = Path(f"/scratch/ddr8143/data/run_results/cifar_sweeps/{sweep_name}")
all_json = []
for file in root_dir.rglob('*.jsonl'):
    if file.is_file():
        with file.open('r') as f:
            # Read the first line of the file
            first_line = f.readline().strip()
            try:
                data = json.loads(first_line)
                fpath = file.resolve()
                if data['value']['model']['weights'] is None:
                    fname = file_path_to_name(fpath, "random")
                else:
                    fname = file_path_to_name(fpath, "pretrain")
                init = "random" if data['value']['model']['weights'] is None else 'pretrain'
                fname = file_path_to_name(fpath, init)
                all_json.append((fpath, file_path_to_name(fpath, init)))
                print(f" - {all_json[-1][1]}")
            except json.JSONDecodeError as e:
                print(f"Could not decode JSON in {file}: {e}")
            except KeyError as e:
                print(f"Missing key in {file}: {e}")


dest_dir.mkdir(parents=True, exist_ok=True)

metadata_path = dest_dir / "metadata.json"
with metadata_path.open('w') as f:
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
