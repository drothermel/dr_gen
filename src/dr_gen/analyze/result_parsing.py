from pathlib import Path
import random
from collections import defaultdict
from datetime import datetime

import dr_util.file_utils as fu

MIN_FILE_LEN = 2  # cfg and "end run"


# Return a list of [(file_path, file_contents), ...]
def get_all_logs(base_dir):
    print(f">> Getting logs from all runs in:\n    {base_dir}")
    log_dir = Path(base_dir)
    all_files = [f.resolve() for f in log_dir.rglob("*.jsonl") if f.is_file()]
    print(f">> Found {len(all_files):,} files")
    print(">> Loading files")
    return [
        {
            "log_path": f,
            "log_data": fu.load_file(f),
        }
        for f in all_files
    ]


def extract_sweeps(
    parsed_runs,
    exclude_prefixes=[
        "paths",
        "write_checkpoint",
        "seed",
    ],
):
    if len(parsed_runs) == 0:
        return {}

    # Group all run indices by cfg values
    cfg_vals = defaultdict(lambda: defaultdict(list))
    for i, (cfg, _, _) in enumerate(parsed_runs):
        for k, v in cfg.items():
            if any([k.startswith(pre) for pre in exclude_prefixes]):
                continue
            cfg_vals[k][str(v)].append(i)
    cfg_vals = {k: dict(v) for k, v in cfg_vals.items()}

    # Identify the keys that have multiple values
    swept_vals = {}
    for k, v in cfg_vals.items():
        if len(v) > 1:
            swept_vals[k] = v

    # Identify the runs in each combination of the swept keys
    combo_inds = defaultdict(list)
    key_order = list(swept_vals.keys())
    for i, (cfg, _, _) in enumerate(parsed_runs):
        ind_combo = tuple([str(cfg[k]) for k in key_order])
        combo_inds[ind_combo].append(i)

    return {
        "all_cfg_vals": cfg_vals,
        "swept_vals": swept_vals,
        "combo_key_order": key_order,
        "combo_inds": combo_inds,
    }


def get_swept_table(swept_vals):
    fns = ["Key", "Values", "Count"]
    rows = []
    for k, v in swept_vals.items():
        rrs = []
        for vv, inds in v.items():
            kstr = k if len(rrs) == 0 else ""
            rrs.append([kstr, vv, len(inds)])
        rows.append(rrs)
    return fns, rows


def get_combo_table_contents(key_order, combo_inds):
    keys = key_order
    name_to_ind = {n: i for i, n in enumerate(keys)}
    name_overrides = {
        "model.weights": "Init",
        "optim.lr": "LR",
        "optim.weight_decay": "WD",
    }
    ind_order = [name_to_ind[k] for k in name_overrides]
    ind_order.extend([i for k, i in name_to_ind.items() if k not in name_overrides])

    field_names = [name_overrides.get(keys[i], keys[i]) for i in ind_order] + ["Count"]
    contents = []
    for k, v in sorted(
        list(combo_inds.items()),
        #              init     epochs   lr       wd
        key=lambda x: (x[0][0], x[0][4], x[0][2], x[0][1]),
        reverse=True,
    ):
        contents.append([k[i] for i in ind_order] + [len(v)])
    return field_names, contents


def get_inds_by_kvs(key_order, combo_inds, kv_select):
    selected_rows = list(combo_inds.keys())
    name_to_ind = {k: i for i, k in enumerate(key_order)}
    for k, v in kv_select.items():
        # Can't control keys that aren't swept
        if k not in key_order:
            continue
        key_ind = name_to_ind[k]
        selected_rows = [sr for sr in selected_rows if sr[key_ind] == v]
    return {sr: combo_inds[sr] for sr in selected_rows}


def remap_run_list_metrics(
    run_list,
    splits=["train", "val", "eval"],
    metric_names=["epoch", "loss", "acc1", "acc5"],
):
    # input: [(cfg, run_met, md) for run in run_group]
    #        run_met = {split: [{"epoch": ..., "metric_name": val, ...} for epochs]}
    # goal: { split: metric_name: [values_list for run in run_group] }
    metrics = {spl: {mn: [] for mn in metric_names} for spl in splits}
    for _, run_metrics, _ in run_list:  # iterate through runs
        for split in splits:
            epoch_list = run_metrics[split]
            for mn in metric_names:
                metrics[split][mn].append(  # append one list per run to mn
                    [vdict[mn] for vdict in epoch_list]
                )
    return metrics


def get_run_metrics(
    remapped_mets,
    split,
    metric,
    run_ind,
):
    return remapped_mets[split][metric][run_ind]


def get_runs_metrics(
    remapped_mets,
    split,
    metric,
    run_ind_list,
):
    return [remapped_mets[split][metric][ri] for ri in run_ind_list]


def get_selected_run_metrics(
    remapped_mets,
    split,
    metric,
    selected_inds,
):
    selected_metrics = {}
    for srs, inds in selected_inds.items():
        selected_metrics[srs] = get_runs_metrics(
            remapped_mets,
            split,
            metric,
            inds,
        )
    return selected_metrics


def get_selected_combo(
    runs,
    remapped_metrics,
    sweep_info,
    kv_select,
    splits=["train", "val", "eval"],
    metric="acc1",
    ignore_keys=[],
    num_seeds=None,
):
    # Get the inds associated with these kv_select
    selected_inds = get_inds_by_kvs(
        sweep_info["combo_key_order"],
        sweep_info["combo_inds"],
        kv_select,
    )

    # Get the metrics associated with these inds
    split_keys = defaultdict(list)
    split_vals = defaultdict(list)
    for split in splits:
        selected_metrics = get_selected_run_metrics(
            remapped_metrics,
            split,
            metric,
            selected_inds,
        )
        for sm_k, sm_vals in selected_metrics.items():
            split_keys[split].append(sm_k)
            if num_seeds is None or len(sm_vals) <= num_seeds:
                split_vals[split].append(sm_vals)
            else:
                split_vals[split].append(random.sample(sm_vals, num_seeds))

    all_kvs = []
    for klist in split_keys[splits[0]]:
        all_kvs.append(
            [
                (k, v)
                for k, v in zip(
                    sweep_info["combo_key_order"],
                    klist,
                )
                if k not in ignore_keys
            ]
        )
    return all_kvs, split_vals, selected_inds
