from collections import defaultdict
from pathlib import Path

from dr_gen.utils.utils import hash_from_time
from dr_gen.analyze.log_file_data import LogFileData


def check_prefix_exclude(check_string, excluded_prefixes):
    for pre in excluded_prefixes:
        if check_string.startswith(pre):
            return True
    return False


def filter_entries_by_selection(all_entries, select_dict):
    """
    Filter a dictionary whose keys are tuples of key-value pairs (tuple of tuples)
    based on a selection dictionary.

    Parameters:
        all_entries (dict): Keys are tuple-of-tuples (e.g. ((key1, val1), (key2, val2), ...))
                            mapping to some value.
        select_dict (dict): A dict where each key maps to a list of acceptable values.

    Returns:
        dict: A new dictionary containing only those entries where, for each key in select_dict,
              the value in the tuple-of-tuples matches one of the allowed values.
    """
    result = {}
    for key_tuple, value in all_entries.items():
        # Convert the tuple-of-tuples into a dict for easy lookup.
        key_dict = dict(key_tuple)
        match = True
        for sel_key, sel_vals in select_dict.items():
            if sel_key not in key_dict or key_dict[sel_key] not in sel_vals:
                match = False
                break
        if match:
            result[key_tuple] = value
    return result


class RunGroup:
    def __init__(
        self,
    ):
        self.log_file_paths = []
        self.log_files = []
        self.name = f"temp_rg_{hash_from_time(5)}"
        self.cfg_key_remap = {
            "model.weights": "Init",
            "optim.lr": "LR",
            "optim.weight_decay": "WD",
        }
        self.cfg_val_remap = {
            "model.weights": {
                "None": "random",
                "DEFAULT": "pretrain",
            },
        }

        self.error_inds = set()
        self.ignore_inds = set()

        self.sweep_exclude_key_prefixes = [
            "paths",
            "write_checkpoint",
            "seed",
        ]
        self.all_cfg_vals = None
        self.swept_vals = None
        self.hpm_combo_to_run_inds = None

    def load_logs_from_base_dir(self, base_dir):
        log_dir = Path(base_dir)
        self.log_file_paths = [
            f.resolve() for f in log_dir.rglob("*.jsonl") if f.is_file()
        ]
        print(f">> Found {len(self.log_file_paths)} log files")

        # Parse all log files
        self.log_files = [LogFileData(fpath) for fpath in self.log_file_paths]

        # Mark the data with parse errors to ignore later
        for i, lfd in enumerate(self.log_files):
            if len(lfd.parse_errors) > 0:
                self.error_inds.add(i)

        print(
            f">> Loaded {len(self.log_files)}, {len(self.error_inds)} with parse errors"
        )

    def extract_sweeps(self):
        if len(self.log_files) == 0:
            return {}

        # cfg_key: cfg_vals: list_of_runs_with_this_value
        all_cfg_vals = defaultdict(lambda: defaultdict(list))
        for run_ind, lf in enumerate(self.log_files):
            for k, v in lf.get_flat_cfg().items():
                if check_prefix_exclude(k, self.sweep_exclude_key_prefixes):
                    continue
                all_cfg_vals[k][str(v)].append(run_ind)
        all_cfg_vals = {k: dict(v) for k, v in all_cfg_vals.items()}

        # Identify the keys that have multiple values
        swept_vals = {}
        for k, v in all_cfg_vals.items():
            if len(v) > 1:
                swept_vals[k] = v

        # Identify the runs in each combination of the swept keys
        hpm_combo_to_run_inds = defaultdict(list)
        for run_ind, lf in enumerate(self.log_files):
            run_sweep_cfg = lf.config.get_sweep_cfg(
                keys=swept_vals.keys(),
                pretty=False,  # get the raw keys
            )
            run_sweep_cfg_tuples = tuple(sorted(list(run_sweep_cfg.items())))
            hpm_combo_to_run_inds[run_sweep_cfg_tuples].append(run_ind)

        self.all_cfg_vals = all_cfg_vals
        self.swept_vals = swept_vals
        self.hpm_combo_to_run_inds = hpm_combo_to_run_inds

    def sweep_cfg_tuples_to_string_tuples(self, sweep_cfg_tuples):
        kv_strs = []
        for k, v in sweep_cfg_tuples:
            kstr = self.cfg_key_remap.get(k, k.split(".")[-1])
            vstr = str(v)
            if k in self.cfg_val_remap:
                vstr = self.cfg_val_remap[k].get(v, vstr)
            kv_strs.append((kstr, vstr))
        return sorted(kv_strs)

    def sweep_cfg_tuples_to_string(self, sweep_cfg_tuples):
        kv_str_tuples = self.sweep_cfg_tuples_to_string_tuples(
            sweep_cfg_tuples,
        )
        kv_strs = [f"{k}={v}" for k, v in kv_str_tuples]
        return " ".join(kv_strs)

    def get_swept_table_data(self):
        field_names = ["Key", "Values", "Count"]
        rows = []
        for k, v in self.swept_vals.items():
            rrs = []
            for vv, inds in v.items():
                kstr = k if len(rrs) == 0 else ""
                rrs.append([kstr, vv, len(inds)])
            rows.append(rrs)
        return field_names, rows

    def get_hpm_combo_table_data(self):
        field_names = None
        rows = []

        good_inds = set(range(len(self.log_files))) - self.error_inds - self.ignore_inds
        for sweep_cfg_tuples, run_inds in self.hpm_combo_to_run_inds.items():
            num_runs = len([ri for ri in run_inds if ri in good_inds])
            kv_str_tuples = self.sweep_cfg_tuples_to_string_tuples(sweep_cfg_tuples)
            if field_names is None:
                field_names = [k for k, _ in kv_str_tuples] + ["Count"]
            rows.append([v for _, v in kv_str_tuples] + [num_runs])
        return field_names, rows

    def select_by_hpm_combo(self, kv_select):
        # make kv_select correct format
        for k in kv_select:
            if not isinstance(kv_select[k], list):
                kv_select[k] = list(kv_select[k])

        # filter the combos by kv_select
        results_dict = filter_entries_by_selection(
            self.hpm_combo_to_run_inds,
            kv_select,
        )

        # drop any inds that have errors or we choose to ignore
        good_inds = set(range(len(self.log_files))) - self.error_inds - self.ignore_inds
        hpm_combos = {}
        for ktuples, inds in results_dict.items():
            good_res = [i for i in inds if i in good_inds]
            if len(good_res) > 0:
                hpm_combos[ktuples] = good_res
        return hpm_combos
