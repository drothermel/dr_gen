import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

from dr_gen.analyze.parsing import RunData


def _make_list(in_val: Any) -> list[Any]:  # noqa: ANN401
    """Convert input to list if not already a list."""
    return in_val if isinstance(in_val, list) else [in_val]


def filter_entries_by_selection(all_entries, **kwargs: Any):  # noqa: ANN401
    result = {}
    for key_tuple, value in all_entries.items():
        # Convert the tuple-of-tuples into a dict for easy lookup.
        key_dict = dict(key_tuple)
        match = True
        for sel_key, sel_vals in kwargs.items():
            sel_vals_list = _make_list(sel_vals)
            if sel_key not in key_dict or key_dict[sel_key] not in sel_vals_list:
                match = False
                break
        if match:
            result[key_tuple] = value
    return result


class HpmGroup:
    """Manages hyperparameter groups and tracks varying hyperparameters across runs."""

    def __init__(
        self,
    ) -> None:
        """Initialize HpmGroup to track hyperparameter mappings and variations."""
        # hpm hash depends on important_values so store
        #  as {rid: hpm} and build {hpm: rids} on demand
        self.rid_to_hpm = {}
        self.varying_kvs = {}

    @property
    def hpm_to_rids(self):
        """Map hyperparameters to their associated run IDs."""
        hpm_to_rids = defaultdict(list)
        for rid, hpm in self.rid_to_hpm.items():
            hpm_to_rids[hpm].append(rid)
        return hpm_to_rids

    @property
    def ordered_varying_keys(self):
        """Get sorted list of hyperparameter keys that vary across runs."""
        return sorted(self.varying_kvs.keys())

    def add_hpm(self, hpm, rid) -> None:
        """Add a hyperparameter configuration for a specific run ID."""
        self.rid_to_hpm[rid] = hpm

    def reset_all_hpms(self) -> None:
        """Reset important keys for all stored hyperparameter configurations."""
        for hpm in self.rid_to_hpm.values():
            hpm.reset_important()

    def update_important_keys_by_varying(self, exclude_prefixes=None) -> None:
        """Update important keys based on which hyperparameters vary across runs."""
        # Start with a clean slate
        if exclude_prefixes is None:
            exclude_prefixes = []
        self.reset_all_hpms()

        # Set the hpm keys-to-ignore when looking for changing values
        self._exclude_prefixes_all_hpms(exclude_prefixes)

        # Calculate which (key, value) pairs are changing
        self._calc_varying_kvs()

        # Set those changing keys as the important ones in hpms
        #   so that the hashes are built based on those values
        self._set_all_hpms_important_to_varying_keys()

    def _exclude_prefixes_all_hpms(self, exclude_prefixes) -> None:
        """Exclude specified prefixes from important keys in all hyperparameters."""
        if len(exclude_prefixes) == 0:
            return

        for hpm in self.rid_to_hpm.values():
            hpm.exclude_prefixes_from_important(exclude_prefixes)

    def _calc_varying_kvs(self) -> None:
        """Calculate which key-value pairs vary across all hyperparameter configs."""
        all_kvs = defaultdict(set)
        for hpm in self.rid_to_hpm.values():
            for k, v in hpm.as_dict().items():
                all_kvs[k].add(str(v))
        self.varying_kvs = {k: vs for k, vs in all_kvs.items() if len(vs) > 1}

    def _set_all_hpms_important_to_varying_keys(self) -> None:
        """Set important keys to only those that vary across configurations."""
        for hpm in self.rid_to_hpm.values():
            hpm.set_important(self.varying_kvs.keys())


class RunGroup:
    """Manages groups of runs with their hyperparameters and metrics for analysis."""

    def __init__(
        self,
    ) -> None:
        """Initialize RunGroup with default configuration and remapping settings."""
        self.name = f"temp_rg_{str(uuid.uuid4())[:8]}"

        self.rid_to_file = []
        self.rid_to_run_data = {}
        self.ignored_rids = {}
        self.hpm_group = HpmGroup()

        self.error_rids = set()

        self.cfg_key_remap = {
            "model.weights": "Init",
            "optim.lr": "LR",
            "optim.weight_decay": "WD",
        }
        self.cfg_val_remap = {
            "model.weights": {
                "None": "random",
                "DEFAULT": "pretrained",
            },
        }
        self.sweep_exclude_key_prefixes = [
            "paths",
            "write_checkpoint",
            "seed",
        ]

        self.all_cfg_vals = None
        self.swept_kvs = None
        self.swept_vals = None
        self.hpm_combo_to_run_inds = None

    @property
    def rids(self):
        """Get set of all run IDs in this group."""
        return set(self.rid_to_run_data.keys())

    @property
    def num_runs(self):
        """Get total number of runs in this group."""
        return len(self.rids)

    def get_display_hpm_key(self, k):
        """Convert hyperparameter key to display format using configured remapping."""
        return self.cfg_key_remap.get(k, k.split(".")[-1])

    def get_display_hpm_val(self, k, v):
        """Convert hyperparameter value to display format using configured remapping."""
        if k not in self.cfg_val_remap:
            return v
        return self.cfg_val_remap[k].get(v, v)

    def get_display_hpm_str(self, hpm):
        """Generate display string for a hyperparameter configuration."""
        display = []
        for k, v in hpm.as_tupledict():
            kstr = self.get_display_hpm_key(k)
            vstr = self.get_display_hpm_val(k, v)
            display.append(f"{kstr}={vstr}")
        return " ".join(display)

    def display_hpm_key_to_real_key(self, hpm, kstr):
        """Convert display key string back to actual hyperparameter key."""

        def preproc(k) -> str:
            return str(k).lower().strip()

        kstr_to_k = {preproc(self.get_display_hpm_key(k)): k for k in hpm}
        kstr = preproc(kstr)
        return kstr_to_k.get(kstr, kstr)

    def display_hpm_key_val_to_real_val(self, hpm, kstr, vstr):
        """Convert display value string back to actual hyperparameter value."""
        k = self.display_hpm_key_to_real_key(hpm, kstr)
        if k in self.cfg_val_remap:
            vstr_to_v = {vstr: v for v, vstr in self.cfg_val_remap[k].items()}
            return vstr_to_v.get(vstr, vstr)
        return vstr

    def filter_rids(self, potential_rids):
        """Filter run IDs to only those present in this group."""
        if isinstance(potential_rids, list):
            potential_rids = set(potential_rids)
        return list(self.rids & potential_rids)

    def ignore_rid(self, rid) -> None:
        """Mark a run as ignored and remove it from active tracking."""
        if rid not in self.rid_to_run_data:
            return
        self.ignored_rids[rid] = self.rid_to_run_data[rid]
        del self.rid_to_run_data[rid]
        del self.hpm_group.rid_to_hpm[rid]

    def load_run(self, rid, file_path) -> None:
        """Load run data from file and add to group if parsing succeeds."""
        run_data = RunData(file_path, rid=rid)
        if len(run_data.parse_errors) > 0:
            for _pe in run_data.parse_errors:
                pass
            self.error_rids.add(rid)
            return
        self.rid_to_run_data[rid] = run_data
        self.hpm_group.add_hpm(run_data.hpms, rid)

    def load_runs_from_base_dir(self, base_dir) -> None:
        """Load all JSONL run files from a base directory recursively."""
        for fp in Path(base_dir).rglob("*.jsonl"):
            if not fp.is_file():
                continue
            rid = len(self.rid_to_file)
            self.rid_to_file.append(fp.resolve)
            self.load_run(rid, fp)
        self.update_hpm_sweep_info()

    def update_hpm_sweep_info(self) -> None:
        """Update hyperparameter sweep information based on current runs."""
        self.hpm_group.update_important_keys_by_varying(
            exclude_prefixes=self.sweep_exclude_key_prefixes,
        )

    def get_swept_table_data(self):
        """Get table data showing swept hyperparameter keys and their values."""
        field_names = ["Key", "Values"]
        row_groups = []
        for k, vs in self.hpm_group.varying_kvs.items():
            rows = []
            for i, v in enumerate(vs):
                kstr = self.get_display_hpm_key(k if i == 0 else "")
                vstr = self.get_display_hpm_val(k, v)
                rows.append([kstr, vstr])
            row_groups.append(rows)
        return field_names, row_groups

    def get_hpms_sweep_table(self):
        """Generate table showing hyperparameter combinations and run counts."""
        raw_keys = self.hpm_group.ordered_varying_keys
        remapped_names = [self.get_display_hpm_key(k) for k in raw_keys]
        field_names = [*remapped_names, "Count"]

        rows = []
        for hpm, potential_rids in self.hpm_group.hpm_to_rids.items():
            rids = self.filter_rids(potential_rids)
            if len(rids) == 0:
                continue
            val_strs = hpm.as_valstrings(remap_kvs=self.cfg_val_remap)
            rows.append([*val_strs, len(rids)])
        return field_names, rows

    def select_run_data_by_hpms(self, **kwargs: Any):  # noqa: ANN401
        """Select run data matching specified hyperparameter filters.

        Returns: { hpm: [rdata ...] }
        """
        selected = {}
        for hpm, potential_rids in self.hpm_group.hpm_to_rids.items():
            # Made complicated by remapping display text from true
            # values
            def comp_hpm(kstr, vs, hpm_val) -> bool:
                new_vs = [str(v) for v in _make_list(vs)]
                k = self.display_hpm_key_to_real_key(hpm_val, kstr)
                k_not_found = False
                if k in hpm_val:
                    hpm_v = str(hpm_val[k])
                elif kstr in hpm_val:
                    hpm_v = str(hpm_val[kstr])
                else:
                    k_not_found = True
                    hpm_v = ""  # Initialize hmp_v for when k_not_found is True
                if not k_not_found:
                    hpm_v = self.get_display_hpm_val(k, hpm_v)
                return k_not_found or hpm_v in new_vs

            if not all(comp_hpm(kstr, vs, hpm) for kstr, vs in kwargs.items()):
                continue

            rids = self.filter_rids(potential_rids)
            if len(rids) > 0:
                selected[hpm] = [self.rid_to_run_data[rid] for rid in rids]
        return selected

    def select_run_split_metrics_by_hpms(self, metric_name, split, **kwargs: Any):  # noqa: ANN401
        """Select metrics for specific split filtered by hyperparameters.

        Returns: { hpm: [runs [metric_data ...]]}
        """
        runs = self.select_run_data_by_hpms(**kwargs)
        hpm_metrics = {}
        for hpm, rdata_list in runs.items():
            hpm_metrics[hpm] = [
                rdata.get_split_metrics(split).get_vals(metric_name)
                for rdata in rdata_list
            ]
        return hpm_metrics

    def select_run_metrics_by_hpms(self, metric_name, splits=None, **kwargs: Any):  # noqa: ANN401
        """Select metrics across multiple splits filtered by hyperparameters.

        Returns: { hpm: { split : [runs [metric_data ...]]}}
        """
        if splits is None:
            splits = ["train", "val", "eval"]
        hpm_split_metrics = defaultdict(dict)
        for split in splits:
            hpm_metrics = self.select_run_split_metrics_by_hpms(
                metric_name,
                split,
                **kwargs,
            )
            for hpm, runs_metrics in hpm_metrics.items():
                hpm_split_metrics[hpm][split] = runs_metrics
        return hpm_split_metrics

    def ignore_runs_by_hpms(self, **kwargs: Any) -> None:  # noqa: ANN401
        """Mark runs matching hyperparameter filters as ignored."""
        runs_to_ignore = self.select_run_data_by_hpms(**kwargs)
        for runs_list in runs_to_ignore.values():
            for run in runs_list:
                self.ignore_rid(run.id)
        self.update_hpm_sweep_info()
