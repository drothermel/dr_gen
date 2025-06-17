from collections import defaultdict
from pathlib import Path
from typing import Any

import dr_gen.utils.utils as gu
from dr_gen.analyze.run_data import RunData


def filter_entries_by_selection(all_entries, **kwargs: Any):  # noqa: ANN401
    result = {}
    for key_tuple, value in all_entries.items():
        # Convert the tuple-of-tuples into a dict for easy lookup.
        key_dict = dict(key_tuple)
        match = True
        for sel_key, sel_vals in kwargs.items():
            sel_vals_list = gu.make_list(sel_vals)
            if sel_key not in key_dict or key_dict[sel_key] not in sel_vals_list:
                match = False
                break
        if match:
            result[key_tuple] = value
    return result


class HpmGroup:
    def __init__(
        self,
    ) -> None:
        # hpm hash depends on important_values so store
        #  as {rid: hpm} and build {hpm: rids} on demand
        self.rid_to_hpm = {}
        self.varying_kvs = {}

    @property
    def hpm_to_rids(self):
        hpm_to_rids = defaultdict(list)
        for rid, hpm in self.rid_to_hpm.items():
            hpm_to_rids[hpm].append(rid)
        return hpm_to_rids

    @property
    def ordered_varying_keys(self):
        return sorted(self.varying_kvs.keys())

    def add_hpm(self, hpm, rid):
        self.rid_to_hpm[rid] = hpm

    def reset_all_hpms(self):
        for hpm in self.rid_to_hpm.values():
            hpm.reset_important()

    def update_important_keys_by_varying(self, exclude_prefixes=None):
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
        if len(exclude_prefixes) == 0:
            return

        for hpm in self.rid_to_hpm.values():
            hpm.exclude_prefixes_from_important(exclude_prefixes)

    def _calc_varying_kvs(self) -> None:
        all_kvs = defaultdict(set)
        for hpm in self.rid_to_hpm.values():
            for k, v in hpm.as_dict().items():
                all_kvs[k].add(str(v))
        self.varying_kvs = {k: vs for k, vs in all_kvs.items() if len(vs) > 1}

    def _set_all_hpms_important_to_varying_keys(self) -> None:
        for hpm in self.rid_to_hpm.values():
            hpm.set_important(self.varying_kvs.keys())


class RunGroup:
    def __init__(
        self,
    ) -> None:
        self.name = f"temp_rg_{gu.hash_from_time(5)}"

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
        return set(self.rid_to_run_data.keys())

    @property
    def num_runs(self):
        return len(self.rids)

    def get_display_hpm_key(self, k):
        return self.cfg_key_remap.get(k, k.split(".")[-1])

    def get_display_hpm_val(self, k, v):
        if k not in self.cfg_val_remap:
            return v
        return self.cfg_val_remap[k].get(v, v)

    def get_display_hpm_str(self, hpm):
        display = []
        for k, v in hpm.as_tupledict():
            kstr = self.get_display_hpm_key(k)
            vstr = self.get_display_hpm_val(k, v)
            display.append(f"{kstr}={vstr}")
        return " ".join(display)

    def display_hpm_key_to_real_key(self, hpm, kstr):
        def preproc(k) -> str:
            return str(k).lower().strip()

        kstr_to_k = {preproc(self.get_display_hpm_key(k)): k for k in hpm}
        kstr = preproc(kstr)
        return kstr_to_k.get(kstr, kstr)

    def display_hpm_key_val_to_real_val(self, hpm, kstr, vstr):
        k = self.display_hpm_key_to_real_key(hpm, kstr)
        if k in self.cfg_val_remap:
            vstr_to_v = {vstr: v for v, vstr in self.cfg_val_remap[k].items()}
            return vstr_to_v.get(vstr, vstr)
        return vstr

    def filter_rids(self, potential_rids):
        if isinstance(potential_rids, list):
            potential_rids = set(potential_rids)
        return list(self.rids & potential_rids)

    def ignore_rid(self, rid):
        if rid not in self.rid_to_run_data:
            return
        self.ignored_rids[rid] = self.rid_to_run_data[rid]
        del self.rid_to_run_data[rid]
        del self.hpm_group.rid_to_hpm[rid]

    def load_run(self, rid, file_path):
        run_data = RunData(file_path, rid=rid)
        if len(run_data.parse_errors) > 0:
            for _pe in run_data.parse_errors:
                pass
            self.error_rids.add(rid)
            return
        self.rid_to_run_data[rid] = run_data
        self.hpm_group.add_hpm(run_data.hpms, rid)

    def load_runs_from_base_dir(self, base_dir):
        for fp in Path(base_dir).rglob("*.jsonl"):
            if not fp.is_file():
                continue
            rid = len(self.rid_to_file)
            self.rid_to_file.append(fp.resolve)
            self.load_run(rid, fp)
        self.update_hpm_sweep_info()

    def update_hpm_sweep_info(self):
        self.hpm_group.update_important_keys_by_varying(
            exclude_prefixes=self.sweep_exclude_key_prefixes,
        )

    def get_swept_table_data(self):
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

    # Returns: { hpm: [rdata ...] }
    def select_run_data_by_hpms(self, **kwargs: Any):  # noqa: ANN401
        selected = {}
        for hpm, potential_rids in self.hpm_group.hpm_to_rids.items():
            # Made complicated by remapping display text from true
            # values
            def comp_hpm(kstr, vs, hpm_val) -> bool:
                new_vs = [str(v) for v in gu.make_list(vs)]
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
        """Select run split metrics by hyperparameters.

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
        """Select run metrics by hyperparameters.

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

    def ignore_runs_by_hpms(self, **kwargs: Any):  # noqa: ANN401
        """Ignore runs by hyperparameters.

        Marks runs matching the hyperparameter filters as ignored.
        """
        runs_to_ignore = self.select_run_data_by_hpms(**kwargs)
        for runs_list in runs_to_ignore.values():
            for run in runs_list:
                self.ignore_rid(run.id)
        self.update_hpm_sweep_info()
