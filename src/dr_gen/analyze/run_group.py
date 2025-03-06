from collections import defaultdict
from pathlib import Path

import dr_gen.utils.utils as gu
from dr_gen.analyze.run_data import RunData


def filter_entries_by_selection(all_entries, **kwargs):
    result = {}
    for key_tuple, value in all_entries.items():
        # Convert the tuple-of-tuples into a dict for easy lookup.
        key_dict = dict(key_tuple)
        match = True
        for sel_key, sel_vals in kwargs.items():
            sel_vals = gu.make_list(sel_vals)
            if sel_key not in key_dict or key_dict[sel_key] not in sel_vals:
                match = False
                break
        if match:
            result[key_tuple] = value
    return result


class HpmGroup:
    def __init__(
        self,
    ):
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
        return sorted(list(self.varying_kvs.keys()))

    def add_hpm(self, hpm, rid):
        self.rid_to_hpm[rid] = hpm

    def reset_all_hpms(self):
        for hpm in self.rid_to_hpm.values():
            hpm.reset_important()

    def update_important_keys_by_varying(self, exclude_prefixes=[]):
        # Start with a clean slate
        self.reset_all_hpms()

        # Set the hpm keys-to-ignore when looking for changing values
        self._exclude_prefixes_all_hpms(exclude_prefixes)

        # Calculate which (key, value) pairs are changing
        self._calc_varying_kvs()

        # Set those changing keys as the important ones in hpms
        #   so that the hashes are built based on those values
        self._set_all_hpms_important_to_varying_keys()

    def _exclude_prefixes_all_hpms(self, exclude_prefixes):
        if len(exclude_prefixes) == 0:
            return

        for hpm in self.rid_to_hpm.values():
            hpm.exclude_prefixes_from_important(exclude_prefixes)

    def _calc_varying_kvs(self):
        all_kvs = defaultdict(set)
        for hpm in self.rid_to_hpm.values():
            for k, v in hpm.as_dict().items():
                all_kvs[k].add(str(v))
        self.varying_kvs = {k: vs for k, vs in all_kvs.items() if len(vs) > 1}

    def _set_all_hpms_important_to_varying_keys(self):
        for hpm in self.rid_to_hpm.values():
            hpm.set_important(self.varying_kvs.keys())


class RunGroup:
    def __init__(
        self,
    ):
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
                None: "random",
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
        run_data = RunData(file_path)
        if len(run_data.parse_errors) > 0:
            for pe in run_data.parse_errors:
                print(pe)
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
        print(f">> {len(self.error_rids)} / {self.num_runs} files failed parsing")
        self.update_hpm_sweep_info()
        print(">> Updated hpm sweep info")

    def update_hpm_sweep_info(self):
        self.hpm_group.update_important_keys_by_varying(
            exclude_prefixes=self.sweep_exclude_key_prefixes,
        )

    def get_display_hpm_key(self, k):
        return self.cfg_key_remap.get(k, k.split(".")[-1])

    def get_swept_table_data(self):
        field_names = ["Key", "Values"]
        row_groups = []
        for k, vs in self.hpm_group.varying_kvs.items():
            rows = []
            for i, v in enumerate(vs):
                kstr = self.get_display_hpm_key(k if i == 0 else "")
                vstr = self.cfg_val_remap.get(k, {}).get(v, str(v))
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

    def select_run_data_by_hpms(self, **kwargs):
        selected = {}
        for hpm, potential_rids in self.hpm_group.hpm_to_rids.items():

            def comp_hpm(k, vs):
                new_vs = [str(v) for v in gu.make_list(vs)]
                hpm_v = hpm.get(k, None)
                return hpm_v is None or str(hpm_v) in new_vs

            if not all([comp_hpm(k, vs) for k, vs in kwargs.items()]):
                continue
            rids = self.filter_rids(potential_rids)
            if len(rids) > 0:
                selected[hpm] = [(rid, self.rid_to_run_data[rid]) for rid in rids]
        return selected

    def ignore_runs_by_hpms(self, **kwargs):
        runs_to_ignore = self.select_run_data_by_hpms(**kwargs)
        for runs_list in runs_to_ignore.values():
            for rid, _ in runs_list:
                self.ignore_rid(rid)
                print(f">> Ignoring rid: {rid}")
        self.update_hpm_sweep_info()
        print(">> Updated hpm sweep info")
