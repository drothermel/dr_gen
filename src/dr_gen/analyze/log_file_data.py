from datetime import datetime
from collections import defaultdict
import dr_util.file_utils as fu

from dr_gen.utils.utils import flatten_dict_tuple_keys
from dr_gen.analyze.metric_curves import SplitMetrics
from dr_gen.analyze.run_log_config import RunLogConfig


def get_train_time(train_time_json):
    if train_time_json.get("type", None) == "str" and "value" in train_time_json:
        return train_time_json["value"].strip("Training time ")
    return None


def get_logged_strings(jsonl_contents):
    all_strings = []
    for jl in jsonl_contents:
        if jl.get("type", None) == "str" and jl.get("value", "").strip() != "":
            all_strings.append(jl["value"].strip())
    return all_strings


def get_logged_metrics_infer_epoch(config, jsonl_contents):
    metrics_by_split = {}

    epochs = defaultdict(int)
    x_name = "epoch"
    for jl in jsonl_contents:
        if "agg_stats" not in jl or "data_name" not in jl:
            continue

        split = jl["data_name"]
        if split not in metrics_by_split:
            metrics_by_split[split] = SplitMetrics(config, split)

        metrics_dict = jl["agg_stats"]
        for metric_name, metric_val in metrics_dict.items():
            metrics_by_split[split].add_x_v(
                x=epochs[split],
                val=metric_val,
                metric_name=metric_name,
                x_name=x_name,
                x_val_hashable=True,
            )
        epochs[split] += 1
    return metrics_by_split


class LogFileData:
    def __init__(
        self,
        file_path,
    ):
        self.file_path = file_path
        self.config = None
        self.metadata = {}
        self.log_strs = []
        self.metrics_by_split = {}  # split: SplitMetrics

        self.parse_errors = []
        self.parse()

    def parse(self):
        contents = fu.load_file(self.file_path)
        if contents is None:
            self.parse_errors.append(f">> Unable to load file: {self.file_path}")
            return
        elif len(contents) <= 2:
            self.parse_errors.append(">> File two lines or less, unable to parse")
            return

        self.config = RunLogConfig(contents[0])
        if len(self.config.parse_errors) > 0:
            self.parse_errors.extend(self.config.parse_errors)
            return

        self.metadata["time_parsed"] = datetime.now()
        self.metadata["train_time"] = get_train_time(contents[-2])

        self.log_strs = get_logged_strings(contents)
        self.metrics_by_split = get_logged_metrics_infer_epoch(self.config, contents)

        # Validate length of metrics
        expected_epochs = self.config.flat_cfg[self.config.epochs_key]
        all_xs = self.get_all_xs_flat()
        all_vals = self.get_all_vals_flat()
        if not all([len(xs) == expected_epochs for xs in all_xs.values()]):
            self.parse_errors.append(">> Not all xs are the expected length")
        if not all([len(vals) == expected_epochs for vals in all_vals.values()]):
            self.parse_errors.append(">> Not all vals are the expected length")

    def get_flat_config(self):
        if self.config is None:
            return {}
        return self.config.flat_cfg

    def get_split_metrics(self, split):
        assert split in self.metrics_by_split, f">> {split} not in metrics"
        return self.metrics_by_split[split]

    def get_all_xs(self):
        xs = {}  # split: metric_name: x_name: list
        for split, split_metrics in self.metrics_by_split.items():
            xs[split] = split_metrics.get_all_xs()
        return xs

    def get_all_xs_flat(self):
        nested_xs = self.get_all_xs()
        # (split, metric_name, x_name): list
        return flatten_dict_tuple_keys(nested_xs)

    def get_all_vals(self):
        vals = {}  # split: metric_name: x_name: list
        for split, split_metrics in self.metrics_by_split.items():
            vals[split] = split_metrics.get_all_vals()
        return vals

    def get_all_vals_flat(self):
        nested_vals = self.get_all_vals()
        # (split, metric_name, x_name): list
        return flatten_dict_tuple_keys(nested_vals)
