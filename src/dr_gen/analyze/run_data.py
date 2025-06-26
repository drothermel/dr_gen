from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, MutableMapping
from datetime import UTC, datetime
from typing import Any

import dr_util.file_utils as fu

import dr_gen.utils.utils as gu
from dr_gen.analyze.metric_curves import SplitMetrics

# Constants for file parsing
MIN_FILE_LINES = 2
TRAIN_TIME_OFFSET_FROM_END = 2
EPOCHS_KEY = "epochs"


def parse_cfg_log_line(cfg_json: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    errors = []
    if cfg_json.get("type") != "dict_config":
        errors.append(">> Config json doesn't have {type: dict_config}")
    if "value" not in cfg_json:
        errors.append(">> Config 'value' not set")
    elif not isinstance(cfg_json["value"], dict):
        errors.append(">> Config type isn't dict")
    elif len(cfg_json["value"]) == 0:
        errors.append(">> The config is empty")
    if len(errors) > 0:
        return {}, errors
    return cfg_json["value"], errors


def get_train_time(train_time_json: dict[str, Any]) -> str | None:
    if train_time_json.get("type") == "str" and "value" in train_time_json:
        return train_time_json["value"].replace("Training time ", "")  # type: ignore[no-any-return]
    return None


def get_logged_strings(jsonl_contents: list[dict[str, Any]]) -> list[str]:
    return [
        jl["value"].strip()
        for jl in jsonl_contents
        if jl.get("type", None) == "str" and jl.get("value", "").strip() != ""
    ]


def get_logged_metrics_infer_epoch(
    hpm: Hpm, jsonl_contents: list[dict[str, Any]]
) -> dict[str, SplitMetrics]:
    metrics_by_split = {}

    epochs: dict[str, int] = defaultdict(int)
    x_name = "epoch"
    for jl in jsonl_contents:
        if "agg_stats" not in jl or "data_name" not in jl:
            continue

        split = jl["data_name"]
        if split not in metrics_by_split:
            metrics_by_split[split] = SplitMetrics(hpm, split)

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


def validate_metrics(
    expected_epochs: list[int], metrics_by_split: dict[str, SplitMetrics]
) -> list[str]:
    if expected_epochs is None or len(metrics_by_split) == 0:
        return [f"invalid input: {expected_epochs}, {len(metrics_by_split)}"]

    errors = []
    for split, split_metrics in metrics_by_split.items():
        for metric_name, xs_dict in split_metrics.get_all_xs().items():
            for x_name, xs in xs_dict.items():
                if len(xs) != expected_epochs:
                    errors.append(f"wrong_xs_len: {split} {metric_name} {x_name}")

        for metric_name, vs_dict in split_metrics.get_all_vals().items():
            for x_name, vs in vs_dict.items():
                if len(vs) != expected_epochs:
                    errors.append(f"wrong_vs_len: {split} {metric_name} {x_name}")
    return errors


# Hashable: can serve as key to a dictionary
class Hpm(MutableMapping):
    """A mutable mapping for hyperparameters with important value tracking."""

    def __init__(
        self,
        all_vals=None,
    ) -> None:
        """Initialize the Hpm instance.

        Args:
            all_vals: Dictionary of all hyperparameter values. Defaults to None.
        """
        if all_vals is None:
            all_vals = {}
        self._all_values = gu.flatten_dict(all_vals)
        self.important_values = {}
        self.reset_important()

    def __getitem__(self, key) -> Any:  # noqa: ANN401
        """Get an item from the hyperparameters."""
        return self._all_values[key]

    def __setitem__(self, key, value) -> None:
        """Set an item in the hyperparameters."""
        self._all_values[key] = value

    def __delitem__(self, key) -> None:
        """Delete an item and exclude it from important values."""
        del self._all_values[key]
        self.exclude_from_important([key])

    def __iter__(self) -> Iterator[str]:
        """Iterate over the important values."""
        return iter(self.important_values)

    def __len__(self) -> int:
        """Return the number of important values."""
        return len(self.important_values)

    def __eq__(self, other) -> bool:
        """Check equality between Hpm instances based on their hash."""
        if not isinstance(other, Hpm):
            return NotImplemented
        return hash(self) == hash(other)

    def __hash__(self) -> int:
        """Return hash of the Hpm instance based on its tupledict representation."""
        return hash(self.as_tupledict())

    def __str__(self) -> str:
        """Return string representation of important values."""
        return " ".join(self.as_strings())

    def get_from_all(self, key):
        """Get a value from all values, returning None if not found."""
        return self._all_values.get(key, None)

    def reset_important(self) -> None:
        """Reset important values to include all values."""
        self.important_values = dict(self._all_values.items())

    def exclude_from_important(self, excludes) -> None:
        """Exclude specified keys from important values."""
        self.important_values = {
            k: v for k, v in self.important_values.items() if k not in excludes
        }

    def exclude_prefixes_from_important(self, exclude_prefixes) -> None:
        """Exclude keys with specified prefixes from important values."""
        important_values = {}
        for k, v in self.important_values.items():
            if gu.check_prefix_exclude(k, exclude_prefixes):
                continue
            important_values[k] = v
        self.important_values = important_values

    def set_important(self, keys) -> None:
        """Set important values to only the specified keys."""
        self.important_values = {k: self._all_values[k] for k in keys}

    def as_dict(self):
        """Return the important values as a dictionary."""
        return self.important_values

    def as_tupledict(self, important_vals=None):
        """Convert important values to a sorted tuple dictionary."""
        if important_vals is None:
            important_vals = self.important_values
        return gu.dict_to_tupledict(important_vals)

    def as_strings(self):
        """Return important values as a list of key=value strings."""
        # Use as_tupledict to get consistent sort order
        return [f"{k}={v}" for k, v in self.as_tupledict()]

    def as_valstrings(self, remap_kvs=None):
        """Return values as strings with optional remapping."""
        if remap_kvs is None:
            remap_kvs = {}
        vs = []
        for k, v in self.as_tupledict():
            vstr = str(v)
            vs.append(remap_kvs.get(k, {}).get(vstr, vstr))
        return vs


class RunData:
    """Container for a single run's data including hyperparameters and metrics."""

    def __init__(self, file_path, rid=None) -> None:
        """Initialize RunData from a log file.

        Args:
            file_path: Path to the log file to parse.
            rid: Optional run identifier.
        """
        self.file_path = file_path
        self.id = rid
        self.hpms = None
        self.metadata = {}
        self.metrics_by_split = {}  # split: SplitMetrics

        self.parse_errors = []
        self.parse_log_file()

    def parse_log_file(self) -> None:
        """Parse the log file and extract hyperparameters, metadata, and metrics."""
        contents = fu.load_file(self.file_path)
        if contents is None:
            self.parse_errors.append(f">> Unable to load file: {self.file_path}")
            return
        if len(contents) <= MIN_FILE_LINES:
            self.parse_errors.append(">> File two lines or less, unable to parse")
            return

        # Extract Run Hyperparameters
        cfg, errors = parse_cfg_log_line(contents[0])
        if len(errors) > 0:
            self.parse_errors.extend(self.config.parse_errors)
            return
        self.hpms = Hpm(cfg)

        # Extract Run Metadata
        self.metadata["time_parsed"] = datetime.now(tz=UTC)
        self.metadata["train_time"] = get_train_time(
            contents[-TRAIN_TIME_OFFSET_FROM_END]
        )
        self.metadata["log_strs"] = get_logged_strings(contents)

        # Extract and validate metrics
        self.metrics_by_split = get_logged_metrics_infer_epoch(self.hpms, contents)
        expected_epochs = self.hpms.get(EPOCHS_KEY, None)
        errors = validate_metrics(expected_epochs, self.metrics_by_split)
        self.parse_errors.extend(errors)

    def get_split_metrics(self, split):
        """Get metrics for a specific data split."""
        assert split in self.metrics_by_split, f">> {split} not in metrics"
        return self.metrics_by_split[split]
