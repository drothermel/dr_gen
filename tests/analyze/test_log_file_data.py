import dr_gen.analyze.log_file_data as lfd
from dr_gen.analyze.log_file_data import LogFileData

# A minimal valid config line.
VALID_CONFIG = {"type": "dict_config", "value": {"epochs": 2, "dummy": "value"}}
# Two metric lines for the same split "train".
METRIC_LINE1 = {"agg_stats": {"loss": 0.8, "accuracy": 0.7}, "data_name": "train"}
METRIC_LINE2 = {"agg_stats": {"loss": 0.75, "accuracy": 0.72}, "data_name": "train"}
# A training time line (note: get_train_time strips "Training time ").
TRAIN_TIME_LINE = {"type": "str", "value": "Training time 5"}
# A log string line.
LOG_STRING_LINE = {"type": "str", "value": "Some log message"}

# Build a valid file content list.
VALID_FILE_CONTENTS = [
    VALID_CONFIG,  # line 1: config
    METRIC_LINE1,  # line 2: metric data
    METRIC_LINE2,  # line 3: metric data
    TRAIN_TIME_LINE,  # line 4: used for train_time (second-last line)
    LOG_STRING_LINE,  # line 5: additional log string
]


# Dummy loader functions to simulate different file scenarios.
def dummy_load_file_valid(file_path):
    return VALID_FILE_CONTENTS


def dummy_load_file_none(file_path):
    return None


def dummy_load_file_few(file_path):
    # Return only two lines (too few to parse)
    return [VALID_CONFIG, METRIC_LINE1]


def dummy_load_file_invalid_config(file_path):
    # Config line is invalid because it does not have type "dict_config".
    invalid_config = {"type": "str", "value": "invalid"}
    return [
        invalid_config,
        METRIC_LINE1,
        METRIC_LINE2,
        TRAIN_TIME_LINE,
        LOG_STRING_LINE,
    ]


def test_get_train_time_valid():
    # A valid input should strip the prefix and return the training time.
    train_time_json = {"type": "str", "value": "Training time 5:30"}
    result = lfd.get_train_time(train_time_json)
    assert result == "5:30"


def test_get_train_time_invalid():
    # Missing "type" key.
    assert lfd.get_train_time({"value": "Training time 5:30"}) is None
    # Wrong type.
    assert lfd.get_train_time({"type": "dict", "value": "Training time 5:30"}) is None
    # Missing "value" key.
    assert lfd.get_train_time({"type": "str"}) is None


def test_get_logged_strings():
    jsonl_contents = [
        {"type": "str", "value": "  hello  "},
        {"type": "str", "value": "   "},  # should be ignored (empty after strip)
        {"type": "dict", "value": "not a string"},
        {"value": "missing type"},
        {"type": "str", "value": "world"},
    ]
    result = lfd.get_logged_strings(jsonl_contents)
    # Only non-empty string lines should be returned, trimmed.
    assert result == ["hello", "world"]


def test_get_logged_metrics_infer_epoch():
    # Dummy config can be an empty dict.
    config = {}
    jsonl_contents = [
        # For split "train": two lines with two metrics each.
        {"agg_stats": {"loss": 0.8, "accuracy": 0.75}, "data_name": "train"},
        {"agg_stats": {"loss": 0.7, "accuracy": 0.78}, "data_name": "train"},
        # For split "val": one line with one metric.
        {"agg_stats": {"loss": 0.9}, "data_name": "val"},
        # A line that should be ignored (wrong type).
        {"type": "str", "value": "This is a log message"},
        # Another valid line for "train" with only the "loss" metric.
        {"agg_stats": {"loss": 0.65}, "data_name": "train"},
        # A line missing "agg_stats" should be ignored.
        {"data_name": "val"},
    ]

    metrics_by_split = lfd.get_logged_metrics_infer_epoch(config, jsonl_contents)

    # We expect two splits: "train" and "val"
    assert "train" in metrics_by_split
    assert "val" in metrics_by_split

    # For split "train", check that the curves contain points in order of appearance.
    train_metrics = metrics_by_split["train"]
    # For metric "loss" in "train", the x values should be [0, 1, 2] and corresponding values [0.8, 0.7, 0.65]
    xs_loss = train_metrics.get_xs(metric_name="loss")
    vals_loss = train_metrics.get_vals(metric_name="loss")
    assert xs_loss == [0, 1, 2]
    assert vals_loss == [0.8, 0.7, 0.65]

    # For metric "accuracy" in "train", there were only two lines.
    xs_accuracy = train_metrics.get_xs(metric_name="accuracy")
    vals_accuracy = train_metrics.get_vals(metric_name="accuracy")
    assert xs_accuracy == [0, 1]
    assert vals_accuracy == [0.75, 0.78]

    # For split "val", only "loss" was provided, with x value 0.
    val_metrics = metrics_by_split["val"]
    xs_loss_val = val_metrics.get_xs(metric_name="loss")
    vals_loss_val = val_metrics.get_vals(metric_name="loss")
    assert xs_loss_val == [0]
    assert vals_loss_val == [0.9]


# Test when the file cannot be loaded.
def test_log_file_data_file_not_found(monkeypatch):
    monkeypatch.setattr("dr_util.file_utils.load_file", dummy_load_file_none)
    file_path = "dummy_path"
    lfd = LogFileData(file_path)
    assert f">> Unable to load file: {file_path}" in lfd.parse_errors
    assert lfd.config is None


# Test when the file has too few lines.
def test_log_file_data_file_too_few_lines(monkeypatch):
    monkeypatch.setattr("dr_util.file_utils.load_file", dummy_load_file_few)
    file_path = "dummy_path"
    lfd = LogFileData(file_path)
    assert ">> File two lines or less, unable to parse" in lfd.parse_errors
    assert lfd.config is None


# Test when the configuration (first line) is invalid.
def test_log_file_data_invalid_config(monkeypatch):
    monkeypatch.setattr("dr_util.file_utils.load_file", dummy_load_file_invalid_config)
    file_path = "dummy_path"
    lfd = LogFileData(file_path)
    # RunLogConfig should add an error if "type" is not "dict_config".
    assert ">> Config json doesn't have {type: dict_config}" in lfd.parse_errors
    # The config attribute is set (though erroneous).
    assert lfd.config is not None


# Test a valid file and all the resulting parsed properties.
def test_log_file_data_valid(monkeypatch):
    monkeypatch.setattr("dr_util.file_utils.load_file", dummy_load_file_valid)
    file_path = "dummy_path"
    lfd = LogFileData(file_path)

    # No parse errors should be present.
    assert lfd.parse_errors == []
    # The config should be set from the first line.
    assert lfd.config is not None

    # Metadata should include the time of parsing and the training time.
    assert "time_parsed" in lfd.metadata
    # get_train_time should strip the prefix, so we expect "5".
    assert lfd.metadata["train_time"] == "5"

    # The logged strings should be extracted from lines with type "str".
    # In our file, that is TRAIN_TIME_LINE and LOG_STRING_LINE.
    log_strs = lfd.log_strs
    assert "Training time 5" in log_strs
    assert "Some log message" in log_strs

    # Metrics should be parsed for the "train" split.
    assert "train" in lfd.metrics_by_split

    # The helper get_logged_metrics_infer_epoch adds points using an epoch counter per split.
    # Since our valid file provided two metric lines for "train", we expect each metric to have 2 entries.
    # Verify via the flattened xs and vals.
    all_xs_flat = lfd.get_all_xs_flat()
    all_vals_flat = lfd.get_all_vals_flat()
    for key, xs in all_xs_flat.items():
        assert len(xs) == 2, f"Expected 2 x values for {key}, got {len(xs)}"
    for key, vals in all_vals_flat.items():
        assert len(vals) == 2, f"Expected 2 metric values for {key}, got {len(vals)}"

    # Additionally, test that get_split_metrics returns the correct object.
    split_metrics = lfd.get_split_metrics("train")
    # For example, for metric "loss" with default x_name "epoch", we expect x values [0, 1] and values [0.8, 0.75].
    xs_loss = split_metrics.get_xs(metric_name="loss")
    vals_loss = split_metrics.get_vals(metric_name="loss")
    assert xs_loss == [0, 1]
    assert vals_loss == [0.8, 0.75]
