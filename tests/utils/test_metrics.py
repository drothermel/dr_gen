import pytest
from dr_util.metrics import (
    BATCH_KEY,
    add_list,
    add_sum,
    agg_batch_weighted_list_avg,
    agg_none,
    agg_passthrough,
)
from omegaconf import OmegaConf

from dr_gen.utils.metrics import (
    GenMetrics,
    GenMetricsSubgroup,
    GenMetricType,
    agg_avg_list,
)

# ---------------------------------------------------
# Fixtures and Helpers
# ---------------------------------------------------


@pytest.fixture
def dummy_data_structure():
    # Define a sample data structure mapping keys to our metric types.
    return {
        "metric_int": GenMetricType.INT.value,
        "metric_list": GenMetricType.LIST.value,
        "metric_batch_weighted": GenMetricType.BATCH_WEIGHTED_AVG_LIST.value,
        "metric_avg": GenMetricType.AVG_LIST.value,
    }


@pytest.fixture
def dummy_cfg(dummy_data_structure):
    # Minimal dummy config; note that GenMetricsSubgroup expects a cfg and a group name.
    return OmegaConf.create(
        {
            "train": {"run": True},
            "dev": {"run": True},
            "val": {"run": True},
            "data": {
                "name": "dummy",
                "train": {"batch_size": 4},
                "val": {"batch_size": 4},
            },
            "metrics": {
                "loggers": [],
                "init": {
                    "batch_size": "list",
                    **dummy_data_structure,
                },
            },
        }
    )


@pytest.fixture
def subgroup_with_fake_add(dummy_cfg):
    # Create a subgroup instance and override _add_tuple to record calls.
    subgroup = GenMetricsSubgroup(dummy_cfg, "test")
    # Initialize an empty data dict for testing.
    subgroup.data = {}
    calls = []

    def fake_add_tuple(key, value):
        calls.append((key, value))
        # For testing, accumulate values in a list per key.
        subgroup.data.setdefault(key, []).append(value)

    subgroup._add_tuple = fake_add_tuple  # override the internal method
    subgroup._recorded_calls = calls  # attach for inspection
    return subgroup


# ---------------------------------------------------
# Tests for GenMetricsSubgroup internal initialization
# ---------------------------------------------------


def test_init_data_values(dummy_cfg):
    # Create an instance and manually set its data_structure
    subgroup = GenMetricsSubgroup(dummy_cfg, "test")
    subgroup._init_data()  # This calls _init_data_values and _init_data_fxns

    # Check that the data values are set correctly
    assert subgroup.data["metric_int"] == 0
    assert subgroup.data["metric_list"] == []
    assert subgroup.data["metric_batch_weighted"] == []
    assert subgroup.data["metric_avg"] == []


def test_init_data_fxns(dummy_cfg):
    subgroup = GenMetricsSubgroup(dummy_cfg, "test")
    subgroup._init_data_fxns()

    # Check that the add and aggregation functions are set as expected.
    # (We compare function objects here.)
    assert subgroup.add_fxns["metric_int"] == add_sum
    assert subgroup.agg_fxns["metric_int"] == agg_passthrough

    assert subgroup.add_fxns["metric_list"] == add_list
    assert subgroup.agg_fxns["metric_list"] == agg_none

    assert subgroup.add_fxns["metric_batch_weighted"] == add_list
    assert subgroup.agg_fxns["metric_batch_weighted"] == agg_batch_weighted_list_avg

    assert subgroup.add_fxns["metric_avg"] == add_list
    assert subgroup.agg_fxns["metric_avg"] == agg_avg_list


def test_clear_data(dummy_cfg):
    subgroup = GenMetricsSubgroup(dummy_cfg, "test")
    subgroup.data_structure = {
        "metric_int": GenMetricType.INT.value,
        "metric_list": GenMetricType.LIST.value,
    }
    subgroup._init_data()  # Initialize data

    # Simulate updates to the data
    subgroup.data["metric_int"] = 42
    subgroup.data["metric_list"].append(99)

    # Call clear_data and verify that the data is reset to its initial state.
    subgroup.clear_data()
    assert subgroup.data["metric_int"] == 0
    assert subgroup.data["metric_list"] == []


# ---------------------------------------------------
# Tests for the overloaded 'add' methods (tuple and dict)
# ---------------------------------------------------


def test_add_tuple_dispatch(subgroup_with_fake_add):
    # Call add with a tuple; expected behavior:
    #   - The tuple version asserts that the input has exactly 2 items.
    #   - It calls _add_tuple(key, val) and, if ns is provided, also _add_tuple(BATCH_KEY, ns).
    subgroup_with_fake_add.add(("metric_int", 5), ns=10)
    # Verify that _add_tuple was called with ("metric_int", 5) and (BATCH_KEY, 10)
    assert ("metric_int", 5) in subgroup_with_fake_add._recorded_calls
    assert (BATCH_KEY, 10) in subgroup_with_fake_add._recorded_calls


def test_add_dict_dispatch(subgroup_with_fake_add):
    # Call add with a dict. It should iterate over the key-value pairs.
    subgroup_with_fake_add.add({"metric_list": 3, "metric_int": 2}, ns=20)
    # Check that both key-value pairs were passed to _add_tuple,
    # plus an additional call for the batch key.
    recorded = subgroup_with_fake_add._recorded_calls
    assert ("metric_list", 3) in recorded
    assert ("metric_int", 2) in recorded
    assert (BATCH_KEY, 20) in recorded


# ---------------------------------------------------
# Tests for the GenMetrics class
# ---------------------------------------------------


def test_gen_metrics_log_data(dummy_cfg):
    # Create a dummy config that includes two groups: train and val.
    metrics = GenMetrics(dummy_cfg)
    # Log some data to the "train" group.
    metrics.log_data({"metric_int": 5}, "train", ns=10)
    # Expect that for group "train" the dict version of add was used.
    group_data = metrics.groups["train"].data

    assert group_data.get("metric_int") == 5
    assert group_data.get(BATCH_KEY) == [10]


def test_gen_metrics_log_data_invalid_group(dummy_cfg):
    metrics = GenMetrics(dummy_cfg)
    # Log data to an invalid group should raise an assertion.
    with pytest.raises(AssertionError, match=">> Invalid group name:"):
        metrics.log_data({"metric_int": 5}, "invalid_group")


def test_gen_metrics_clear_data(dummy_cfg):
    metrics = GenMetrics(dummy_cfg)
    metrics.log_data({"metric_list": 2}, "val", ns=None)
    metrics.log_data({"metric_list": 4}, "train", ns=10)
    metrics.log_data(
        {"metric_int": 3, "metric_batch_weighted": 5, "metric_avg": 6}, "train", ns=None
    )

    assert metrics.groups["val"].data.get("metric_list") == [2]
    group_data = metrics.groups["train"].data
    assert group_data.get("metric_int") == 3
    assert group_data.get("metric_list") == [4]
    assert group_data.get("metric_batch_weighted") == [5]
    assert group_data.get("metric_avg") == [6]
    assert group_data.get(BATCH_KEY) == [10]

    # Clear only the "train" group.
    metrics.clear_data("train")
    assert metrics.groups["val"].data.get("metric_list") == [2]
    group_data = metrics.groups["train"].data
    assert group_data.get("metric_int") == 0
    assert group_data.get("metric_list") == []
    assert group_data.get("metric_batch_weighted") == []
    assert group_data.get("metric_avg") == []
    assert group_data.get(BATCH_KEY) == []
