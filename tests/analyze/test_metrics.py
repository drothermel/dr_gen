"""Tests for metric aggregation and curve management."""

import pytest

from dr_gen.analyze.metrics import (
    MetricCurve,
    MetricCurves,
    SplitMetrics,
    _flatten_dict_tuple_keys,
)


def test_flatten_dict_tuple_keys() -> None:
    """Test dictionary flattening with tuple keys."""
    # Simple case
    d = {"a": 1, "b": 2}
    flat = _flatten_dict_tuple_keys(d)
    assert flat == {("a",): 1, ("b",): 2}

    # Nested case
    nested = {"a": {"b": 1, "c": 2}, "d": 3}
    flat = _flatten_dict_tuple_keys(nested)
    assert flat == {("a", "b"): 1, ("a", "c"): 2, ("d",): 3}

    # Empty dict
    assert _flatten_dict_tuple_keys({}) == {}


def test_metric_curve() -> None:
    """Test MetricCurve basic functionality."""
    config = {"test": "config"}
    curve = MetricCurve(config, "train", "loss", x_name="epoch")

    # Add values
    curve.add_x_v(0, 1.0)
    curve.add_x_v(1, 0.8)
    curve.add_x_v(2, 0.6)

    assert curve.xs == [0, 1, 2]
    assert curve.vals == [1.0, 0.8, 0.6]

    # Get by x value
    assert curve.get_by_xval(1) == 0.8

    # Test with string x values
    str_curve = MetricCurve(
        config, "train", "accuracy", x_name="step", x_val_hashable=True
    )
    str_curve.add_x_v("step1", 0.5)
    str_curve.add_x_v("step2", 0.7)
    assert str_curve.get_by_xval("step1") == 0.5


def test_metric_curves() -> None:
    """Test MetricCurves managing multiple curves."""
    config = {"test": "config"}
    curves = MetricCurves(config, "train", "loss")

    # Add values with different x names
    curves.add_x_v(0, 1.0, x_name="epoch")
    curves.add_x_v(100, 0.9, x_name="step")
    curves.add_x_v(1, 0.8, x_name="epoch")

    # Get specific curves
    assert curves.get_xs("epoch") == [0, 1]
    assert curves.get_vals("epoch") == [1.0, 0.8]
    assert curves.get_xs("step") == [100]
    assert curves.get_vals("step") == [0.9]

    # Get all
    all_xs = curves.get_all_xs()
    assert "epoch" in all_xs
    assert "step" in all_xs
    assert all_xs["epoch"] == [0, 1]


def test_split_metrics() -> None:
    """Test SplitMetrics for managing metrics across different metrics."""
    config = {"test": "config"}
    split_metrics = SplitMetrics(config, "train")

    # Add different metrics
    split_metrics.add_x_v(0, 1.0, "loss")
    split_metrics.add_x_v(0, 0.1, "accuracy")
    split_metrics.add_x_v(1, 0.8, "loss")
    split_metrics.add_x_v(1, 0.3, "accuracy")

    # Get specific metrics
    assert split_metrics.get_vals("loss") == [1.0, 0.8]
    assert split_metrics.get_vals("accuracy") == [0.1, 0.3]

    # Get flattened values
    flat_vals = split_metrics.get_all_vals_flat()
    assert flat_vals[("loss", "epoch")] == [1.0, 0.8]
    assert flat_vals[("accuracy", "epoch")] == [0.1, 0.3]

    # Get by specific x value
    assert split_metrics.get_by_xval(0, "loss") == 1.0
    assert split_metrics.get_by_xval(1, "accuracy") == 0.3


def test_edge_cases() -> None:
    """Test edge cases and error handling."""
    config = {"test": "config"}
    split_metrics = SplitMetrics(config, "val")

    # Try to get non-existent metric
    with pytest.raises(AssertionError, match="loss not in curves"):
        split_metrics.get_vals("loss")

    # Add some data
    split_metrics.add_x_v(0, 1.0, "loss")

    # Try to get non-existent x_name
    curves = split_metrics.curves["loss"]
    with pytest.raises(AssertionError, match="nonexistent not in curves"):
        curves.get_xs("nonexistent")
