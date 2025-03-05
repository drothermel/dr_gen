import pytest
from dr_gen.analyze.metric_curves import (
    DEFAULT_XNAME,
    MetricCurve,  
    MetricCurves,
    SplitMetrics,
)

# A dummy configuration for testing (not used in the methods)
DUMMY_CONFIG = {}

def test_initialization():
    mc = MetricCurve(DUMMY_CONFIG, split="train", metric_name="accuracy")
    assert mc.config == DUMMY_CONFIG
    assert mc.split == "train"
    assert mc.metric_name == "accuracy"
    assert mc.x_vals == []
    assert mc.metric_vals == []
    assert mc.x2met == {}

def test_add_x_v_and_properties():
    mc = MetricCurve(DUMMY_CONFIG, split="val", metric_name="loss")
    # Add a point
    mc.add_x_v(1, 0.5)
    # Verify properties and internal dict
    assert mc.xs == [1]
    assert mc.vals == [0.5]
    assert mc.x2met[1] == 0.5

    # Add another point with a different x value
    mc.add_x_v(2, 0.4)
    assert mc.xs == [1, 2]
    assert mc.vals == [0.5, 0.4]
    assert mc.x2met[2] == 0.4

def test_add_x_v_duplicate_error():
    mc = MetricCurve(DUMMY_CONFIG, split="train", metric_name="acc")
    mc.add_x_v(1, 0.8)
    with pytest.raises(AssertionError, match=">> 1 already exists"):
        mc.add_x_v(1, 0.85)

def test_add_curve():
    mc = MetricCurve(DUMMY_CONFIG, split="test", metric_name="f1")
    xs = [3, 1, 2]
    vals = [0.3, 0.1, 0.2]
    mc.add_curve(xs, vals)
    # Check that the curve is added as given.
    assert mc.x_vals == xs
    assert mc.metric_vals == vals
    # Verify mapping
    for x, val in zip(xs, vals):
        assert mc.x2met[x] == val

def test_add_curve_when_not_empty_error():
    mc = MetricCurve(DUMMY_CONFIG, split="test", metric_name="f1")
    # Add one point first.
    mc.add_x_v(1, 0.5)
    with pytest.raises(AssertionError, match=">> x vals already exist"):
        mc.add_curve([2, 3], [0.6, 0.7])

def test_sort_curve_by_x():
    mc = MetricCurve(DUMMY_CONFIG, split="train", metric_name="loss")
    # Add points in unsorted order.
    mc.add_x_v(5, 0.55)
    mc.add_x_v(2, 0.22)
    mc.add_x_v(8, 0.88)
    mc.sort_curve_by_x()
    # After sorting, the x values should be in increasing order.
    assert mc.x_vals == [2, 5, 8]
    # And the metric values should reorder accordingly.
    assert mc.metric_vals == [0.22, 0.55, 0.88]

def test_get_by_xval():
    mc = MetricCurve(DUMMY_CONFIG, split="val", metric_name="accuracy")
    mc.add_x_v(10, 0.9)
    mc.add_x_v(20, 0.95)
    # Retrieving an existing x value should return the corresponding metric value.
    assert mc.get_by_xval(10) == 0.9
    assert mc.get_by_xval(20) == 0.95
    # Requesting a non-existent x should raise an error.
    with pytest.raises(AssertionError, match=">> 30 doesn't exist"):
        mc.get_by_xval(30)

def test_non_hashable_x_conversion():
    # Test that when x_val_hashable is False, x values are converted to strings.
    mc = MetricCurve(DUMMY_CONFIG, split="train", metric_name="loss", x_val_hashable=False)
    # Using a non-hashable type (e.g., list) would normally error, so we simulate this
    # by adding a value that is not hashable but then converted to a string.
    x_val = [1, 2]  # a list, not hashable
    # When added, the x should be converted to its string representation.
    mc.add_x_v(x_val, 0.75)
    # Check that the stored x is a string.
    assert isinstance(mc.x_vals[0], str)
    # Retrieval should also convert the queried x value to string.
    assert mc.get_by_xval(x_val) == 0.75

def test_initialization():
    mc = MetricCurves(DUMMY_CONFIG, split="train", metric_name="accuracy")
    # Initially, no curves should be present.
    assert mc.curves == {}

def test_add_x_v_default_curve():
    mc = MetricCurves(DUMMY_CONFIG, split="train", metric_name="accuracy")
    # Add a point using the default x_name.
    mc.add_x_v(1, 0.8)
    # The default curve should be created automatically.
    assert DEFAULT_XNAME in mc.curves
    # Check that get_xs and get_vals return the expected lists.
    assert mc.get_xs() == [1]
    assert mc.get_vals() == [0.8]

def test_add_x_v_custom_curve():
    mc = MetricCurves(DUMMY_CONFIG, split="train", metric_name="loss")
    custom_xname = "step"
    # Add two points under a custom x_name.
    mc.add_x_v(10, 0.5, x_name=custom_xname)
    mc.add_x_v(20, 0.45, x_name=custom_xname)
    # The custom curve should be created.
    assert custom_xname in mc.curves
    # Verify that the x and metric values are stored correctly.
    assert mc.get_xs(x_name=custom_xname) == [10, 20]
    assert mc.get_vals(x_name=custom_xname) == [0.5, 0.45]

def test_get_all_xs_and_vals():
    mc = MetricCurves(DUMMY_CONFIG, split="val", metric_name="f1")
    # Add a point to the default curve.
    mc.add_x_v(1, 0.9)
    # Add a point to another curve.
    custom_xname = "not_epoch"
    mc.add_x_v(5, 0.75, x_name=custom_xname)
    
    all_xs = mc.get_all_xs()
    all_vals = mc.get_all_vals()

    # Both x_names should appear in the aggregated dictionaries.
    assert DEFAULT_XNAME in all_xs and DEFAULT_XNAME in all_vals
    assert custom_xname in all_xs and custom_xname in all_vals
    # Verify the lists are correct.
    assert all_xs[DEFAULT_XNAME] == [1]
    assert all_vals[DEFAULT_XNAME] == [0.9]
    assert all_xs[custom_xname] == [5]
    assert all_vals[custom_xname] == [0.75]

def test_get_by_xval_success():
    mc = MetricCurves(DUMMY_CONFIG, split="test", metric_name="precision")
    # Add two points to the default curve.
    mc.add_x_v(3, 0.85)
    mc.add_x_v(7, 0.90)
    # Retrieve the metric value for x=7.
    val = mc.get_by_xval(7)
    assert val == 0.90

def test_get_by_xval_nonexistent_curve():
    mc = MetricCurves(DUMMY_CONFIG, split="test", metric_name="recall")
    # Attempting to retrieve from a non-existent curve should raise an assertion error.
    with pytest.raises(AssertionError, match=">> .* not in curves"):
        mc.get_by_xval(1, x_name="nonexistent")
    

def test_initialization():
    sm = SplitMetrics(DUMMY_CONFIG, "train")
    # Initially, curves should be empty.
    assert sm.curves == {}

def test_add_x_v_default_metric():
    sm = SplitMetrics(DUMMY_CONFIG, "train")
    # Add a point for the "loss" metric using the default x_name ("epoch").
    sm.add_x_v(1, 0.8, metric_name="loss")
    xs = sm.get_xs(metric_name="loss")
    vals = sm.get_vals(metric_name="loss")
    assert xs == [1]
    assert vals == [0.8]

def test_add_x_v_custom_xname():
    sm = SplitMetrics(DUMMY_CONFIG, "train")
    custom_xname = "step"
    # Add a point for the "accuracy" metric using a custom x_name.
    sm.add_x_v(10, 0.95, metric_name="accuracy", x_name=custom_xname)
    xs = sm.get_xs(metric_name="accuracy", x_name=custom_xname)
    vals = sm.get_vals(metric_name="accuracy", x_name=custom_xname)
    assert xs == [10]
    assert vals == [0.95]

def test_get_all_xs_and_vals():
    sm = SplitMetrics(DUMMY_CONFIG, "val")
    # Add a point for metric "loss" (default x_name "epoch")
    sm.add_x_v(1, 0.8, metric_name="loss")
    # Add a point for metric "accuracy" with a custom x_name "step"
    sm.add_x_v(5, 0.95, metric_name="accuracy", x_name="step")
    
    all_xs = sm.get_all_xs()
    all_vals = sm.get_all_vals()
    
    # Check that both metric names appear.
    assert "loss" in all_xs
    assert "accuracy" in all_xs
    # Verify the inner dictionaries.
    assert all_xs["loss"].get(DEFAULT_XNAME) == [1]
    assert all_xs["accuracy"].get("step") == [5]
    assert all_vals["loss"].get(DEFAULT_XNAME) == [0.8]
    assert all_vals["accuracy"].get("step") == [0.95]

def test_get_all_xs_flat():
    sm = SplitMetrics(DUMMY_CONFIG, "test")
    # Add multiple points to "loss" and one point for "accuracy" on a custom x_name.
    sm.add_x_v(1, 0.8, metric_name="loss")
    sm.add_x_v(2, 0.75, metric_name="loss")
    sm.add_x_v(10, 0.95, metric_name="accuracy", x_name="step")
    
    flat_xs = sm.get_all_xs_flat()
    # Expected flattened keys: "loss.epoch" and "accuracy.step"
    assert ("loss", "epoch") in flat_xs
    assert ("accuracy", "step") in flat_xs
    assert flat_xs[("loss", "epoch")] == [1, 2]
    assert flat_xs[("accuracy", "step")] == [10]

def test_get_all_vals_flat():
    sm = SplitMetrics(DUMMY_CONFIG, "test")
    # Add points similarly.
    sm.add_x_v(1, 0.8, metric_name="loss")
    sm.add_x_v(2, 0.75, metric_name="loss")
    sm.add_x_v(10, 0.95, metric_name="accuracy", x_name="step")
    
    flat_vals = sm.get_all_vals_flat()
    # Expected flattened keys.
    assert ("loss", "epoch") in flat_vals
    assert ("accuracy", "step") in flat_vals
    assert flat_vals[("loss", "epoch")] == [0.8, 0.75]
    assert flat_vals[("accuracy", "step")] == [0.95]

def test_get_by_xval():
    sm = SplitMetrics(DUMMY_CONFIG, "test")
    # Add two points for the "loss" metric.
    sm.add_x_v(3, 0.85, metric_name="loss")
    sm.add_x_v(7, 0.90, metric_name="loss")
    # Retrieve the metric value by its x value.
    val = sm.get_by_xval(7, metric_name="loss")
    assert val == 0.90

def test_get_by_xval_invalid_metric():
    sm = SplitMetrics(DUMMY_CONFIG, "test")
    # Attempting to get a value for a non-existent metric should raise an assertion error.
    with pytest.raises(AssertionError, match=">> .* not in curves"):
        sm.get_by_xval(1, metric_name="nonexistent")

