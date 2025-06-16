import numpy as np

from dr_gen.analyze.ks_stats import calculate_ks_for_run_sets, find_max_diff_point


def test_find_max_diff_point_basic():
    # Use two small lists to test basic functionality.
    vals1 = [1, 2, 3]
    vals2 = [2, 3, 4]
    results = find_max_diff_point(vals1, vals2)

    # Expected keys in the results.
    expected_keys = {"all_vals", "cdf1", "cdf2", "max_idx", "max_diff_value", "ks_stat"}
    assert expected_keys.issubset(results.keys())

    # Verify that the maximum difference is correctly computed.
    differences = np.abs(results["cdf1"] - results["cdf2"])
    computed_max_diff = differences[results["max_idx"]]
    assert np.isclose(computed_max_diff, results["ks_stat"])


def test_find_max_diff_point_monotonicity():
    # Create random arrays and verify that the computed CDFs are non-decreasing and go from 0 to 1.
    np.random.seed(42)
    vals1 = np.random.rand(100)
    vals2 = np.random.rand(150)
    results = find_max_diff_point(vals1, vals2)

    cdf1 = results["cdf1"]
    cdf2 = results["cdf2"]

    # Check that both CDF arrays are non-decreasing.
    assert np.all(np.diff(cdf1) >= 0)
    assert np.all(np.diff(cdf2) >= 0)

    # The final value of each CDF should be 1.
    assert np.isclose(cdf1[-1], 1.0)
    assert np.isclose(cdf2[-1], 1.0)


def test_calculate_ks_for_run_sets(capsys):
    # Generate two different samples using a normal distribution.
    np.random.seed(0)
    vals1 = np.random.normal(loc=0, scale=1, size=1000)
    vals2 = np.random.normal(loc=0.5, scale=1, size=1000)

    results = calculate_ks_for_run_sets(vals1, vals2)

    # Verify the returned dictionary contains all the expected keys.
    expected_keys = {
        "ks_stat",
        "p_value",
        "max_diff_value",
        "all_vals",
        "cdf1",
        "cdf2",
        "max_idx",
        "seeds_group_1",
        "seeds_group_2",
    }
    assert expected_keys.issubset(results.keys())

    # Check that the seeds counts are correct.
    assert results["seeds_group_1"] == len(vals1)
    assert results["seeds_group_2"] == len(vals2)

    # Capture and verify the printed output contains key statistics.
    captured = capsys.readouterr().out
    assert "ks_stat:" in captured
    assert "p_value:" in captured
