import numpy as np
import pytest
from dr_gen.analyze.result_plotting import (
    runs_metrics_to_ndarray,
    runs_metrics_dict_to_ndarray_dict,
    select_runs_by_hpms,
    trim_runs_metrics_dict,
    summary_stats,
    bootstrap_samples,
    bootstrap_summary_stats,
    make_hpm_specs,
)


def test_runs_metrics_to_ndarray():
    # Test with list of lists
    metrics = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    result = runs_metrics_to_ndarray(metrics)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 3)
    np.testing.assert_array_equal(result, np.array(metrics))

    # Test with different length lists
    metrics = [[1, 2, 3], [4, 5], [7, 8, 9]]
    result = runs_metrics_to_ndarray(metrics)
    assert result.shape == (3, 2)  # Should trim to shortest length
    np.testing.assert_array_equal(result, np.array([[1, 2], [4, 5], [7, 8]]))

    # Test with already ndarray
    metrics = np.array([[1, 2], [3, 4]])
    result = runs_metrics_to_ndarray(metrics)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, metrics)


def test_runs_metrics_dict_to_ndarray_dict():
    metrics_dict = {
        "hpm1": [[1, 2], [3, 4]],
        "hpm2": [[5, 6], [7, 8]]
    }
    result = runs_metrics_dict_to_ndarray_dict(metrics_dict)
    
    assert isinstance(result, dict)
    assert all(isinstance(v, np.ndarray) for v in result.values())
    assert result["hpm1"].shape == (2, 2)
    assert result["hpm2"].shape == (2, 2)
    
    np.testing.assert_array_equal(result["hpm1"], np.array([[1, 2], [3, 4]]))
    np.testing.assert_array_equal(result["hpm2"], np.array([[5, 6], [7, 8]]))


def test_select_runs_by_hpms():
    run_data = {
        "hpm1": [1, 2, 3],
        "hpm2": [4, 5, 6],
        "hpm3": [7, 8, 9]
    }
    
    # Test selecting specific hpms
    result = select_runs_by_hpms(run_data, ["hpm1", "hpm3"])
    assert set(result.keys()) == {"hpm1", "hpm3"}
    assert result["hpm1"] == [1, 2, 3]
    assert result["hpm3"] == [7, 8, 9]
    
    # Test with None input
    assert select_runs_by_hpms(None, ["hpm1"]) is None


def test_trim_runs_metrics_dict():
    metrics_dict = {
        "hpm1": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        "hpm2": [[10, 11, 12], [13, 14, 15]]
    }
    
    # Test trimming to 2 runs and 2 time steps
    result = trim_runs_metrics_dict(metrics_dict, nmax=2, tmax=2)
    assert len(result["hpm1"]) == 2
    assert len(result["hpm2"]) == 2
    assert all(len(run) == 2 for run in result["hpm1"])
    assert all(len(run) == 2 for run in result["hpm2"])
    
    # Test with None input
    assert trim_runs_metrics_dict(None, nmax=2, tmax=2) is None


def test_summary_stats():
    # Create sample data with known statistics
    data = np.array([
        [3, 2, 1],
        [6, 5, 4],
        [9, 8, 7]
    ])
    sorted_data = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    
    stats = summary_stats(data)

    assert np.array_equal(stats['sorted_vals'], sorted_data)
    assert np.array_equal(stats['n'], np.array([3, 3, 3]))
    assert np.array_equal(stats['mean'], np.array([2, 5, 8]))
    assert np.array_equal(stats['median'], np.array([2, 5, 8]))
    assert np.array_equal(stats['min'], np.array([1, 4, 7]))
    assert np.array_equal(stats['max'], np.array([3, 6, 9]))
    assert np.array_equal(stats['variance'], np.array([1, 1, 1]))
    assert np.array_equal(stats['std'], np.array([1, 1, 1]))
    assert np.allclose(stats['sem'], np.array([0.57735027, 0.57735027, 0.57735027]))

    # Test specific statistic request
    mean = summary_stats(data, stat="mean")
    assert np.array_equal(mean, np.array([2, 5, 8]))
    

def test_bootstrap_samples():
    dataset = np.array([1, 2, 3, 4, 5])
    
    # Test without bootstrapping
    result = bootstrap_samples(dataset)
    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 5)
    np.testing.assert_array_equal(result[0], dataset)
    
    # Test with bootstrapping
    result = bootstrap_samples(dataset, b=3)
    assert result.shape == (3, 5)
    assert all(len(set(sample)) <= 5 for sample in result)  # Each sample should have at most 5 unique values


def test_bootstrap_summary_stats():
    dataset = np.array([1, 2, 3, 4, 5])
    
    # Test with bootstrapping
    result = bootstrap_summary_stats(dataset, num_bootstraps=100)
    
    assert "point" in result
    assert "std" in result
    assert "sem" in result
    assert "ci_95" in result
    
    # Test point estimates
    assert "mean" in result["point"]
    assert "std" in result["point"]
    
    # Test confidence intervals
    assert len(result["ci_95"]["mean"]) == 2
    assert result["ci_95"]["mean"][0] < result["ci_95"]["mean"][1]


def test_make_hpm_specs():
    # Test with default values
    specs = make_hpm_specs()
    assert specs == {
        "optim.lr": 0.1,
        "optim.weight_decay": 1e-4,
        "epochs": 270
    }
    
    # Test with custom values
    specs = make_hpm_specs(lr=0.01, wd=1e-5, epochs=100)
    assert specs == {
        "optim.lr": 0.01,
        "optim.weight_decay": 1e-5,
        "epochs": 100
    }
