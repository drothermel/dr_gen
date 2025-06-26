import numpy as np
import scipy.stats as sps

from dr_gen.analyze.stats import ks_analysis


def test_ks_analysis_basic() -> None:
    # Use two small lists to test basic functionality.
    vals1 = [1, 2, 3]
    vals2 = [2, 3, 4]
    ks_stat, p_value, max_diff_value, max_diff_idx = ks_analysis(vals1, vals2)

    # Verify all return values are reasonable
    assert isinstance(ks_stat, float)
    assert isinstance(p_value, float)
    assert isinstance(max_diff_value, int | float | np.integer | np.floating)
    assert max_diff_idx is None  # Expected to be None in our implementation

    # Check that values are within expected ranges
    assert 0 <= ks_stat <= 1
    assert 0 <= p_value <= 1


def test_ks_analysis_with_scipy_verification() -> None:
    # Create random arrays and verify our results match scipy directly
    # Note: Using legacy NumPy random API for test reproducibility
    np.random.seed(42)  # noqa: NPY002
    vals1 = np.random.rand(100)  # noqa: NPY002
    vals2 = np.random.rand(150)  # noqa: NPY002

    ks_stat, p_value, max_diff_value, _ = ks_analysis(vals1, vals2)

    # Verify our results match scipy's ks_2samp directly
    scipy_result = sps.ks_2samp(vals1, vals2)
    assert np.isclose(ks_stat, scipy_result.statistic)
    assert np.isclose(p_value, scipy_result.pvalue)
    assert np.isclose(max_diff_value, scipy_result.statistic_location)


def test_ks_analysis_different_distributions() -> None:
    # Generate two different samples using a normal distribution.
    # Note: Using legacy NumPy random API for test reproducibility
    np.random.seed(0)  # noqa: NPY002
    vals1 = np.random.normal(loc=0, scale=1, size=1000)  # noqa: NPY002
    vals2 = np.random.normal(loc=0.5, scale=1, size=1000)  # noqa: NPY002

    ks_stat, p_value, max_diff_value, max_diff_idx = ks_analysis(vals1, vals2)

    # For different distributions, KS stat should be significant
    assert ks_stat > 0.1  # Expect meaningful difference
    assert p_value < 0.05  # Expect significant p-value
    assert max_diff_value is not None
