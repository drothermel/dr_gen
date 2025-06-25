import scipy.stats as sps


def ks_analysis(values1, values2):
    """Compute KS test statistics with location of maximum difference."""
    result = sps.ks_2samp(values1, values2)
    return result.statistic, result.pvalue, result.statistic_location, None
