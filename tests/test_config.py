"""Tests for configuration constants."""

import dr_gen.config as config


def test_split_names() -> None:
    """Test that SPLIT_NAMES contains expected training phases."""
    assert isinstance(config.SPLIT_NAMES, list)
    assert len(config.SPLIT_NAMES) == 3
    assert "train" in config.SPLIT_NAMES
    assert "val" in config.SPLIT_NAMES
    assert "eval" in config.SPLIT_NAMES
    # Ensure no duplicates
    assert len(set(config.SPLIT_NAMES)) == len(config.SPLIT_NAMES)


def test_avail_datasets() -> None:
    """Test that AVAIL_DATASETS contains expected dataset names."""
    assert isinstance(config.AVAIL_DATASETS, list)
    assert len(config.AVAIL_DATASETS) >= 2
    assert "cifar10" in config.AVAIL_DATASETS
    assert "cifar100" in config.AVAIL_DATASETS
    # Ensure no duplicates
    assert len(set(config.AVAIL_DATASETS)) == len(config.AVAIL_DATASETS)
    # Ensure all lowercase
    assert all(d.islower() for d in config.AVAIL_DATASETS)
