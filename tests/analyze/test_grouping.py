"""Tests for experiment grouping and hyperparameter management."""

from unittest.mock import MagicMock

from dr_gen.analyze.grouping import (
    HpmGroup,
    RunGroup,
    _make_list,
    filter_entries_by_selection,
)


def test_make_list() -> None:
    """Test make_list utility function."""
    # Already a list
    assert _make_list([1, 2, 3]) == [1, 2, 3]

    # Single value
    assert _make_list(5) == [5]
    assert _make_list("test") == ["test"]

    # None
    assert _make_list(None) == [None]

    # Empty list stays empty
    assert _make_list([]) == []


def test_filter_entries_by_selection() -> None:
    """Test filtering entries by selection criteria."""
    # Create test data with tuple keys
    entries = {
        (("lr", 0.01), ("batch_size", 32)): "result1",
        (("lr", 0.01), ("batch_size", 64)): "result2",
        (("lr", 0.001), ("batch_size", 32)): "result3",
    }

    # Filter by single criterion
    filtered = filter_entries_by_selection(entries, lr=0.01)
    assert len(filtered) == 2
    assert all("result1" in v or "result2" in v for v in filtered.values())

    # Filter by multiple criteria
    filtered = filter_entries_by_selection(entries, lr=0.01, batch_size=32)
    assert len(filtered) == 1
    assert next(iter(filtered.values())) == "result1"

    # Filter with list of values
    filtered = filter_entries_by_selection(entries, batch_size=[32, 64])
    assert len(filtered) == 3

    # No matches
    filtered = filter_entries_by_selection(entries, lr=0.1)
    assert len(filtered) == 0


def test_hpm_group_basic() -> None:
    """Test HpmGroup basic functionality."""
    hpm_group = HpmGroup()

    # Create mock hyperparameter objects
    hpm1 = MagicMock()
    hpm1.as_dict.return_value = {"lr": 0.01, "batch_size": 32}
    hpm2 = MagicMock()
    hpm2.as_dict.return_value = {"lr": 0.001, "batch_size": 32}

    # Add hyperparameters
    hpm_group.add_hpm(hpm1, "run1")
    hpm_group.add_hpm(hpm2, "run2")

    # Check storage
    assert len(hpm_group.rid_to_hpm) == 2
    assert hpm_group.rid_to_hpm["run1"] == hpm1
    assert hpm_group.rid_to_hpm["run2"] == hpm2

    # Check hpm_to_rids property
    hpm_to_rids = hpm_group.hpm_to_rids
    assert hpm1 in hpm_to_rids
    assert "run1" in hpm_to_rids[hpm1]


def test_hpm_group_varying_keys() -> None:
    """Test detection of varying hyperparameters."""
    hpm_group = HpmGroup()

    # Create hyperparameters with some varying and some constant values
    hpm1 = MagicMock()
    hpm1.as_dict.return_value = {"lr": 0.01, "batch_size": 32, "epochs": 10}
    hpm2 = MagicMock()
    hpm2.as_dict.return_value = {"lr": 0.001, "batch_size": 64, "epochs": 10}
    hpm3 = MagicMock()
    hpm3.as_dict.return_value = {"lr": 0.01, "batch_size": 32, "epochs": 10}

    hpm_group.add_hpm(hpm1, "run1")
    hpm_group.add_hpm(hpm2, "run2")
    hpm_group.add_hpm(hpm3, "run3")

    # Calculate varying keys
    hpm_group._calc_varying_kvs()  # noqa: SLF001

    # lr and batch_size vary, epochs doesn't
    assert "lr" in hpm_group.varying_kvs
    assert "batch_size" in hpm_group.varying_kvs
    assert "epochs" not in hpm_group.varying_kvs

    # Check the values that vary
    assert len(hpm_group.varying_kvs["lr"]) == 2  # 0.01 and 0.001
    assert len(hpm_group.varying_kvs["batch_size"]) == 2  # 32 and 64


def test_run_group_basic() -> None:
    """Test RunGroup basic functionality."""
    run_group = RunGroup()

    # Test initial state
    assert run_group.name.startswith("temp_rg_")
    assert run_group.num_runs == 0
    assert len(run_group.rids) == 0

    # Add some runs
    run_group.rid_to_run_data["run1"] = MagicMock()
    run_group.rid_to_run_data["run2"] = MagicMock()

    assert run_group.num_runs == 2
    assert "run1" in run_group.rids
    assert "run2" in run_group.rids


def test_run_group_display_remapping() -> None:
    """Test hyperparameter key/value remapping for display."""
    run_group = RunGroup()

    # Test key remapping
    assert run_group.get_display_hpm_key("model.weights") == "Init"
    assert run_group.get_display_hpm_key("optim.lr") == "LR"
    assert (
        run_group.get_display_hpm_key("some.other.key") == "key"
    )  # Default to last part

    # Test value remapping
    assert run_group.get_display_hpm_val("model.weights", "None") == "random"
    assert run_group.get_display_hpm_val("model.weights", "DEFAULT") == "pretrained"
    assert (
        run_group.get_display_hpm_val("model.weights", "other") == "other"
    )  # No remap
    assert (
        run_group.get_display_hpm_val("optim.lr", "0.01") == "0.01"
    )  # No remap for this key

    # Test display string generation
    hpm = MagicMock()
    hpm.as_tupledict.return_value = [
        ("model.weights", "None"),
        ("optim.lr", "0.01"),
    ]
    display_str = run_group.get_display_hpm_str(hpm)
    assert display_str == "Init=random LR=0.01"


def test_run_group_reverse_mapping() -> None:
    """Test converting display keys back to real keys."""
    run_group = RunGroup()

    # Create mock hpm with known keys
    hpm = {"model.weights": "value1", "optim.lr": "value2", "other.key": "value3"}

    # Test reverse mapping
    assert run_group.display_hpm_key_to_real_key(hpm, "Init") == "model.weights"
    assert (
        run_group.display_hpm_key_to_real_key(hpm, "init") == "model.weights"
    )  # Case insensitive
    assert run_group.display_hpm_key_to_real_key(hpm, "LR") == "optim.lr"
    assert (
        run_group.display_hpm_key_to_real_key(hpm, "unknown") == "unknown"
    )  # No mapping
