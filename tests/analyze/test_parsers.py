"""Tests for JSONL parsing utilities."""

import json
from pathlib import Path

from dr_gen.analyze.parsing import load_runs_from_dir, parse_jsonl_file


def test_parse_valid_jsonl(tmp_path: Path):
    """Test parsing valid JSONL file."""
    file_path = tmp_path / "valid.jsonl"
    data = [{"id": 1, "value": "a"}, {"id": 2, "value": "b"}]

    with file_path.open("w") as f:
        for record in data:
            f.write(json.dumps(record) + "\n")

    records, errors = parse_jsonl_file(file_path)
    assert len(records) == 2
    assert records == data
    assert len(errors) == 0


def test_parse_jsonl_with_empty_lines(tmp_path: Path):
    """Test parsing JSONL with empty lines."""
    file_path = tmp_path / "empty_lines.jsonl"
    file_path.write_text('{"id": 1}\n\n{"id": 2}\n\n')

    records, errors = parse_jsonl_file(file_path)
    assert len(records) == 2
    assert records[0]["id"] == 1
    assert records[1]["id"] == 2
    assert len(errors) == 0


def test_parse_jsonl_with_invalid_json(tmp_path: Path):
    """Test parsing JSONL with invalid JSON lines."""
    file_path = tmp_path / "invalid.jsonl"
    file_path.write_text('{"id": 1}\n{invalid json\n{"id": 2}\n')

    records, errors = parse_jsonl_file(file_path)
    assert len(records) == 2
    assert records[0]["id"] == 1
    assert records[1]["id"] == 2
    assert len(errors) == 1
    assert "Line 2" in errors[0]


def test_parse_missing_file():
    """Test parsing non-existent file."""
    records, errors = parse_jsonl_file(Path("/nonexistent/file.jsonl"))
    assert len(records) == 0
    assert len(errors) == 1
    assert "File error" in errors[0]


def test_load_runs_from_dir(tmp_path: Path):
    """Test loading runs from directory."""
    # Create test data
    run1_data = {
        "run_id": "exp001",
        "hyperparameters": {"lr": 0.01},
        "metrics": {"train/loss": [0.5, 0.4]},
    }
    run2_data = {
        "run_id": "exp002",
        "hyperparameters": {"lr": 0.02},
        "metrics": {"train/loss": [0.6, 0.5]},
    }

    # Write test files
    (tmp_path / "run1.jsonl").write_text(json.dumps(run1_data))
    (tmp_path / "run2.jsonl").write_text(json.dumps(run2_data))
    (tmp_path / "other.txt").write_text("not jsonl")

    # Load runs
    runs = load_runs_from_dir(tmp_path)
    assert len(runs) == 2
    assert runs[0].run_id == "exp001"
    assert runs[0].hpms.lr == 0.01
    assert runs[1].run_id == "exp002"
    assert runs[1].hpms.lr == 0.02


def test_load_runs_with_invalid_data(tmp_path: Path):
    """Test loading runs handles invalid data gracefully."""
    # Valid run
    valid_data = {
        "run_id": "valid",
        "hyperparameters": {"lr": 0.01},
        "metrics": {"loss": [0.5]},
    }

    # Invalid run (will use defaults)
    invalid_data = {"other": "data"}

    # Write mixed valid/invalid data
    with (tmp_path / "mixed.jsonl").open("w") as f:
        f.write(json.dumps(valid_data) + "\n")
        f.write(json.dumps(invalid_data) + "\n")

    runs = load_runs_from_dir(tmp_path)
    # Both records create valid runs (2nd uses defaults)
    assert len(runs) == 2
    assert runs[0].run_id == "valid"
    assert runs[0].hpms.lr == 0.01
    assert runs[1].run_id == "mixed"  # Uses filename as default


def test_convert_legacy_format():
    """Test converting legacy format to new Run format."""
    from dr_gen.analyze.parsing import convert_legacy_format

    legacy_data = {
        "name": "old_experiment",
        "config": {"lr": 0.01, "batch_size": 32},
        "history": [
            {"epoch": 0, "train/loss": 2.5, "val/acc": 0.5},
            {"epoch": 1, "train/loss": 2.0, "val/acc": 0.7},
        ],
        "metadata": {"timestamp": "2024-01-01"},
    }

    converted = convert_legacy_format(legacy_data)

    assert converted["run_id"] == "old_experiment"
    assert converted["hyperparameters"]["lr"] == 0.01
    assert converted["metrics"]["train/loss"] == [2.5, 2.0]
    assert converted["metrics"]["val/acc"] == [0.5, 0.7]
    assert converted["metadata"]["timestamp"] == "2024-01-01"


def test_convert_legacy_format_edge_cases():
    """Test legacy converter handles missing fields."""
    from dr_gen.analyze.parsing import convert_legacy_format

    # Minimal data
    minimal = convert_legacy_format({})
    assert minimal["run_id"] == "unknown"
    assert minimal["hyperparameters"] == {}
    assert minimal["metrics"] == {}

    # Different hyperparameter location
    with_hparams = convert_legacy_format({"hyperparameters": {"lr": 0.1}})
    assert with_hparams["hyperparameters"]["lr"] == 0.1
