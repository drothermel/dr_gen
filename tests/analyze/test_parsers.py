"""Tests for JSONL parsing utilities."""

import json
from pathlib import Path

from dr_gen.analyze.parsers import load_runs_from_dir, parse_jsonl_file


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
    assert runs[0].hyperparameters.lr == 0.01
    assert runs[1].run_id == "exp002"
    assert runs[1].hyperparameters.lr == 0.02


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
    assert runs[0].hyperparameters.lr == 0.01
    assert runs[1].run_id == "mixed"  # Uses filename as default
