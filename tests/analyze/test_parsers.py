"""Tests for JSONL parsing utilities."""

import json
from pathlib import Path

from dr_gen.analyze.parsers import parse_jsonl_file


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
