"""JSONL parsing utilities for experiment data."""

import json
from pathlib import Path
from typing import Any

from dr_gen.analyze.models import Hyperparameters, Run


def parse_jsonl_file(filepath: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Parse a JSONL file and return valid records and errors.

    Returns:
        Tuple of (valid_records, error_messages)
    """
    records = []
    errors = []

    try:
        with filepath.open() as f:
            for line_num, line_raw in enumerate(f, 1):
                line = line_raw.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    errors.append(f"Line {line_num}: {e}")
    except OSError as e:
        errors.append(f"File error: {e}")

    return records, errors


def load_runs_from_dir(directory: Path, pattern: str = "*.jsonl") -> list[Run]:
    """Load all runs from JSONL files in a directory.

    Args:
        directory: Directory containing JSONL files
        pattern: Glob pattern for files to load

    Returns:
        List of validated Run models
    """
    runs = []
    for filepath in sorted(directory.glob(pattern)):
        records, _ = parse_jsonl_file(filepath)
        for record in records:
            try:
                hp = Hyperparameters(**record.get("hyperparameters", {}))
                run = Run(
                    run_id=record.get("run_id", filepath.stem),
                    hyperparameters=hp,
                    metrics=record.get("metrics", {}),
                    metadata=record.get("metadata", {}),
                )
                runs.append(run)
            except Exception:  # noqa: BLE001, S112
                continue
    return runs
