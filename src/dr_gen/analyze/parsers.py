"""JSONL parsing utilities for experiment data."""

import json
from pathlib import Path
from typing import Any


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
