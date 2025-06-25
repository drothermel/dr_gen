"""Tests for ExperimentDB class."""

import json

from dr_gen.analyze.experiment_db import ExperimentDB
from dr_gen.analyze.models import AnalysisConfig


def test_experiment_db_init(tmp_path):
    """Test ExperimentDB initialization."""
    config = AnalysisConfig()
    db = ExperimentDB(config=config, base_path=tmp_path, lazy=True)
    assert db.config == config
    assert db.base_path == tmp_path
    assert db.lazy is True


def test_experiment_db_load_and_query(tmp_path):
    """Test loading experiments and querying metrics."""
    # Create test data with proper run structure
    run_data = {
        "run_id": "test_run",
        "hyperparameters": {"lr": 0.01, "batch_size": 32},
        "metrics": {
            "train/loss": [2.5, 2.0, 1.5],
            "val/acc": [0.0, 0.5, 0.7],
        },
        "metadata": {"seed": 42},
    }
    
    test_file = tmp_path / "run1.jsonl"
    test_file.write_text(json.dumps(run_data))
    
    config = AnalysisConfig()
    db = ExperimentDB(config=config, base_path=tmp_path, lazy=False)
    db.load_experiments()
    
    # Test query methods
    metrics = db.query_metrics()
    assert len(metrics) > 0
    
    # Test filtering
    train_metrics = db.query_metrics(metric_filter="train")
    assert all("train" in m for m in train_metrics["metric"])