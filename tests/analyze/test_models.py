"""Tests for Pydantic models in experiment analysis."""

import json

import pytest

from dr_gen.analyze.schemas import AnalysisConfig, Hpms, Run


def test_hyperparameters_basic():
    """Test basic hyperparameter creation and access."""
    hp = Hpms(learning_rate=0.01, batch_size=32)
    assert hp.learning_rate == 0.01
    assert hp.batch_size == 32


def test_hyperparameters_flatten():
    """Test flattening of nested hyperparameters."""
    hp = Hpms(
        optim={"lr": 0.01, "weight_decay": 0.0001},
        model={"name": "resnet50", "pretrained": True},
        batch_size=32,
    )
    flat = hp.flatten()
    assert flat["optim.lr"] == 0.01
    assert flat["optim.weight_decay"] == 0.0001
    assert flat["model.name"] == "resnet50"
    assert flat["model.pretrained"] is True
    assert flat["batch_size"] == 32


def test_hyperparameters_deep_nesting():
    """Test flattening with deeply nested structures."""
    hp = Hpms(optim={"scheduler": {"type": "cosine", "params": {"T_max": 100}}})
    flat = hp.flatten()
    assert flat["optim.scheduler.type"] == "cosine"
    assert flat["optim.scheduler.params.T_max"] == 100


def test_hyperparameters_serialization():
    """Test JSON serialization round-trip."""
    hp = Hpms(lr=0.01, model={"layers": [1, 2, 3]})
    json_str = json.dumps(hp.model_dump())
    reloaded = Hpms(**json.loads(json_str))
    assert reloaded.lr == hp.lr
    assert reloaded.model == hp.model


def test_run_basic():
    """Test basic Run model creation."""
    run = Run(
        run_id="exp001",
        hpms=Hpms(lr=0.01),
        metrics={"train/loss": [0.5, 0.4, 0.3]},
    )
    assert run.run_id == "exp001"
    assert run.hpms.lr == 0.01
    assert run.metrics["train/loss"] == [0.5, 0.4, 0.3]


def test_run_computed_fields():
    """Test computed fields on Run model."""
    run = Run(
        run_id="exp002",
        hpms=Hpms(),
        metrics={
            "train/loss": [0.5, 0.4, 0.3, 0.35],
            "val/acc": [0.8, 0.85, 0.9, 0.88],
        },
    )
    assert run.best_train_loss == 0.3
    assert run.best_val_acc == 0.9
    assert run.final_epoch == 3


def test_run_empty_metrics():
    """Test Run with empty or missing metrics."""
    run = Run(
        run_id="exp003",
        hpms=Hpms(),
        metrics={},
    )
    assert run.best_train_loss is None
    assert run.best_val_acc is None
    assert run.final_epoch == 0


def test_run_serialization():
    """Test Run model JSON serialization."""
    run = Run(
        run_id="exp004",
        hpms=Hpms(lr=0.01),
        metrics={"train/loss": [0.5, 0.4]},
        metadata={"timestamp": "2024-01-01", "device": "cuda"},
    )
    json_str = json.dumps(run.model_dump())
    reloaded = Run(**json.loads(json_str))
    assert reloaded.run_id == run.run_id
    assert reloaded.hpms.lr == 0.01
    assert reloaded.metadata["device"] == "cuda"


def test_run_validation_error():
    """Test Run model validation errors."""
    with pytest.raises(ValueError, match="Input should be a valid"):
        Run(
            run_id="exp005",
            hpms="not_a_hyperparameters_object",  # type: ignore
            metrics={},
        )


def test_analysis_config_defaults():
    """Test AnalysisConfig with default values."""
    config = AnalysisConfig()
    assert config.experiment_dir == "./experiments"
    assert config.output_dir == "./analysis_output"
    assert config.metric_display_names["train/loss"] == "Training Loss"
    assert config.hparam_display_names["lr"] == "Learning Rate"


def test_analysis_config_from_dict():
    """Test AnalysisConfig from dictionary."""
    config_dict = {
        "experiment_dir": "/custom/path",
        "output_dir": "/custom/output",
        "metric_display_names": {"custom/metric": "Custom Metric"},
    }
    config = AnalysisConfig(**config_dict)
    assert config.experiment_dir == "/custom/path"
    assert config.output_dir == "/custom/output"
    assert config.metric_display_names["custom/metric"] == "Custom Metric"


def test_analysis_config_env_vars(monkeypatch):
    """Test AnalysisConfig from environment variables."""
    monkeypatch.setenv("ANALYSIS_EXPERIMENT_DIR", "/env/experiments")
    monkeypatch.setenv("ANALYSIS_OUTPUT_DIR", "/env/output")

    config = AnalysisConfig()
    assert config.experiment_dir == "/env/experiments"
    assert config.output_dir == "/env/output"
