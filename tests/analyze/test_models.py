"""Tests for Pydantic models in experiment analysis."""

import json

from dr_gen.analyze.models import Hyperparameters


def test_hyperparameters_basic():
    """Test basic hyperparameter creation and access."""
    hp = Hyperparameters(learning_rate=0.01, batch_size=32)
    assert hp.learning_rate == 0.01
    assert hp.batch_size == 32


def test_hyperparameters_flatten():
    """Test flattening of nested hyperparameters."""
    hp = Hyperparameters(
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
    hp = Hyperparameters(
        optim={"scheduler": {"type": "cosine", "params": {"T_max": 100}}}
    )
    flat = hp.flatten()
    assert flat["optim.scheduler.type"] == "cosine"
    assert flat["optim.scheduler.params.T_max"] == 100


def test_hyperparameters_serialization():
    """Test JSON serialization round-trip."""
    hp = Hyperparameters(lr=0.01, model={"layers": [1, 2, 3]})
    json_str = json.dumps(hp.model_dump())
    reloaded = Hyperparameters(**json.loads(json_str))
    assert reloaded.lr == hp.lr
    assert reloaded.model == hp.model
