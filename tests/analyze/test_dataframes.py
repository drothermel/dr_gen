"""Tests for Polars DataFrame conversion functions."""

import polars as pl

from dr_gen.analyze.dataframes import runs_to_dataframe, runs_to_metrics_df
from dr_gen.analyze.models import Hyperparameters, Run


def test_runs_to_dataframe_empty():
    """Test converting empty list of runs."""
    df = runs_to_dataframe([])
    assert df.is_empty()


def test_runs_to_dataframe_basic():
    """Test basic run to DataFrame conversion."""
    runs = [
        Run(
            run_id="exp1",
            hyperparameters=Hyperparameters(lr=0.01, batch_size=32),
            metrics={"loss": [0.5]},
            metadata={"seed": 42},
        ),
        Run(
            run_id="exp2",
            hyperparameters=Hyperparameters(lr=0.02, batch_size=64),
            metrics={"loss": [0.6]},
            metadata={"seed": 43},
        ),
    ]

    df = runs_to_dataframe(runs)
    assert len(df) == 2
    assert "run_id" in df.columns
    assert "lr" in df.columns
    assert "batch_size" in df.columns
    assert "metadata.seed" in df.columns
    assert df["lr"].to_list() == [0.01, 0.02]


def test_runs_to_metrics_df():
    """Test converting runs to metrics DataFrame."""
    runs = [
        Run(
            run_id="exp1",
            hyperparameters=Hyperparameters(),
            metrics={"train/loss": [0.5, 0.4, 0.3], "val/acc": [0.8, 0.85]},
        )
    ]

    df = runs_to_metrics_df(runs)
    assert len(df) == 5  # 3 train/loss + 2 val/acc
    assert set(df.columns) == {"run_id", "metric", "epoch", "value"}

    # Check specific values
    train_loss = df.filter(pl.col("metric") == "train/loss")
    assert len(train_loss) == 3
    assert train_loss["value"].to_list() == [0.5, 0.4, 0.3]
