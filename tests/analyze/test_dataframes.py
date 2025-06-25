"""Tests for Polars DataFrame conversion functions."""

import polars as pl

from dr_gen.analyze.dataframes import (
    find_varying_hparams,
    group_by_hparams,
    query_metrics,
    remap_display_names,
    runs_to_dataframe,
    runs_to_metrics_df,
    summarize_by_hparams,
)
from dr_gen.analyze.models import AnalysisConfig, Hyperparameters, Run


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


def test_find_varying_hparams():
    """Test finding hyperparameters that vary across runs."""
    runs = [
        Run(
            run_id="exp1",
            hyperparameters=Hyperparameters(lr=0.01, batch_size=32, seed=42),
            metrics={},
        ),
        Run(
            run_id="exp2",
            hyperparameters=Hyperparameters(lr=0.02, batch_size=32, seed=42),
            metrics={},
        ),
    ]

    df = runs_to_dataframe(runs)
    varying = find_varying_hparams(df)
    assert varying == ["lr"]  # Only lr varies


def test_group_by_hparams():
    """Test grouping runs by hyperparameters."""
    runs = [
        Run(
            run_id="exp1",
            hyperparameters=Hyperparameters(lr=0.01, arch="cnn"),
            metrics={},
        ),
        Run(
            run_id="exp2",
            hyperparameters=Hyperparameters(lr=0.01, arch="cnn"),
            metrics={},
        ),
        Run(
            run_id="exp3",
            hyperparameters=Hyperparameters(lr=0.02, arch="cnn"),
            metrics={},
        ),
    ]

    df = runs_to_dataframe(runs)
    grouped = group_by_hparams(df, ["lr"])

    assert len(grouped) == 2
    assert set(grouped.columns) == {"lr", "run_ids"}

    # Check grouping
    lr_01 = grouped.filter(pl.col("lr") == 0.01)
    assert len(lr_01["run_ids"][0]) == 2  # Two runs with lr=0.01


def test_query_metrics():
    """Test metric querying with filters."""
    runs = [
        Run(
            run_id="exp1",
            hyperparameters=Hyperparameters(),
            metrics={"train/loss": [0.5], "val/loss": [0.6]},
        )
    ]

    metrics_df = runs_to_metrics_df(runs)

    # Test metric filter
    train_only = query_metrics(metrics_df, metric_filter="train")
    assert len(train_only) == 1
    assert train_only["metric"][0] == "train/loss"

    # Test run filter
    filtered = query_metrics(metrics_df, run_filter=["exp1"])
    assert len(filtered) == 2  # Both metrics for exp1


def test_summarize_by_hparams():
    """Test metric summarization by hyperparameters."""
    runs = [
        Run(
            run_id="exp1",
            hyperparameters=Hyperparameters(lr=0.01),
            metrics={"loss": [0.5, 0.4]},
        ),
        Run(
            run_id="exp2",
            hyperparameters=Hyperparameters(lr=0.01),
            metrics={"loss": [0.6, 0.5]},
        ),
    ]

    runs_df = runs_to_dataframe(runs)
    metrics_df = runs_to_metrics_df(runs)

    summary = summarize_by_hparams(runs_df, metrics_df, ["lr"])

    assert len(summary) == 1  # One group (lr=0.01)
    assert "mean" in summary.columns
    assert "std" in summary.columns
    assert summary["n_runs"][0] == 2


def test_remap_display_names_metrics():
    """Test remapping metric display names."""
    config = AnalysisConfig()

    metrics_df = pl.DataFrame(
        {
            "run_id": ["exp1", "exp1", "exp2"],
            "metric": ["train/loss", "val/acc", "train/loss"],
            "value": [0.5, 0.9, 0.4],
        }
    )

    remapped = remap_display_names(metrics_df, config, target="metric")

    assert remapped["metric"].to_list() == [
        "Training Loss",
        "Validation Accuracy",
        "Training Loss",
    ]


def test_remap_display_names_hparams():
    """Test remapping hyperparameter column names."""
    config = AnalysisConfig()

    runs_df = pl.DataFrame(
        {
            "run_id": ["exp1", "exp2"],
            "lr": [0.01, 0.02],
            "batch_size": [32, 64],
            "other_param": [1, 2],  # Not in mappings
        }
    )

    remapped = remap_display_names(runs_df, config, target="hparam")

    assert "Learning Rate" in remapped.columns
    assert "Batch Size" in remapped.columns
    assert "other_param" in remapped.columns  # Unchanged
    assert remapped["Learning Rate"].to_list() == [0.01, 0.02]
