"""Polars DataFrame builders for experiment analysis."""

import polars as pl

from dr_gen.analyze.models import Run


def runs_to_dataframe(runs: list[Run]) -> pl.DataFrame:
    """Convert list of Run models to Polars DataFrame.

    Returns DataFrame with columns: run_id, hyperparameters, metadata
    """
    if not runs:
        return pl.DataFrame()

    data = []
    for run in runs:
        flat_hparams = run.hyperparameters.flatten()
        data.append(
            {
                "run_id": run.run_id,
                **flat_hparams,
                **{f"metadata.{k}": v for k, v in run.metadata.items()},
            }
        )

    return pl.DataFrame(data)


def runs_to_metrics_df(runs: list[Run]) -> pl.DataFrame:
    """Convert runs to long-format metrics DataFrame.

    Returns DataFrame with columns: run_id, metric, epoch, value
    """
    if not runs:
        return pl.DataFrame()

    records = []
    for run in runs:
        for metric_name, values in run.metrics.items():
            for epoch, value in enumerate(values):
                records.append(
                    {
                        "run_id": run.run_id,
                        "metric": metric_name,
                        "epoch": epoch,
                        "value": value,
                    }
                )

    return pl.DataFrame(records)


def find_varying_hparams(df: pl.DataFrame) -> list[str]:
    """Find hyperparameter columns that vary across runs.

    Returns list of column names that have more than one unique value.
    """
    if df.is_empty():
        return []

    # Exclude non-hyperparameter columns
    exclude = {"run_id", "metadata"}
    hparam_cols = [
        c for c in df.columns if c not in exclude and not c.startswith("metadata.")
    ]

    varying = [col for col in hparam_cols if df[col].n_unique() > 1]
    return sorted(varying)


def group_by_hparams(df: pl.DataFrame, hparams: list[str]) -> pl.DataFrame:
    """Group runs by specified hyperparameters.

    Returns DataFrame with grouping columns and run_id lists.
    """
    if not hparams or df.is_empty():
        return df

    return df.group_by(hparams).agg(pl.col("run_id").alias("run_ids"))
