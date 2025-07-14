"""Polars DataFrame builders for experiment analysis."""

import polars as pl

from dr_gen.analyze.schemas import AnalysisConfig, Run


def runs_to_dataframe(runs: list[Run]) -> pl.DataFrame:
    """Convert list of Run models to Polars DataFrame.

    Returns DataFrame with columns: run_id, hyperparameters, metadata
    """
    if not runs:
        return pl.DataFrame()

    data = []
    for run in runs:
        flat_hparams = run.hpms.flatten()
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


def query_metrics(
    metrics_df: pl.DataFrame,
    metric_filter: str | None = None,
    run_filter: list[str] | None = None,
) -> pl.DataFrame:
    """Query metrics DataFrame with optional filters.

    Args:
        metrics_df: Long-format metrics DataFrame
        metric_filter: Regex pattern to filter metric names
        run_filter: List of run_ids to include

    Returns:
        Filtered DataFrame
    """
    result = metrics_df

    if metric_filter:
        result = result.filter(pl.col("metric").str.contains(metric_filter))

    if run_filter:
        result = result.filter(pl.col("run_id").is_in(run_filter))

    return result


def summarize_by_hparams(
    runs_df: pl.DataFrame, metrics_df: pl.DataFrame, hparams: list[str]
) -> pl.DataFrame:
    """Summarize metrics grouped by hyperparameters.

    Returns DataFrame with mean/std/min/max for each metric.
    """
    if not hparams or runs_df.is_empty() or metrics_df.is_empty():
        return pl.DataFrame()

    # Join run info with metrics
    joined = metrics_df.join(runs_df, on="run_id")

    # Group and aggregate
    return joined.group_by([*hparams, "metric"]).agg(
        pl.col("value").mean().alias("mean"),
        pl.col("value").std().alias("std"),
        pl.col("value").min().alias("min"),
        pl.col("value").max().alias("max"),
        pl.col("run_id").n_unique().alias("n_runs"),
    )


def remap_display_names(
    df: pl.DataFrame, config: AnalysisConfig, target: str = "metric"
) -> pl.DataFrame:
    """Remap column values using display name mappings from config.

    Args:
        df: DataFrame to remap
        config: Analysis configuration with display mappings
        target: Which column to remap ('metric' or 'hparam')

    Returns:
        DataFrame with remapped values
    """
    if target == "metric" and "metric" in df.columns:
        mapping = config.metric_display_names
        return df.with_columns(
            pl.col("metric").replace(mapping, default=None).alias("metric")
        )

    # For hparams, remap column names
    if target == "hparam":
        rename_map = {
            k: v for k, v in config.hparam_display_names.items() if k in df.columns
        }
        return df.rename(rename_map)

    return df
