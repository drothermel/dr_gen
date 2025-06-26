"""Example usage of the experiment analysis system."""

from pathlib import Path

import polars as pl

from dr_gen.analyze.database import ExperimentDB
from dr_gen.analyze.schemas import AnalysisConfig


def basic_analysis_example():
    """Basic example of loading and analyzing experiments."""
    # Configure analysis
    config = AnalysisConfig(
        experiment_dir="./experiments",
        metric_display_names={
            "train/loss": "Training Loss",
            "val/acc": "Validation Accuracy",
        },
    )
    
    # Create database interface
    db = ExperimentDB(config=config, base_path=Path(config.experiment_dir))
    
    # Load experiments
    db.load_experiments()
    
    # Query all metrics
    all_metrics = db.query_metrics()
    print(f"Total metric entries: {len(all_metrics)}")
    
    # Filter for validation metrics
    val_metrics = db.query_metrics(metric_filter="val")
    print(f"Validation metrics: {len(val_metrics)}")
    
    # Summarize by learning rate
    summary = db.summarize_metrics(["lr"])
    print("\nSummary by learning rate:")
    print(summary)


def lazy_evaluation_example():
    """Example using lazy evaluation for large datasets."""
    config = AnalysisConfig()
    db = ExperimentDB(
        config=config, 
        base_path=Path("./large_experiments"),
        lazy=True  # Enable lazy loading
    )
    
    # Get lazy frame without loading all data
    lazy_frame = db.lazy_query()
    
    # Build complex query
    filtered = (
        lazy_frame
        .filter(pl.col("metric") == "val/acc")
        .filter(pl.col("value") > 0.9)
        .select(["run_id", "epoch", "value"])
    )
    
    # Execute query (data loaded here)
    high_accuracy_runs = filtered.collect()
    print(f"High accuracy runs: {len(high_accuracy_runs)}")


def legacy_migration_example():
    """Example of migrating legacy experiment data."""
    from dr_gen.analyze.parsing import convert_legacy_format
    from dr_gen.analyze.schemas import Run, Hyperparameters
    
    # Load legacy data (e.g., from pickle file)
    legacy_data = {
        "name": "old_experiment_001",
        "config": {"learning_rate": 0.001, "batch_size": 64},
        "history": [
            {"epoch": 0, "loss": 2.3, "accuracy": 0.1},
            {"epoch": 1, "loss": 1.8, "accuracy": 0.3},
        ],
    }
    
    # Convert to new format
    converted = convert_legacy_format(legacy_data)
    
    # Create Run model
    run = Run(
        run_id=converted["run_id"],
        hyperparameters=Hyperparameters(**converted["hyperparameters"]),
        metrics=converted["metrics"],
        metadata=converted["metadata"],
    )
    
    print(f"Migrated run: {run.run_id}")
    print(f"Best loss: {run.best_train_loss}")


if __name__ == "__main__":
    print("=== Basic Analysis Example ===")
    basic_analysis_example()
    
    print("\n=== Lazy Evaluation Example ===")
    # lazy_evaluation_example()  # Uncomment with real data
    
    print("\n=== Legacy Migration Example ===")
    legacy_migration_example()