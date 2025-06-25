# Experiment Analysis System Overview

## Introduction

The experiment analysis system provides a modern, efficient framework for analyzing deep learning experiments. Built on Pydantic v2 for data validation and Polars for high-performance data manipulation, it replaces complex class hierarchies with a clean, functional approach.

## Architecture

### Core Components

1. **Data Models** (`models.py`)
   - `Hyperparameters`: Flexible model with dot-notation flattening
   - `Run`: Complete experiment run with metrics and metadata
   - `AnalysisConfig`: Configuration with environment variable support

2. **Data Loading** (`parsers.py`)
   - JSONL parsing with error collection
   - Batch loading from directories
   - Legacy format conversion utilities

3. **DataFrame Operations** (`dataframes.py`)
   - Polars-based DataFrame builders
   - Hyperparameter variation detection
   - Metric querying and aggregation

4. **High-Level API** (`experiment_db.py`)
   - Unified interface for analysis
   - Lazy evaluation support
   - Memory-efficient streaming

## Usage Guide

### Basic Setup

```python
from pathlib import Path
from dr_gen.analyze.experiment_db import ExperimentDB
from dr_gen.analyze.models import AnalysisConfig

# Configure analysis
config = AnalysisConfig(
    experiment_dir="./experiments",
    metric_display_names={
        "train/loss": "Training Loss",
        "val/acc": "Validation Accuracy",
    }
)

# Create database
db = ExperimentDB(config=config, base_path=Path(config.experiment_dir))
db.load_experiments()
```

### Querying Metrics

```python
# Get all metrics
all_metrics = db.query_metrics()

# Filter by metric name
train_metrics = db.query_metrics(metric_filter="train")

# Filter by specific runs
specific_runs = db.query_metrics(run_filter=["run_001", "run_002"])
```

### Analyzing Hyperparameters

```python
# Find which hyperparameters vary across runs
from dr_gen.analyze.dataframes import find_varying_hparams
varying = find_varying_hparams(runs_df)

# Summarize metrics by hyperparameters
summary = db.summarize_metrics(["lr", "batch_size"])
```

### Working with Large Datasets

```python
# Enable lazy evaluation
db = ExperimentDB(config, base_path, lazy=True)

# Build complex queries without loading data
lazy_frame = db.lazy_query()
filtered = lazy_frame.filter(pl.col("metric") == "val/acc")

# Stream results
results = db.stream_metrics()
```

## Migration from Legacy System

### Converting Old Data

The system includes utilities for migrating from the previous analysis system:

```python
from dr_gen.analyze.parsers import convert_legacy_format

# Convert old format
legacy_data = load_old_experiment()  # Your old loading code
converted = convert_legacy_format(legacy_data)

# Create new Run model
run = Run(**converted)
```

### Key Differences

1. **Data Models**: Pydantic models instead of custom classes
2. **DataFrames**: Polars instead of Pandas for better performance
3. **Configuration**: Environment variable support via Pydantic Settings
4. **API**: Functional approach with immutable data structures

## Configuration

### Environment Variables

The system supports configuration via environment variables:

```bash
export ANALYSIS_EXPERIMENT_DIR=/path/to/experiments
export ANALYSIS_OUTPUT_DIR=/path/to/output
```

### Config File

Create a `.env` file in your project root:

```
ANALYSIS_EXPERIMENT_DIR=./experiments
ANALYSIS_OUTPUT_DIR=./analysis_output
```

### Programmatic Configuration

```python
config = AnalysisConfig(
    experiment_dir="./custom/path",
    output_dir="./custom/output",
    metric_display_names={
        "custom/metric": "Custom Metric Name"
    }
)
```

## Performance Considerations

1. **Lazy Evaluation**: Use `lazy=True` for large datasets
2. **Streaming**: Use `stream_metrics()` for memory-constrained environments
3. **Categorical Data**: Hyperparameters are automatically converted to categoricals
4. **Parallel Processing**: Polars uses all available cores by default

## Examples

See `examples/analyze_experiments.py` for complete working examples including:
- Basic analysis workflows
- Lazy evaluation for large datasets
- Legacy data migration
- Custom metric aggregations

## API Reference

### Models

- `Hyperparameters`: Flexible hyperparameter storage with flattening
- `Run`: Complete experiment run with computed properties
- `AnalysisConfig`: Configuration with validation

### Functions

- `parse_jsonl_file()`: Parse JSONL with error handling
- `load_runs_from_dir()`: Batch load experiments
- `runs_to_dataframe()`: Convert runs to wide-format DataFrame
- `runs_to_metrics_df()`: Convert to long-format metrics DataFrame
- `find_varying_hparams()`: Detect varying hyperparameters
- `query_metrics()`: Filter metrics with patterns
- `summarize_by_hparams()`: Aggregate metrics by groups

### ExperimentDB Methods

- `load_experiments()`: Load all experiments from directory
- `query_metrics()`: Query with filters
- `summarize_metrics()`: Get summary statistics
- `lazy_query()`: Get lazy frame for complex queries
- `stream_metrics()`: Memory-efficient streaming