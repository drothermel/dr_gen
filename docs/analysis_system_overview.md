# ML Experiment Analysis System Overview

This document describes the goals and design philosophy behind the experiment analysis system in `src/dr_gen/analyze/`.

## Primary Goals

### 1. Experiment Organization & Tracking

The system automatically manages collections of machine learning experiments by:

- **Automated Log Parsing**: Loads and parses training logs from JSONL files across directory structures
- **Hyperparameter Extraction**: Extracts complete hyperparameter configurations from each training run
- **Metadata Collection**: Captures training time, logged messages, and run completion status
- **Error Handling**: Gracefully handles malformed, incomplete, or failed training logs with detailed error tracking
- **Run Grouping**: Organizes experiments by hyperparameter combinations for systematic comparison

### 2. Flexible Metric Analysis

The system provides comprehensive metric tracking capabilities:

- **Time Series Storage**: Stores training metrics as curves with flexible x-axes (epochs, steps, iterations, etc.)
- **Multi-Split Support**: Handles multiple data splits (train/validation/evaluation) within each experiment
- **Metric Querying**: Enables retrieval of specific metric values at particular points or complete training curves
- **Type Flexibility**: Supports arbitrary metric types (loss, accuracy, custom evaluation metrics)
- **Curve Validation**: Ensures metric completeness across expected training duration

### 3. Hyperparameter Sweep Management

The system intelligently manages large-scale hyperparameter experiments:

- **Automatic Sweep Detection**: Identifies which hyperparameters vary across experiment suites
- **Smart Grouping**: Groups runs by varying hyperparameters while excluding noise (paths, seeds, timestamps)
- **Display Remapping**: Provides user-friendly parameter names (`model.weights` → `Init`) and values (`None` → `random`)
- **Selective Filtering**: Enables filtering and selection of runs based on specific hyperparameter combinations
- **Sweep Summarization**: Generates tables showing parameter combinations and run counts

### 4. Statistical Analysis Support

The system facilitates rigorous experimental analysis:

- **Multi-Seed Aggregation**: Collects results across multiple random seeds for the same hyperparameter setting
- **Training Dynamics Comparison**: Supports comparison of learning curves across different configurations
- **Completeness Validation**: Built-in checks ensure metric completeness across expected training epochs
- **Robust Error Handling**: Tracks and reports runs that fail to complete properly
- **Statistical Preparation**: Organizes data for downstream statistical analysis and visualization

## Design Philosophy

### Hierarchical Data Organization

The system follows a clear hierarchical structure:

```
RunGroup → RunData → SplitMetrics → MetricCurves → MetricCurve
     ↓         ↓          ↓            ↓           ↓
Collections Individual  Train/Val   Multiple    Single
of runs    experiments  splits      metrics     curve
```

### Smart Importance Filtering

The **"important values"** concept in the `Hpm` class enables dynamic grouping based on which hyperparameters actually vary in an experiment suite. This automatically excludes irrelevant parameters like:
- File paths (`paths.*`)
- Random seeds (`seed`)
- Checkpoint settings (`write_checkpoint`)
- Runtime metadata

### Flexibility Over Simplicity

The system prioritizes:
- **Flexibility**: Multiple metric types, arbitrary x-axes, configurable remapping
- **Robustness**: Comprehensive error handling, validation, graceful degradation
- **Scalability**: Efficient handling of large experiment collections

This design choice reflects its target use case for serious research rather than casual experimentation.

## Target Research Workflow

This system is designed for **deep learning research workflows** where researchers:

1. **Run Large Sweeps**: Execute hyperparameter sweeps across multiple random seeds
2. **Compare Configurations**: Analyze training dynamics between different model/optimizer configurations  
3. **Generate Statistics**: Produce statistical summaries and publication-ready analysis
4. **Handle Failures**: Robustly manage incomplete or failed training runs in large experiment batches
5. **Iterate Quickly**: Rapidly filter and analyze subsets of experiments during research

## Key Components

- **`RunData`**: Individual experiment container with hyperparameters, metadata, and metrics
- **`RunGroup`**: Collection manager for multiple experiments with sweep analysis
- **`HpmGroup`**: Hyperparameter group tracker that identifies varying parameters
- **`SplitMetrics`**: Multi-split metric manager for train/validation/evaluation data
- **`MetricCurves`**: Time series container supporting multiple x-axes per metric

## Benefits

This architecture enables researchers to:
- Focus on scientific questions rather than data wrangling
- Systematically compare experimental results across large parameter spaces
- Generate reliable statistical summaries from multi-seed experiments
- Quickly identify and analyze successful configurations
- Maintain organized records of experimental history for reproducibility