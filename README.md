# dr_gen

A PyTorch-based deep learning training framework focused on generative models and computer vision tasks. Provides a complete pipeline for training, evaluating, and analyzing neural networks with extensive experiment management capabilities.

## Plotting and Analysis

The `dr_gen.analyze.plots` module provides a simplified, efficient plotting system for ML experiment analysis. It replaces complex abstractions with direct matplotlib usage while maintaining all essential functionality.

### Core Plotting Functions

#### Line Plots
```python
import dr_gen.analyze.plots as plots
import numpy as np

# Single curve
curve = np.random.randn(100).cumsum()
plots.plot_lines(curve, title="Training Loss", xlabel="Epoch", ylabel="Loss")

# Multiple curves with sampling
curves = [np.random.randn(100).cumsum() + i for i in range(10)]
plots.plot_lines(curves, sample=5, legend=True, title="Multiple Runs")
```

#### Histograms
```python
# Single distribution
values = np.random.normal(0, 1, 1000)
plots.plot_histogram(values, bins=50, title="Final Accuracies")

# Multiple distributions comparison
dist1 = np.random.normal(85, 5, 500)  # Model A accuracies
dist2 = np.random.normal(82, 3, 500)  # Model B accuracies
plots.plot_histogram([dist1, dist2], bins=30, legend=True, alpha=0.7)
```

#### Statistical Comparisons
```python
# CDF plots with Kolmogorov-Smirnov statistics
baseline_acc = np.random.normal(80, 5, 100)
improved_acc = np.random.normal(85, 4, 100)
plots.plot_cdf(baseline_acc, improved_acc, title="Model Comparison")
```

#### Training Splits
```python
# Train/validation/test curves with different line styles
train_curves = [np.random.randn(50).cumsum() for _ in range(5)]
val_curves = [np.random.randn(50).cumsum() + 2 for _ in range(5)]
split_data = [train_curves, val_curves]
plots.plot_splits(split_data, splits=["train", "val"], title="Learning Curves")
```

#### Summary Statistics
```python
# Mean curves with uncertainty bands
experiment_runs = [np.random.randn(100).cumsum() for _ in range(20)]

# Standard deviation bands
plots.plot_summary(experiment_runs, uncertainty="std", title="Experiment Summary")

# Standard error bands
plots.plot_summary(experiment_runs, uncertainty="sem", title="Mean ± SEM")

# Min-max bands
plots.plot_summary(experiment_runs, uncertainty="minmax", title="Range")
```

#### Grid Layouts
```python
# Grid of different experiments
experiment_groups = [
    [np.random.randn(50).cumsum() for _ in range(5)],  # Experiment 1
    [np.random.randn(50).cumsum() for _ in range(5)],  # Experiment 2
    [np.random.randn(50).cumsum() for _ in range(5)],  # Experiment 3
    [np.random.randn(50).cumsum() for _ in range(5)],  # Experiment 4
]

plots.plot_grid(plots.plot_lines, experiment_groups, 
                subplot_shape=(2, 2), suptitle="Hyperparameter Sweep")
```

### Advanced Features

#### Data Sampling
Large datasets are automatically sampled for performance:
```python
# Automatically samples from 1000 curves for readability
many_curves = [np.random.randn(100).cumsum() for _ in range(1000)]
plots.plot_lines(many_curves, sample=10)  # Shows only 10 random curves
```

#### Custom Styling
```python
# Set global defaults
plots.set_plot_defaults(figsize=(12, 8), linewidth=3, alpha=0.8)

# Or pass styling per plot
plots.plot_lines(curves, figsize=(10, 6), grid=False, 
                linewidth=2, alpha=0.6, legend=True)
```

#### Subplot Integration
```python
import matplotlib.pyplot as plt

# Use with existing matplotlib figures
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

plots.plot_lines(train_curves, ax=ax1, title="Training")
plots.plot_histogram(final_accuracies, ax=ax2, title="Final Results")

plt.tight_layout()
plt.show()
```

### Key Benefits

- **Simple API**: Direct matplotlib usage without complex abstractions
- **ML-focused**: Built-in support for training curves, uncertainty quantification, and statistical comparisons
- **Performance**: Intelligent sampling for large datasets
- **Flexibility**: Works with existing matplotlib workflows
- **Type-safe**: Full type hints for better development experience

### Migration from Old System

The new plotting system consolidates `plot_utils.py` and `common_plots.py` into a single, simplified module. Key changes:

- Replace `import dr_gen.analyze.common_plots as cp` with `import dr_gen.analyze.plots as plots`
- Use descriptive function names: `plots.plot_lines()` instead of `cp.line_plot()`
- Pass matplotlib parameters directly instead of using OmegaConf configs
- Leverage pandas/numpy for statistics instead of custom implementations
