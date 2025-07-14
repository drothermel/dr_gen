#%%
# type: ignore
"""
Integrated plotting pipeline that combines group_metric_by_hpms_v2 with plotting functions.
This script demonstrates the full workflow from loading experiments to creating plots.
"""

%load_ext autoreload
%autoreload 2

from pathlib import Path
import pandas as pd
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import Any, Dict, Tuple, List
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from pprint import pprint

from dr_gen.analyze.parsing import load_runs_from_dir
from dr_gen.analyze.database import ExperimentDB
from dr_gen.analyze.schemas import AnalysisConfig
from IPython.display import display

#%%
# Initialize configuration
config_path = Path("../configs/").absolute()

with initialize_config_dir(config_dir=str(config_path), version_base="1.3"):
    cfg = compose(config_name="config", overrides=["paths=mac"])
OmegaConf.resolve(cfg)
print("Configuration loaded successfully")

#%%
# Set up experiment directory and analysis config
exp_dir = Path(f"{cfg.paths.data}/loss_slope/exps_v1/experiments/test_sweep/")
print("Loading runs from experiment directory:", exp_dir)

analysis_cfg = AnalysisConfig(
    experiment_dir=str(exp_dir),
    output_dir=f"{cfg.paths.root}/repos/dr_results/projects/deconCNN_v1",
    metric_display_names={
        'train_loss': 'Train Loss',
        'train_acc': 'Train Accuracy',
        'val_loss': 'Validation Loss',
        'val_acc': 'Validation Accuracy',
    },
    main_hpms=[
        'run_id',
        'model.architecture',
        'model.width_mult',
        'optim.name',
        'optim.weight_decay',
        'optim.lr',
        'seed',
        'batch_size',
        'tag'
    ],
)

#%%
# Load experiments into database
db = ExperimentDB(config=analysis_cfg, lazy=False)
db.load_experiments()
print(f"Number of runs: {len(db.all_runs)}")
print(f"Number of active runs: {len(db.active_runs)}")

#%%
def group_metric_by_hpms_v2(
    db: ExperimentDB,
    metric: str,
    **hpm_filters: Any
) -> Dict[tuple, pd.DataFrame]:
    """
    Group metric time series by main_hpms (excluding 'seed'), filtered by hpm_filters.
    
    Args:
        db: The ExperimentDB instance.
        metric: The metric name (e.g., 'train_loss', 'val_acc').
        **hpm_filters: Keyword arguments for filtering (e.g., batch_size=128, model__width_mult=1.0).
                      Note: Use double underscore for nested keys (e.g., model__width_mult for model.width_mult)
    
    Returns:
        Dict mapping from tuple of hpm values (excluding 'seed' and 'run_id') to DataFrame with columns:
        - epoch: timestep
        - seed: seed value
        - [metric]: the metric value
    """
    # Get the filtered runs based on hpm filters
    filtered_runs_df = db._runs_df
    
    # Apply filters - handle both dot notation and double underscore notation
    for k, v in hpm_filters.items():
        # Convert double underscore to dot notation if needed
        col_name = k.replace('__', '.')
        
        # Check if column exists
        if col_name in filtered_runs_df.columns:
            filtered_runs_df = filtered_runs_df.filter(pl.col(col_name) == v)
        else:
            print(f"Warning: Column '{col_name}' not found in runs dataframe")
    
    # Get run_ids from filtered runs
    run_ids = filtered_runs_df['run_id'].to_list()
    
    if not run_ids:
        print("No runs found matching the filters")
        return {}
    
    # Get metrics for these runs
    metrics_df = db._metrics_df.filter(
        (pl.col('run_id').is_in(run_ids)) & 
        (pl.col('metric') == metric)
    )
    
    # Join metrics with run info to get hpms
    joined_df = metrics_df.join(
        filtered_runs_df.select(['run_id'] + db.important_hpms),
        on='run_id',
        how='left'
    )
    
    # Group by all important hpms except 'seed' and 'run_id'
    group_keys = [h for h in db.important_hpms if h not in ['seed', 'run_id']]
    
    # Convert to pandas for easier grouping
    joined_pd = joined_df.to_pandas()
    
    # Create result dictionary
    result = {}
    
    # Group by the hpm keys
    for group_values, group_df in joined_pd.groupby(group_keys):
        # Create a clean dataframe with just epoch, seed, and metric value
        metric_df = group_df[['epoch', 'seed', 'value']].copy()
        metric_df.rename(columns={'value': metric}, inplace=True)
        metric_df = metric_df.sort_values(['seed', 'epoch'])
        
        # Use tuple of group values as key
        if isinstance(group_values, tuple):
            key = group_values
        else:
            key = (group_values,)
            
        result[key] = metric_df
    
    return result

#%%
def prepare_data_for_plotting(
    grouped_results: Dict[tuple, pd.DataFrame],
    metric_name: str,
    aggregate_seeds: bool = True
) -> pd.DataFrame:
    """
    Prepare the grouped results for use with plot_training_metrics.
    
    Args:
        grouped_results: Output from group_metric_by_hpms_v2
        metric_name: Name of the metric being plotted
        aggregate_seeds: If True, average across seeds; if False, include seed as a column
    
    Returns:
        DataFrame formatted for plot_training_metrics with columns:
        epoch, lr, wd, [metric_name], and optionally seed
    """
    all_dfs = []
    
    for group_key, df in grouped_results.items():
        # Extract lr and wd from the group key
        # The group key order follows db.important_hpms (excluding seed and run_id)
        # Based on the main_hpms list: model.architecture, model.width_mult, optim.name, 
        # optim.weight_decay, optim.lr, batch_size, tag
        
        # Find indices for lr and wd in the group key
        group_keys = [h for h in db.important_hpms if h not in ['seed', 'run_id']]
        lr_idx = group_keys.index('optim.lr')
        wd_idx = group_keys.index('optim.weight_decay')
        
        lr = group_key[lr_idx]
        wd = group_key[wd_idx]
        
        # Add lr and wd columns
        df = df.copy()
        df['lr'] = lr
        df['wd'] = wd
        
        if aggregate_seeds:
            # Average across seeds
            agg_df = df.groupby(['epoch', 'lr', 'wd'])[metric_name].mean().reset_index()
            all_dfs.append(agg_df)
        else:
            all_dfs.append(df)
    
    # Combine all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)
    return combined_df

#%%
def plot_training_metrics_multi(
    db: ExperimentDB,
    metrics: List[str],
    hpm_filters: Dict[str, Any],
    aggregate_seeds: bool = True,
    figsize: Tuple[float, float] = (17, 4.2)
) -> None:
    """
    Plot multiple metrics using the group_metric_by_hpms_v2 function.
    
    Args:
        db: ExperimentDB instance
        metrics: List of metric names to plot
        hpm_filters: Filters to apply when selecting runs
        aggregate_seeds: Whether to average across seeds
        figsize: Figure size tuple
    """
    # Prepare data for each metric
    metric_dfs = {}
    for metric in metrics:
        grouped = group_metric_by_hpms_v2(db, metric, **hpm_filters)
        if not grouped:
            print(f"No data found for metric: {metric}")
            continue
        metric_dfs[metric] = prepare_data_for_plotting(grouped, metric, aggregate_seeds)
    
    if not metric_dfs:
        print("No data to plot")
        return
    
    # Combine all metric dataframes
    # We need to merge them on epoch, lr, wd
    combined_df = None
    for metric, df in metric_dfs.items():
        if combined_df is None:
            combined_df = df
        else:
            # Merge on common columns
            merge_cols = ['epoch', 'lr', 'wd']
            if 'seed' in df.columns and 'seed' in combined_df.columns:
                merge_cols.append('seed')
            combined_df = combined_df.merge(df, on=merge_cols, how='outer')
    
    # Now use the plotting function
    plot_training_metrics(combined_df)

#%%
def plot_training_metrics(df: pd.DataFrame) -> None:
    """
    Adapted from plotting_fxns.py to work with our data structure.
    """
    # Discover the hyper-parameter palette / style
    lrs = sorted(df['lr'].unique())
    wds = sorted(df['wd'].unique())
    
    # Colors for learning rates
    cmap = plt.get_cmap('tab10')
    colour_map = {lr: cmap(i % 10) for i, lr in enumerate(lrs)}
    
    # Linestyles for weight decay
    style_cycle = ['solid', 'dashed', 'dotted', 'dashdot', (0, (3, 1, 1, 1))]
    style_map = {wd: style_cycle[i % len(style_cycle)] for i, wd in enumerate(wds)}
    
    # Determine which metrics are available
    available_metrics = []
    metric_configs = []
    
    if 'train_loss' in df.columns:
        available_metrics.append('train_loss')
        metric_configs.append(('train_loss', 'Train Loss', True))
    if 'val_loss' in df.columns:
        available_metrics.append('val_loss')
        metric_configs.append(('val_loss', 'Val Loss', True))
    if 'train_acc' in df.columns:
        available_metrics.append('train_acc')
        metric_configs.append(('train_acc', 'Train Acc', False))
    if 'val_acc' in df.columns:
        available_metrics.append('val_acc')
        metric_configs.append(('val_acc', 'Val Acc', False))
    
    # Set up subplots
    n_panels = len(metric_configs)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.25 * n_panels, 4.2), sharex=True)
    
    if n_panels == 1:
        axes = [axes]
    
    for (metric, title, ylog), ax in zip(metric_configs, axes):
        for lr in lrs:
            for wd in wds:
                subset = df[(df.lr == lr) & (df.wd == wd)]
                if not subset.empty:
                    ax.plot(subset['epoch'], subset[metric],
                            label=f'lr={lr}, wd={wd}',
                            color=colour_map[lr],
                            linestyle=style_map[wd],
                            linewidth=1.8)
        
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.grid(alpha=.3, which='both', linestyle=':')
        
        if ylog:
            ax.set_yscale('log')
            ax.set_ylabel('Loss (log scale)')
        else:
            ax.set_ylabel('Accuracy')
    
    # Legend
    title = 'Learning Rate (color) × Weight Decay (style)'
    
    handles, labels = [], []
    
    for lr in lrs:
        # First column of the row: lr label
        handles.append(Line2D([], [], color='none', label=f'lr={lr}:'))
        labels.append(f'lr={lr}:')
        # Remaining columns: one entry per wd
        for wd in wds:
            handles.append(Line2D([], [], color=colour_map[lr],
                                linestyle=style_map[wd], linewidth=2,
                                label=f'wd={wd}'))
            labels.append(f'wd={wd}')
    
    ncol = len(wds) + 1
    
    leg = fig.legend(handles, labels,
                    ncol=ncol,
                    loc='lower center',
                    bbox_to_anchor=(0.5, -0.10),
                    frameon=False,
                    columnspacing=1.5,
                    handletextpad=0.6)
    
    leg.set_title(f'{title}', prop={'weight': 'bold', 'size': 'medium'})
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.32)
    plt.show()

#%%
# Example 1: Plot train and val loss for specific hyperparameters
print("Example 1: Plotting train and val loss for batch_size=128, width_mult=1.0")
plot_training_metrics_multi(
    db, 
    metrics=['train_loss', 'val_loss'],
    hpm_filters={'batch_size': 128, 'model__width_mult': 1.0},
    aggregate_seeds=True
)

#%%
# Example 2: Plot all four metrics (train/val loss/acc)
print("\nExample 2: Plotting all metrics for batch_size=128, width_mult=1.0")
plot_training_metrics_multi(
    db,
    metrics=['train_loss', 'val_loss', 'train_acc', 'val_acc'],
    hpm_filters={'batch_size': 128, 'model__width_mult': 1.0},
    aggregate_seeds=True
)

#%%
# Example 3: Filter by specific architecture
print("\nExample 3: Plotting metrics for resnet18 architecture")
plot_training_metrics_multi(
    db,
    metrics=['train_loss', 'val_loss'],
    hpm_filters={'model__architecture': 'resnet18'},
    aggregate_seeds=True
)

#%%
# Debug: Let's see what hyperparameter combinations we have
print("\nAvailable hyperparameter combinations:")
unique_hpms = db.active_runs_df.select([
    'model.architecture', 'model.width_mult', 'optim.lr', 
    'optim.weight_decay', 'batch_size'
]).unique()
display(unique_hpms.to_pandas())

#%%
# Test the group_metric_by_hpms_v2 function directly
print("\nTesting group_metric_by_hpms_v2 directly:")
test_results = group_metric_by_hpms_v2(
    db, 
    'train_loss', 
    batch_size=128, 
    model__width_mult=1.0
)
print(f"Number of groups found: {len(test_results)}")
if test_results:
    first_key = list(test_results.keys())[0]
    print(f"First group key: {first_key}")
    print(f"First group data shape: {test_results[first_key].shape}")
    
    # Prepare data for plotting
    plot_df = prepare_data_for_plotting(test_results, 'train_loss', aggregate_seeds=True)
    print(f"\nPrepared data shape: {plot_df.shape}")
    print(f"Unique lr values: {sorted(plot_df['lr'].unique())}")
    print(f"Unique wd values: {sorted(plot_df['wd'].unique())}")
    display(plot_df.head())

# %%