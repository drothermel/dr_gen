#%%
# type: ignore


%load_ext autoreload
%autoreload 2

from pathlib import Path
import pandas as pd
import polars as pl
import json
from typing import Any, Dict, Tuple, List
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from pprint import pprint

from torch._higher_order_ops.run_const_graph import run_const_graph_dispatch_mode
from dr_gen.analyze.parsing import load_runs_from_dir
from dr_gen.analyze.database import ExperimentDB
from dr_gen.analyze.schemas import AnalysisConfig, Hpms
from IPython.display import display
from dr_gen.analyze.visualization import prep_all_data, plot_training_metrics

# %% Function: Plot Training Metrics

def plot_training_metrics(df: pd.DataFrame, xlog: bool = False, ylog: bool = False, yrange_loss: tuple[float, float] = None, yrange_acc: tuple[float, float] = None) -> None:
    """
    Draw the *four* panels (train‑loss, val‑loss, train‑acc, val‑acc) with
    colour = learning‑rate and linestyle = weight‑decay.  The mapping is derived
    automatically from whatever unique values appear in the dataframe.

    Parameters
    ----------
    df : tidy dataframe with at least the columns produced by
         `make_mock_cifar_df`.
    """
    # ----- discover the hyper‑parameter palette / style -----------------
    lrs = sorted(df['lr'].unique())         # e.g. [0.01, 0.03, 0.1]
    wds = sorted(df['wd'].unique())         # e.g. [1e‑4, 3e‑4, 1e‑3]

    # colours – reuse tab10 but subsample in case there are more than 10 lrs
    cmap = plt.get_cmap('tab10')
    colour_map = {lr: cmap(i % 10) for i, lr in enumerate(lrs)}

    # linestyles – cycle through the classic trio, then fallback to dash‑dot etc.
    style_cycle = ['solid', 'dashed', 'dotted', 'dashdot', (0, (3, 1, 1, 1))]
    style_map = {wd: style_cycle[i % len(style_cycle)] for i, wd in enumerate(wds)}

    # ----- set‑up the 1×4 sub‑plots ------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2), sharex=True)

    panels = [('train_loss', 'Train Loss', ylog),
              ('val_loss',   'Val Loss',   ylog),
              ('train_acc',  'Train Acc',  False),
              ('val_acc',    'Val Acc',    False)]

    for (metric, title, met_ylog), ax in zip(panels, axes):
        for lr in lrs:
            for wd in wds:
                subset = df[(df.lr == lr) & (df.wd == wd)]
                ax.plot(subset['epoch'], subset[metric],
                        label=f'lr={lr}, wd={wd}',
                        color=colour_map[lr],
                        linestyle=style_map[wd],
                        linewidth=1.8)

        ax.set_title(title)
        ax.grid(alpha=.3, which='both', linestyle=':')
        if metric == 'train_loss' or metric == 'val_loss':
            ax.set_ylim(yrange_loss)
        elif metric == 'train_acc' or metric == 'val_acc':
            ax.set_ylim(yrange_acc)
        if xlog:
            ax.set_xscale('log')
            ax.set_xlabel('Epoch (log scale)')
        else:
            ax.set_xlabel('Epoch')
        if met_ylog:
            ax.set_yscale('log')
            ax.set_ylabel(f'{metric} (log scale)')
        else:
            ax.set_ylabel(f'{metric}')

    # -------------------  Legend (row-per-lr layout)  --------------------

    title = 'Learning Rate (color) × Weight Decay (style)'

    handles, labels = [], []

    for lr in lrs:
        # first column of the row: lr label (no visible line)
        handles.append(Line2D([], [], color='none', label=f'lr={lr}:'))
        labels.append(f'lr={lr}:')
        # remaining columns: one entry per wd, styled correctly
        for wd in wds:
            handles.append(Line2D([], [], color=colour_map[lr],
                                linestyle=style_map[wd], linewidth=2,
                                label=f'wd={wd}'))
            labels.append(f'wd={wd}')

    # ncol = (# wds + 1)   →  exactly one row per lr group
    ncol = len(wds) + 1

    leg = fig.legend(handles, labels,
                    ncol=ncol,
                    loc='lower center',
                    bbox_to_anchor=(0.5, -0.20),
                    frameon=False,
                    columnspacing=1.5,
                    handletextpad=0.6)

    # add a bold title and subtitle (two separate lines)
    leg.set_title(f'{title}', prop={'weight': 'bold', 'size': 'medium'})

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.32)   # leave room beneath the plots
    plt.show()


# %%




# %%
config_path = Path("../configs/").absolute()

with initialize_config_dir(config_dir=str(config_path), version_base="1.3"):
    cfg = compose(config_name="config", overrides=["paths=mac"])
OmegaConf.resolve(cfg)
print(OmegaConf.to_yaml(cfg))

# %%
exp_dir = Path(f"{cfg.paths.data}/loss_slope/exps_v1/experiments/test_sweep/")
print("Loaing runs from experiment directory:", exp_dir)
all_runs = load_runs_from_dir(exp_dir, pattern="*metrics.jsonl")
"""
print("Number of runs:", len(all_runs))
print("First run ID:", all_runs[0].run_id)
print("All metric names:", all_runs[0].metric_names())
pprint(f'best_train_loss: {all_runs[0].best_metric("train_loss"):0.3f}')
pprint(f'best_train_acc: {all_runs[0].best_metric("train_acc"):0.3f}')
pprint(f'best_val_loss: {all_runs[0].best_metric("val_loss"):0.3f}')
pprint(f'best_val_acc: {all_runs[0].best_metric("val_acc"):0.3f}')
print("Hyperparameters:")
pprint(all_runs[0].hpms.flatten())
"""

# %%

analysis_cfg = AnalysisConfig(
    experiment_dir=str(exp_dir),
    output_dir=f"{cfg.paths.root}/repos/dr_results/projects/deconCNN_v1",
    metric_display_names={
        'train_loss': 'Train Loss',
        'train_loss_bits': 'Train Loss (bits)',
        'train_acc': 'Train Accuracy',
        'lr': 'Learning Rate',
        'wd': 'Weight Decay',
        'global_step': 'Global Step',
        'val_loss': 'Validation Loss',
        'val_acc': 'Validation Accuracy',
    },
    hparam_display_names={
        'optim.lr': 'Learning Rate',
        'optim.weight_decay': 'Weight Decay',
        'optim.name': 'Optimizer',
        'batch_size': 'Batch Size',
        'epochs': 'Epochs',
        'lrsched.sched_type': 'LR Sched',
        'lrsched.warmup_epochs': 'Warmup Epochs',
        'model.architecture': 'Model Arch',
        'model.dropout_prob': 'Dropout Prob',
        'model.name': 'Model Name',
        'model.norm_type': 'Norm Type',
        'model.use_residual': 'Residual On?',
        'seed': 'Seed',
        'tag': 'Run Name',
        'train_transforms.rcc': 'RCC On?',
        'train_transforms.hflip': 'HFlip On?',
        'train_transforms.label_smoothing': 'Label Smoothing On?',
        'train_transforms.mixup': 'Mixup On?',
        'train_transforms.cutmix': 'Cutmix On?',
        'train_transforms.randaug': 'RandAug On?',
        'train_transforms.colorjitter': 'ColorJitter On?',
    },
    use_runs_filters={
        '50 epochs': lambda run: run.hpms._flat_dict['epochs'] == 50,
        'lrsched cosine': lambda run: run.hpms._flat_dict['lrsched.sched_type'] == 'cosine_annealing',
        'batchnorm': lambda run: run.hpms._flat_dict['model.norm_type'] == 'batchnorm',
        'no dropout': lambda run: run.hpms._flat_dict['model.dropout_prob'] == 0.0,
        'sgd momentum0.9': lambda run: run.hpms._flat_dict['optim.name'] == 'sgdm' and run.hpms._flat_dict['optim.momentum'] == 0.9 and run.hpms._flat_dict['optim.nesterov'] == False,
        'no residual': lambda run: run.hpms._flat_dict['model.use_residual'] == True,
        'no mixup': lambda run: run.hpms._flat_dict['train_transforms.mixup'] == False,
        'no cutmix': lambda run: run.hpms._flat_dict['train_transforms.cutmix'] == False,
        'no randaug': lambda run: run.hpms._flat_dict['train_transforms.randaug'] == False,
        'no colorjitter': lambda run: run.hpms._flat_dict['train_transforms.colorjitter'] == False,
        'no hflip': lambda run: run.hpms._flat_dict['train_transforms.hflip'] == True,
        'no label smoothing': lambda run: run.hpms._flat_dict['train_transforms.label_smoothing'] == 0.0,
        'no limit train batches': lambda run: run.hpms._flat_dict['limit_train_batches'] is None,
        'lrmin 0.0': lambda run: run.hpms._flat_dict['lrsched.lr_min'] == 0.0 and run.hpms._flat_dict['lrsched.warmup_start_lr'] == 0.0,
        'relu': lambda run: run.hpms._flat_dict['model.nonlinearity'] == 'relu',
        'he init': lambda run: run.hpms._flat_dict['model.init_method'] == 'he',
        '5 warmup epochs': lambda run: run.hpms._flat_dict['lrsched.warmup_epochs'] == 5,
        'completed run': lambda run: run.hpms._flat_dict['status'] == 'completed',

    },
    main_hpms=[
        'run_id',
        #'machine.root_dir',
        #'machine.device',
        #'paths.data',
        #'paths.logs',
        #'paths.my_data',
        #'paths.my_logs',
        #'paths.run_dir',
        #'paths.dataset_cache_root',
        #'paths.agg_results',
        #'model.source',
        'model.architecture',
        #'model.nonlinearity',
        #'model.norm_type',
        #'model.dropout_prob',
        #'model.use_residual',
        #'model.init_method',
        #'model.use_imagenet_head',
        'model.width_mult',
        'optim.name',
        'optim.weight_decay',
        'optim.lr',
        #'optim.momentum',
        #'optim.nesterov',
        #'lrsched.source',
        #'lrsched.sched_type',
        #'lrsched.lr_min',
        #'lrsched.warmup_epochs',
        #'lrsched.warmup_start_lr',
        #'train_transforms.rcc',
        #'train_transforms.hflip',
        #'train_transforms.randaug',
        #'train_transforms.colorjitter',
        #'train_transforms.mixup',
        #'train_transforms.cutmix',
        #'train_transforms.normalize_mean',
        #'train_transforms.normalize_std',
        #'train_transforms.rcc_scale_min',
        #'train_transforms.rcc_init_size',
        #'train_transforms.colorjitter_brightness',
        #'train_transforms.colorjitter_contrast',
        #'train_transforms.colorjitter_saturation',
        #'train_transforms.colorjitter_hue',
        #'train_transforms.label_smoothing',
        #'train_transforms.mixup_alpha',
        #'train_transforms.cutmix_alpha',
        #'eval_transforms.rcc',
        #'eval_transforms.hflip',
        #'eval_transforms.randaug',
        #'eval_transforms.colorjitter',
        #'eval_transforms.mixup',
        #'eval_transforms.cutmix',
        #'eval_transforms.normalize_mean',
        #'eval_transforms.normalize_std',
        #'eval_transforms.rcc_scale_min',
        #'eval_transforms.rcc_init_size',
        #'eval_transforms.colorjitter_brightness',
        #'eval_transforms.colorjitter_contrast',
        #'eval_transforms.colorjitter_saturation',
        #'eval_transforms.colorjitter_hue',
        #'eval_transforms.label_smoothing',
        #'eval_transforms.mixup_alpha',
        #'eval_transforms.cutmix_alpha',
        #'data.name',
        #'data.num_workers',
        #'data.download',
        #'data.data_split_seed',
        #'data.train_val_split_factor',
        #'proj_dir_name',
        #'load_checkpoint',
        #'enable_checkpointing',
        #'log_every',
        #'limit_train_batches',
        #'val_check_interval',
        'seed',
        #'train',
        #'eval',
        #'epochs',
        'batch_size',
        #'loss',
        #'clip_grad_norm',
        #'_target_',
        #'status',
        'tag'
    ],
)
#display(analysis_cfg)

# %%

db = ExperimentDB(
    config=analysis_cfg, lazy=False
)
print(db)
print(db.base_path)
db.load_experiments()
print(f"Number of runs: {len(db.all_runs)}")
print(f"Number of active runs: {len(db.active_runs)}")






# %%


# %%

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
    # Remove 'run_id' from important_hpms to avoid duplicate
    hpms_to_select = [h for h in db.important_hpms if h != 'run_id']
    joined_df = metrics_df.join(
        filtered_runs_df.select(['run_id'] + hpms_to_select),
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





# %%
def prepare_data_for_plotting(
    grouped_results: Dict[tuple, pd.DataFrame],
    metric_name: str,
    db: ExperimentDB,
    aggregate_seeds: bool = True
) -> pd.DataFrame:
    """
    Prepare the grouped results for use with plot_training_metrics.
    
    Args:
        grouped_results: Output from group_metric_by_hpms_v2
        metric_name: Name of the metric being plotted
        db: ExperimentDB instance (to access important_hpms)
        aggregate_seeds: If True, average across seeds; if False, include seed as a column
    
    Returns:
        DataFrame formatted for plot_training_metrics with columns:
        epoch, lr, wd, [metric_name], and optionally seed
    """
    all_dfs = []
    
    for group_key, df in grouped_results.items():
        # Extract lr and wd from the group key
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


# %%

def prep_all_data(
    db: ExperimentDB,
    metrics: List[str],
    hpm_filters: Dict[str, Any],
    aggregate_seeds: bool = True,
) -> None:
    # Prepare data for each metric
    metric_dfs = {}
    for metric in metrics:
        grouped = group_metric_by_hpms_v2(db, metric, **hpm_filters)
        if not grouped:
            print(f"No data found for metric: {metric}")
            continue
        metric_dfs[metric] = prepare_data_for_plotting(grouped, metric, db, aggregate_seeds)
    
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
    
    return combined_df


# %%

all_data = prep_all_data(
    db, 
    ['train_loss', 'train_acc', 'val_loss', 'val_acc'],
    {'batch_size':128, 'model.width_mult': 1.0},
)
all_data.head()

# %%

# The original code tries to use polars `.filter` on a pandas DataFrame, which will not work.
# Instead, use pandas boolean indexing:
no_outliers = all_data[
    #all_data["lr"].isin([0.01, 0.03, 0.1]) & all_data["wd"].isin([0.0001, 0.0003, 0.001])
    all_data["lr"].isin([0.01, 0.03, 0.1]))
]
# Take mean over seeds for each (epoch) (averaging across all lr and wd)
mean_no_outliers = no_outliers.groupby(['epoch', 'wd', 'lr']).mean(numeric_only=True).reset_index()
print(mean_no_outliers.head())



# %%
plot_training_metrics(mean_no_outliers, yrange_loss=(-0.1, 2.5), yrange_acc=(0, 1.1))







# %%





























# %%
