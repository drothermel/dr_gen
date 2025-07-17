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

# %% Load Config
config_path = Path("../configs/").absolute()
with initialize_config_dir(config_dir=str(config_path), version_base="1.3"):
    cfg = compose(config_name="config", overrides=["paths=mac"])
OmegaConf.resolve(cfg)
#print(OmegaConf.to_yaml(cfg))

# %% Load all runs
exp_dir = Path(f"{cfg.paths.data}/loss_slope/exps_v1/experiments/test_sweep/")
print("Loaing runs from experiment directory:", exp_dir)
all_runs = load_runs_from_dir(exp_dir, pattern="*metrics.jsonl")

# %% Create Analysis Config

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
        'model.architecture',
        'model.width_mult',
        'optim.name',
        'optim.weight_decay',
        'optim.lr',
        'seed',
        'batch_size',
        'tag'
    ],
    grouping_exclude_hpms=['seed', 'run_id', 'tag'],
)
#display(analysis_cfg)

# %% Create ExperimentDB

db = ExperimentDB(
    config=analysis_cfg, lazy=False
)
db.load_experiments()
print(f"All Runs: {len(db.all_runs)}, Active Runs: {len(db.active_runs)}")



# %%
hpm_names, run_groups = db.active_runs_grouped_by_hpms()
print(f"Grouping hyperparameters: {hpm_names}")
# Print the count of runs in each group
print(f"\nNumber of groups: {len(run_groups)}")
print(f"Total runs across all groups: {sum(len(runs) for runs in run_groups.values())}")

# Show first few groups with their hyperparameter values
for i, (hpm_values, runs) in enumerate(run_groups.items()):
    if i >= 5:  # Only show first 5 groups
        print("...")
        break
    # Create a dict mapping hyperparameter names to their values for this group
    hpm_dict = db.group_key_to_dict(hpm_values, hpm_names)
    print(f"\nGroup {i+1}: {len(runs)} runs")
    print(f"  Hyperparameters: {hpm_dict}")

#%%



#%%
# Filter groups to specific hyperparameter values
all_group_keys = list(run_groups.keys())
filtered_groups = [
    (
        group_key, run_groups[group_key]
    ) for group_key in all_group_keys
    if (
        group_key[0] == 512 and # batch_size
        group_key[2] == 1.0 and # width_mult
        group_key[3] in [0.01, 0.003, 0.001] and #lr
        group_key[5] in [0.0001, 0.0003] # wd
    )
]

# Display filtered groups
for group_key, group_runs in filtered_groups:
    print(f"Group {group_key}: {len(group_runs)} runs")

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Collect train_loss and epoch for all runs in the first filtered group
cumulative_lr = []
lr = []
epoch_losses = defaultdict(list)
for run in filtered_groups[0][1]:  # Get runs from first filtered group
    epochs = run.metrics['epoch']
    train_losses = run.metrics['train_loss']
    cumulative_lr.append(
        run.metrics['lr'] if len(cumulative_lr) == 0 else cumulative_lr[-1] + run.metrics['lr']
    )
    lr.append(run.metrics['lr'])
    for e, l in zip(epochs, train_losses):
        epoch_losses[e].append(l)

# Compute mean train_loss for each epoch
epochs_sorted = sorted(epoch_losses.keys())
mean_losses = [np.mean(epoch_losses[e]) for e in epochs_sorted]
print(cumulative_lr)

# Plot mean train_loss vs epoch
plt.figure(figsize=(8,5))
plt.plot(epochs_sorted, mean_losses, marker='o')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Mean Train Loss')
plt.title('Mean Train Loss vs Epoch')
plt.grid(True)
plt.show()





# %% Use the new run_group_to_metric_dfs function to extract metrics and plot
# Extract metrics for the first filtered group
if filtered_groups:
    first_group_key, first_group_runs = filtered_groups[0]
    
    # Extract all relevant metrics
    metric_dfs = db.run_group_to_metric_dfs(
        first_group_runs, 
        ['train_loss', 'train_acc', 'val_loss', 'val_acc', 'epoch', 'lr']
    )
    
    # Get epochs from first column (all runs should have same epochs)
    epochs = metric_dfs['epoch'].iloc[:, 0].values
    
    # Plot epoch vs train_loss for all runs in the group
    plt.figure(figsize=(10, 6))
    
    # Plot individual runs with transparency
    for col in metric_dfs['train_loss'].columns:
        plt.plot(epochs, metric_dfs['train_loss'][col], alpha=0.3, linewidth=1)
    
    # Plot mean with thicker line
    mean_loss = metric_dfs['train_loss'].mean(axis=1)
    plt.plot(epochs, mean_loss, 'k-', linewidth=2, label='Mean')
    
    # Add standard deviation band
    std_loss = metric_dfs['train_loss'].std(axis=1)
    plt.fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss, 
                     alpha=0.2, color='gray', label='±1 std')
    
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.title(f'Training Loss for Group: {db.group_key_to_dict(first_group_key, hpm_names)}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    
    # Print summary statistics
    print(f"\nGroup hyperparameters: {db.group_key_to_dict(first_group_key, hpm_names)}")
    print(f"Number of runs: {len(first_group_runs)}")
    print(f"Final train loss: {mean_loss.iloc[-1]:.4f} ± {std_loss.iloc[-1]:.4f}")



# %% Select and Prep Data

db.active_runs_hpms[0]
for i, run_hpms in enumerate(db.active_runs_hpms):
    print(f"Run {i}: {run_hpms}")
    print(f"Run {i} metrics: {db.all_runs[i].metrics.keys()}")
    print("-"*100)
    if i > 2:
        break




















# %%

def get_data(batch_size, width_mult, lr_list=None, wd_list=None):
    if lr_list is None:
        lr_list = [0.01, 0.03, 0.1, 0.13]
    if wd_list is None:
        wd_list = [0.0001, 0.0003]
    all_data = prep_all_data(
        db, 
        ['train_loss', 'train_acc', 'val_loss', 'val_acc'],
        {'batch_size':batch_size, 'model.width_mult': width_mult},
    )
    no_outliers = all_data[
        all_data["lr"].isin(lr_list) & all_data["wd"].isin(wd_list)
    ]
    mean_no_outliers = no_outliers.groupby(['epoch', 'wd', 'lr']).mean(numeric_only=True).reset_index()
    std_no_outliers = no_outliers.groupby(['epoch', 'wd', 'lr']).std(numeric_only=True).reset_index()
    return mean_no_outliers, std_no_outliers

# %%

batch_size = 128
width_mult = 1.0
all_data = prep_all_data(
    db, 
    ['train_loss', 'train_acc', 'val_loss', 'val_acc'],
    {'batch_size':batch_size, 'model.width_mult': width_mult},
)
no_outliers = all_data[
    all_data["lr"].isin([0.01, 0.03, 0.1, 0.13]) & all_data["wd"].isin([0.0001, 0.0003])
    #all_data["lr"].isin([0.01, 0.03, 0.1])
    #all_data["wd"].isin([0.0001, 0.03, 0.1])
]
"""
no_outliers = all_data
"""
# Take mean and std over seeds for each (epoch, wd, lr)
mean_no_outliers = no_outliers.groupby(['epoch', 'wd', 'lr']).mean(numeric_only=True).reset_index()
std_no_outliers = no_outliers.groupby(['epoch', 'wd', 'lr']).std(numeric_only=True).reset_index()



# %% 

# %%

batch_size = 128
width_mult = 1.0
all_data = prep_all_data(
    db, 
    ['lr', 'train_loss', 'train_acc', 'val_loss', 'val_acc'],
    {'batch_size':batch_size, 'model.width_mult': width_mult},
)
all_data[(all_data.run_lr == 0.01) & (all_data.run_wd == 0.0003)].head(100)

#%%
no_outliers = all_data[
    all_data["lr"].isin([0.01, 0.03, 0.1, 0.13]) & all_data["wd"].isin([0.0001, 0.0003])
    #all_data["lr"].isin([0.01, 0.03, 0.1])
    #all_data["wd"].isin([0.0001, 0.03, 0.1])
]
bs128_1x.head()

# %%
bs = 128 # 10, ..., 128, 256, 512
wm = 0.5
bs_wm_data, bs_wm_data_std = get_data(bs, wm, lr_list=[0.01, 0.03, 0.1, 0.13], wd_list=[0.0001, 0.0003])

plot_training_metrics(
    bs_wm_data,
    yrange_loss=(-0.1, 2.5),
    yrange_acc=(-0.01, 1.01),
    figsize=(4.2*2, 4.2*2.5),
    xlog=False,
    ylog=False,
    title=f"Batch Size: {bs}, Width Mult: {wm}",
    leave_space_for_legend=0.05,
    df_std=bs_wm_data_std,
)

# %%
bs = 128 # 10, ..., 128, 256, 512
wm = 0.5
bs_wm_data, bs_wm_data_std = get_data(bs, wm, lr_list=[0.01, 0.03, 0.1, 0.13], wd_list=[0.0001, 0.0003])

plot_training_metrics(
    bs_wm_data,
    yrange_loss=(-0.1, 2.5),
    yrange_acc=(-0.01, 1.01),
    figsize=(4.2*2, 4.2*2.5),
    xlog=True,
    ylog=False,
    title=f"Batch Size: {bs}, Width Mult: {wm}",
    leave_space_for_legend=0.05,
    df_std=bs_wm_data_std,
)
# %%
bs = 256 # 10, ..., 128, 256, 512
wm = 1.0
bs_wm_data, bs_wm_data_std = get_data(bs, wm, lr_list=[0.01, 0.03, 0.1, 0.13], wd_list=[0.0001, 0.0003])

plot_training_metrics(
    bs_wm_data,
    yrange_loss=(-0.1, 2.5),
    yrange_acc=(-0.01, 1.01),
    figsize=(4.2*2, 4.2*2.5),
    xlog=False,
    ylog=False,
    title=f"Batch Size: {bs}, Width Mult: {wm}",
    leave_space_for_legend=0.05,
    df_std=bs_wm_data_std,
)


# %%
bs = 512 # 10, ..., 128, 256, 512
wm = 1.0
bs_wm_data, bs_wm_data_std = get_data(bs, wm, lr_list=[0.01, 0.03, 0.1], wd_list=[0.0001, 0.0003])

plot_training_metrics(
    bs_wm_data,
    yrange_loss=(-0.1, 2.5),
    yrange_acc=(-0.01, 1.01),
    figsize=(4.2*2, 4.2*2.5),
    xlog=False,
    ylog=False,
    title=f"Batch Size: {bs}, Width Mult: {wm}",
    leave_space_for_legend=0.05,
    df_std=bs_wm_data_std,
)

# %%
bs = 512 # 10, ..., 128, 256, 512
wm = 1.0
bs_wm_data, bs_wm_data_std = get_data(bs, wm, lr_list=[0.01, 0.03, 0.1], wd_list=[0.0001, 0.0003])

plot_training_metrics(
    bs_wm_data,
    yrange_loss=(-0.1, 2.5),
    yrange_acc=(-0.01, 1.01),
    figsize=(4.2*2, 4.2*2.5),
    xlog=True,
    ylog=False,
    title=f"Batch Size: {bs}, Width Mult: {wm}",
    leave_space_for_legend=0.05,
    df_std=bs_wm_data_std,
)

# %%

bs = 128 # 10, ..., 128, 256, 512
wm = 1.0
bs_wm_data, bs_wm_data_std = get_data(bs, wm, lr_list=[0.01, 0.03, 0.1], wd_list=[0.0001, 0.0003])

plot_training_metrics(
    bs_wm_data,
    yrange_loss=(-0.1, 2.5),
    yrange_acc=(-0.01, 1.01),
    figsize=(4.2*2, 4.2*2.5),
    xlog=True,
    ylog=False,
    title=f"Batch Size: {bs}, Width Mult: {wm}",
    leave_space_for_legend=0.05,
    df_std=bs_wm_data_std,
)
# %%

bs = 128 # 10, ..., 128, 256, 512
wm = 1.0
bs_wm_data, bs_wm_data_std = get_data(bs, wm, lr_list=[0.01, 0.03, 0.1], wd_list=[0.0001, 0.0003])

plot_training_metrics(
    bs_wm_data,
    yrange_loss=(0.01, 2.5),
    yrange_acc=(-0.01, 1.01),
    figsize=(4.2*4, 4.2*1.5),
    xlog=True,
    ylog=True,
    title=f"Batch Size: {bs}, Width Mult: {wm}",
    leave_space_for_legend=0.05,
    df_std=bs_wm_data_std,
)



















# %% START HERE!!!!
bs = 512 # 10, ..., 128, 256, 512
wm = 1.0
bs_wm_data, bs_wm_data_std = get_data(bs, wm, lr_list=[0.01, 0.03, 0.1], wd_list=[0.0001, 0.0003])

plot_training_metrics(
    bs_wm_data,
    yrange_train_loss=(0.0001, 2.5),
    yrange_val_loss=(0.5, 1.5),
    yrange_acc=(-0.01, 1.01),
    figsize=(5*4, 4.2*1.5),
    xlog=True,
    ylog=True,
    title=f"Power Law: log(CE) vs log(epochs)",#Batch Size: {bs}, Width Mult: {wm}",
    leave_space_for_legend=0.05,
    df_std=bs_wm_data_std,
    hbar2=20,
)

# %%
bs = 512 # 10, ..., 128, 256, 512
wm = 1.0
bs_wm_data, bs_wm_data_std = get_data(bs, wm, lr_list=[0.01, 0.03, 0.1], wd_list=[0.0001, 0.0003])

plot_training_metrics(
    bs_wm_data,
    yrange_train_loss=(0.0001, 2.5),
    yrange_val_loss=(0.5, 1.5),
    yrange_acc=(-0.01, 1.01),
    figsize=(5*4, 4.2*1.5),
    xlog=False,
    ylog=True,
    title="Exponential Decay: log(CE) vs epochs",#f"Batch Size: {bs}, Width Mult: {wm}",
    leave_space_for_legend=0.05,
    df_std=bs_wm_data_std,
    hbar2=20,
)

# %%
bs = 512 # 10, ..., 128, 256, 512
wm = 1.0
bs_wm_data, bs_wm_data_std = get_data(bs, wm, lr_list=[0.01, 0.03, 0.1], wd_list=[0.0001, 0.0003])

plot_training_metrics(
    bs_wm_data,
    yrange_train_loss=(0.0001, 2.5),
    yrange_val_loss=(0.5, 1.5),
    yrange_acc=(-0.01, 1.01),
    figsize=(5*4, 4.2*1.5),
    xlog=True,
    ylog=False,
    title="??? CE vs log(epochs)",#f"Batch Size: {bs}, Width Mult: {wm}",
    leave_space_for_legend=0.05,
    df_std=bs_wm_data_std,
    hbar2=20,
)
# %% .START
bs = 512 # 10, ..., 128, 256, 512
wm = 1.0
bs_wm_data, bs_wm_data_std = get_data(bs, wm, lr_list=[0.01, 0.03, 0.1], wd_list=[0.0001, 0.0003])

plot_training_metrics(
    bs_wm_data,
    yrange_train_loss=(-0.1, 2.5),
    yrange_val_loss=(0.5, 1.5),
    yrange_acc=(-0.01, 1.01),
    figsize=(5*4, 4.2*1.5),
    xlog=False,
    ylog=False,
    title="CE vs Epochs",#f"Batch Size: {bs}, Width Mult: {wm}",
    leave_space_for_legend=0.05,
    df_std=bs_wm_data_std,
    hbar2=20,
)
# %%




















































# %%
bs = 512 # 10, ..., 128, 256, 512
wm = 1.0
bs_wm_data, bs_wm_data_std = get_data(bs, wm, lr_list=[0.01, 0.03, 0.1], wd_list=[0.0001, 0.0003])

plot_training_metrics(
    bs_wm_data,
    yrange_loss=(0.01, 2.5),
    yrange_acc=(-0.01, 1.01),
    figsize=(5*4, 4.2*1.5),
    xlog=False,
    ylog=False,
    title=f"Batch Size: {bs}, Width Mult: {wm}",
    leave_space_for_legend=0.05,
    df_std=bs_wm_data_std,
    hbar2=20,
)

# %%
bs = 512 # 10, ..., 128, 256, 512
wm = 1.0
bs_wm_data, bs_wm_data_std = get_data(bs, wm, lr_list=[0.01, 0.03, 0.1], wd_list=[0.0001, 0.0003])

plot_training_metrics(
    bs_wm_data,
    yrange_loss=(0.01, 2.5),
    yrange_acc=(-0.01, 1.01),
    figsize=(5*4, 4.2*1.5),
    xlog=True,
    ylog=False,
    title=f"Batch Size: {bs}, Width Mult: {wm}",
    leave_space_for_legend=0.05,
    df_std=bs_wm_data_std,
    hbar2=20,
)






















# %%
