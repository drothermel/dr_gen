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
)
#display(analysis_cfg)

# %% Create ExperimentDB

db = ExperimentDB(
    config=analysis_cfg, lazy=False
)
db.load_experiments()
print(f"All Runs: {len(db.all_runs)}, Active Runs: {len(db.active_runs)}")

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
# Take mean over seeds for each (epoch) (averaging across all lr and wd)
mean_no_outliers = no_outliers.groupby(['epoch', 'wd', 'lr']).mean(numeric_only=True).reset_index()


# %%
plot_training_metrics(
    mean_no_outliers,
    yrange_loss=(-0.1, 2.5),
    yrange_acc=(-0.01, 1.01),
    figsize=(4.2*2, 4.2*2.5),
    xlog=False,
    ylog=False,
    title=f"Batch Size: {batch_size}, Width Mult: {width_mult}",
    leave_space_for_legend=0.05,
)

# %%
plot_training_metrics(
    mean_no_outliers,
    yrange_loss=(-0.1, 2.5),
    yrange_acc=(-0.01, 1.01),
    figsize=(4.2*2, 4.2*2.5),
    xlog=True,
    ylog=False,
)
# %%


plot_training_metrics(
    mean_no_outliers,
    yrange_loss=(0.01, 2.5),
    #yrange_acc=(0, 1.1),
    figsize=(4.2*4, 4.2*1.5),
    xlog=True,
    ylog=True,
)

# %%


plot_training_metrics(
    mean_no_outliers,
    xrange=(5, 40),
    yrange_loss=(0.01, 2.5),
    #yrange_acc=(0, 1.1),
    figsize=(4.2*4, 4.2*1.5),
    xlog=True,
    ylog=True,
)


























# %%
