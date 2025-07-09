#%%
# type: ignore


%load_ext autoreload
%autoreload 2

from pathlib import Path
import pandas as pd
import json
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from pprint import pprint
from dr_gen.analyze.parsing import load_runs_from_dir
from dr_gen.analyze.database import ExperimentDB
from dr_gen.analyze.schemas import AnalysisConfig, Hpms
from IPython.display import display

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
        'sgd': lambda run: run.hpms._flat_dict['optim.name'] == 'sgdm',
        #'no residual': lambda run: run.hpms._flat_dict['model.use_residual'] == True,
        'no mixup': lambda run: run.hpms._flat_dict['train_transforms.mixup'] == False,
        'no cutmix': lambda run: run.hpms._flat_dict['train_transforms.cutmix'] == False,
        'no randaug': lambda run: run.hpms._flat_dict['train_transforms.randaug'] == False,
        'no colorjitter': lambda run: run.hpms._flat_dict['train_transforms.colorjitter'] == False,
        'no hflip': lambda run: run.hpms._flat_dict['train_transforms.hflip'] == True,
        #'no label smoothing': lambda run: run.hpms._flat_dict['train_transforms.label_smoothing'] == 0.0,
        'completed run': lambda run: run.hpms._flat_dict['status'] == 'completed',
    },
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
db.config.use_runs_filters['no label smoothing'] = lambda run: run.hpms._flat_dict['train_transforms.label_smoothing'] == 0.0
db.update_filtered_runs()
print(f"Number of active runs (without label smoothing): {len(db.active_runs)}")
db.config.use_runs_filters['use residual'] = lambda run: run.hpms._flat_dict['model.use_residual'] == True
db.update_filtered_runs()
print(f"Number of active runs (with residual): {len(db.active_runs)}")

# %%

db._runs_df.head()

db._metrics_df.head()


# %%
pprint(db._runs_df['tag'].unique().to_list())


# %%