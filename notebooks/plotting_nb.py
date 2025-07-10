#%%
# type: ignore


%load_ext autoreload
%autoreload 2

from pathlib import Path
import pandas as pd
import polars as pl
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
#db.config.use_runs_filters['no label smoothing'] = lambda run: run.hpms._flat_dict['train_transforms.label_smoothing'] == 0.0
#db.update_filtered_runs()
#print(f"Number of active runs (without label smoothing): {len(db.active_runs)}")

# %%
print('Important HPMS:', db.important_hpms)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
display(db.active_runs_df.drop('run_id', 'seed', 'tag', 'optim.name', 'model.architecture').unique().select(['batch_size', 'model.width_mult', 'optim.lr', 'optim.weight_decay']).sort(pl.all()).to_pandas())

# %%
db._metrics_df.head()


# %%
pprint(db._runs_df['tag'].unique().to_list())


# %%