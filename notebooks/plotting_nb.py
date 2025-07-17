#%%
# type: ignore
%load_ext autoreload
%autoreload 2

from pathlib import Path
import matplotlib.pyplot as plt
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from pprint import pprint

from dr_gen.analyze.parsing import load_runs_from_dir
from dr_gen.analyze.database import ExperimentDB
from dr_gen.analyze.schemas import AnalysisConfig
from IPython.display import display
from dr_gen.analyze.visualization import plot_training_metrics, plot_metric_group
from dr_gen.analyze import GroupedRuns

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
        'epoch': 'Epoch',
    },
    hparam_display_names={
        'optim.lr': 'LR',
        'optim.weight_decay': 'WD',
        'optim.name': 'Opt',
        'batch_size': 'BS',
        'epochs': 'Epochs',
        'lrsched.sched_type': 'LRSched',
        'lrsched.warmup_epochs': 'Warmup Epochs',
        'model.architecture': 'Arch',
        'model.dropout_prob': 'Dropout',
        'model.name': 'Model',
        'model.norm_type': 'Norm',
        'model.use_residual': 'Residual On?',
        'seed': 'Seed',
        'tag': 'Run Name',
        'model.width_mult': 'WidthMult',
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
    grouping_exclude_hpm_display_names=['model.architecture'],
)
#display(analysis_cfg)

# %% Create ExperimentDB

db = ExperimentDB(
    config=analysis_cfg, lazy=False
)
db.load_experiments()
print(f"All Runs: {len(db.all_runs)}, Active Runs: {len(db.active_runs)}")


# %% ========== UPDATED EXAMPLES USING GROUPEDRUNS API ==========
# All examples below have been updated to use the new GroupedRuns class
# Benefits: cleaner code, immutable operations, better encapsulation
# Old workflow: manual state management with multiple variables
# New workflow: fluent API with clear data flow

# %% Basic grouping with new GroupedRuns API
grouped = GroupedRuns.from_db(db)
print(f"Grouping: {grouped}")
print(f"Hyperparameters: {grouped.hpm_names}")

# Get group information
info = grouped.get_group_info()
print(f"Number of groups: {info['num_groups']}")
print(f"Total runs: {info['total_runs']}")
print(f"Runs per group - min: {info['runs_per_group']['min']}, max: {info['runs_per_group']['max']}, mean: {info['runs_per_group']['mean']:.1f}")

# Show first few groups with their hyperparameter values
hpm_dicts = grouped.group_keys_to_dicts()
display_dicts = grouped.group_keys_to_dicts(use_display_names=True)
descriptions = grouped.describe_groups()

print("\nFirst 5 groups:")
for i, (group_key, runs) in enumerate(grouped):
    if i >= 5:  # Only show first 5 groups
        print("...")
        break
    print(f"\nGroup {i+1}: {len(runs)} runs")
    print(f"  Technical names: {hpm_dicts[group_key]}")
    print(f"  Display names: {display_dicts[group_key]}")
    print(f"  Formatted: {descriptions[group_key]}")

# %% Filter groups by minimum run count
# Example: Keep only groups with at least 3 runs for robust statistics
print("\n=== Filtering by Minimum Run Count ===")
min_runs_required = 3

# Show run count distribution before filtering
run_counts = [len(runs) for runs in grouped.groups.values()]
print(f"Run counts before filtering: min={min(run_counts)}, max={max(run_counts)}")
print(f"Groups with < {min_runs_required} runs: {sum(1 for count in run_counts if count < min_runs_required)}")

# Filter to well-sampled groups only
well_sampled = grouped.filter(min_runs=min_runs_required)
print(f"\nAfter filtering (min_runs={min_runs_required}): {well_sampled}")

# Show what was removed
removed_count = len(grouped) - len(well_sampled)
print(f"Removed {removed_count} groups with insufficient runs")

# Show remaining run count distribution
if len(well_sampled) > 0:
    remaining_run_counts = [len(runs) for runs in well_sampled.groups.values()]
    print(f"Remaining run counts: min={min(remaining_run_counts)}, max={max(remaining_run_counts)}")
    
    # Show a few examples of what remained
    print(f"\nFirst 3 well-sampled groups:")
    well_sampled_descriptions = well_sampled.describe_groups()
    for i, (group_key, runs) in enumerate(well_sampled):
        if i >= 3:
            break
        print(f"  {len(runs)} runs: {well_sampled_descriptions[group_key]}")
else:
    print("No groups meet the minimum run requirement!")

# %% Additional run count filtering examples
print("\n=== Additional Run Count Filtering Examples ===")

# Filter to groups with exactly a certain range
moderate_sampled = grouped.filter(min_runs=2, max_runs=5)
print(f"Groups with 2-5 runs: {len(moderate_sampled)} groups")

# Filter to highly sampled groups only
if max(run_counts) >= 5:
    highly_sampled = grouped.filter(min_runs=5)
    print(f"Highly sampled groups (≥5 runs): {len(highly_sampled)} groups")

# Show the power of combining filters
# Example: Well-sampled groups with specific hyperparameter constraints
if len(well_sampled) > 0:
    print(f"\nExample: Combining run count + hyperparameter filtering")
    
    # Example: Get well-sampled groups for a specific learning rate range
    try:
        # This assumes we have lr in our hyperparameters
        combined_filtered = well_sampled.filter(
            include={'optim.lr': [0.01, 0.03, 0.1]} if 'optim.lr' in well_sampled.hpm_names else None
        )
        print(f"Well-sampled + specific LR values: {len(combined_filtered)} groups")
    except:
        print("Note: Specific hyperparameter filtering depends on available hpms in your data")
    
    print("This ensures robust statistics (≥3 runs) AND focuses on relevant hyperparameter ranges")

# %% Single group metrics extraction using GroupedRuns
# Get first group for detailed analysis
first_group_key = list(grouped.groups.keys())[0]
first_group_runs = grouped.groups[first_group_key]

# Extract metrics using GroupedRuns (creates dict with single group)
single_group_metrics = {first_group_key: grouped.groups[first_group_key]}
single_grouped = GroupedRuns(single_group_metrics, grouped.hpm_names, db)

# Extract all relevant metrics
metrics_dict = single_grouped.to_metric_dfs(['train_loss', 'train_acc', 'val_loss', 'val_acc', 'epoch', 'lr'])
metric_dfs = metrics_dict[first_group_key]  # Get the metrics for our single group

pprint(list(metric_dfs.keys()))
print(f"Train loss shape: {metric_dfs['train_loss'].shape}")
print("First few train loss values:")
print(metric_dfs['train_loss'].head())

# Print summary statistics
mean_y = metric_dfs['train_loss'].mean(axis=1)
std_y = metric_dfs['train_loss'].std(axis=1)
group_description = single_grouped.describe_groups()[first_group_key]
print(f"\nGroup hyperparameters: {group_description}")
print(f"Number of runs: {len(first_group_runs)}")
print(f"Final {db.get_display_name('train_loss').lower()}: {mean_y.iloc[-1]:.4f} ± {std_y.iloc[-1]:.4f}")


# %% Different metrics - validation accuracy using GroupedRuns
# Example: Validation accuracy over epochs using the single group
if 'val_acc' in metric_dfs:
    plot_metric_group(
        metrics_dict,  # Use the full dict with group key
        x_metric='epoch',
        y_metrics='val_acc',
        db=db,
        group_descriptions=single_grouped.describe_groups(),
        ylim=(0, 1.0),
        ylabel='Validation Accuracy',
        title='Model Performance'
    )

# %% Multiple metrics on same plot using GroupedRuns
# Example: Plot both train and validation loss together
plot_metric_group(
    metrics_dict,  # Use the full dict with group key
    x_metric='epoch',
    y_metrics=['train_loss', 'val_loss'],
    db=db,
    group_descriptions=single_grouped.describe_groups(),
    color_scheme='colorblind',
    figsize=(10, 6),
    ylabel='Loss',
    title='Training vs Validation Loss'
)

# %% Multiple metrics with different color schemes using GroupedRuns
# Example: Using Paul Tol's color palette
if all(m in metric_dfs for m in ['train_acc', 'val_acc']):
    plot_metric_group(
        metrics_dict,  # Use the full dict with group key
        x_metric='epoch',
        y_metrics=['train_acc', 'val_acc'],
        db=db,
        group_descriptions=single_grouped.describe_groups(),
        color_scheme='paul_tol',
        figsize=(10, 6),
        ylim=(0, 1.0),
        ylabel='Accuracy',
        title='Model Accuracy Comparison'
    )

# %% ========== SYSTEMATIC PLOTTING EXAMPLES ==========
# These examples demonstrate the NEW color_by and linestyle_by parameters for plot_metric_group:
#
# color_by options:
#   - 'metric': Colors distinguish different metrics (default, previous behavior)
#   - 'group': Colors distinguish different groups/hyperparameter combinations 
#   - 'hyperparameter_name': Colors distinguish different values of that hyperparameter
#
# linestyle_by options:
#   - 'group': Linestyles distinguish different groups (default, previous behavior)
#   - 'metric': Linestyles distinguish different metrics
#   - 'hyperparameter_name': Linestyles distinguish different values of that hyperparameter
#
# This allows flexible control over which dimension (metrics vs hyperparameters) drives visual distinction

# Filter to well-sampled groups and select specific BS + Width Mult configuration

# Step 1: Filter to groups with at least 3 runs for robust statistics
well_sampled = grouped.filter(min_runs=3)
print(f"Well-sampled groups (≥3 runs): {well_sampled}")

# Step 2: Select a specific batch size and width multiplier combination
# First, let's see what BS and width mult values are available
if 'batch_size' in well_sampled.hpm_names and 'model.width_mult' in well_sampled.hpm_names:
    available_configs = well_sampled.group_keys_to_dicts()
    bs_values = set()
    wm_values = set()
    for config in available_configs.values():
        if 'batch_size' in config:
            bs_values.add(config['batch_size'])
        if 'model.width_mult' in config:
            wm_values.add(config['model.width_mult'])
    
    print(f"Available batch sizes: {sorted(bs_values)}")
    print(f"Available width multipliers: {sorted(wm_values)}")
    
    # Select specific configuration for focused analysis
    selected_bs = 512  # Choose based on what's available in your data
    selected_wm = 1.0  # Choose based on what's available in your data
    
    base_config = {
        'batch_size': selected_bs,
        'model.width_mult': selected_wm
    }
    
    print(f"\nSelected configuration: BS={selected_bs}, Width Mult={selected_wm}")
    
    # Get (lr, wd) combinations for this configuration
    lr_wd_analysis = well_sampled.matching(
        base_hpms=base_config,
        varying_hpms=['optim.lr', 'optim.weight_decay']
    )
    
    print(f"Found {len(lr_wd_analysis)} (lr, wd) combinations for analysis")
    if len(lr_wd_analysis) > 0:
        print("Available combinations:")
        descriptions = lr_wd_analysis.describe_groups()
        for key, desc in descriptions.items():
            print(f"  {key}: {desc}")
    
else:
    print("Note: Adjusting for available hyperparameters in your dataset")
    # Fallback: use whatever hyperparameters are available
    lr_wd_analysis = well_sampled

# %% Example 1: Fixed Learning Rate across Weight Decay values (Single Metric)
print("\n=== Example 1: Fixed LR across WD values (Single Metric) ===")

# Get all unique lr and wd values
lr_wd_dicts = lr_wd_analysis.group_keys_to_dicts()
lr_values = set()
wd_values = set()

for config in lr_wd_dicts.values():
    if 'optim.lr' in config:
        lr_values.add(config['optim.lr'])
    if 'optim.weight_decay' in config:
        wd_values.add(config['optim.weight_decay'])

# Pick a specific learning rate to analyze
selected_lr = sorted(lr_values)[0]  # Use first available LR
print(f"Analyzing LR={selected_lr} across different weight decay values")

# Filter to this specific learning rate
fixed_lr_analysis = lr_wd_analysis.filter(
    include={'optim.lr': [selected_lr]}
)

# Extract metrics and plot using the NEW color control feature
# Get all plotting data in one call: metrics, descriptions, and hyperparameter dicts
fixed_lr_metrics, fixed_lr_descriptions, fixed_lr_hparams = fixed_lr_analysis.get_plotting_data(['train_loss', 'val_loss', 'epoch'])

print(f"Groups varying by weight decay (LR={selected_lr} fixed):")
for key, desc in fixed_lr_descriptions.items():
    print(f"  {key}: {desc}")

# Alternative: Use specific hyperparameter for color control
# This achieves the same result but more explicitly
print(f"\nAlternative approach - color by specific hyperparameter:")
plot_metric_group(
    fixed_lr_metrics,
    x_metric='epoch',
    y_metrics='train_loss',
    db=db,
    group_descriptions=fixed_lr_descriptions,
    group_hparams=fixed_lr_hparams,
    color_by='optim.weight_decay',  # NEW: Colors by weight decay hyperparameter
    linestyle_by='optim.lr',  # Linestyles by weight decay too
    figsize=(10, 6),
    title=f'Training Loss: LR={selected_lr} by Weight Decay (alternative coloring)'
)


# %% Example 2: Fixed Weight Decay across Learning Rate values (Single Metric)
print("\n=== Example 2: Fixed WD across LR values (Single Metric) ===")

# Pick a specific weight decay to analyze
selected_wd = sorted(wd_values)[0]  # Use first available WD
print(f"Analyzing WD={selected_wd} across different learning rate values")

# Filter to this specific weight decay
fixed_wd_analysis = lr_wd_analysis.filter(
    include={'optim.weight_decay': [selected_wd]}
)

# Extract metrics and plot using the NEW color control feature
# Get all plotting data in one call: metrics, descriptions, and hyperparameter dicts
fixed_wd_metrics, fixed_wd_descriptions, fixed_wd_hparams = fixed_wd_analysis.get_plotting_data(['train_loss', 'val_loss', 'epoch'])

print(f"Groups varying by weight decay (WD={selected_wd} fixed):")
for key, desc in fixed_wd_descriptions.items():
    print(f"  {key}: {desc}")


plot_metric_group(
    fixed_wd_metrics,
    x_metric='epoch',
    y_metrics='train_loss',
    db=db,
    group_descriptions=fixed_wd_descriptions,
    group_hparams=fixed_wd_hparams,
    color_by='optim.lr',  # NEW: Colors by weight decay hyperparameter
    linestyle_by='optim.weight_decay',  # Linestyles by weight decay too
    figsize=(10, 6),
    title=f'Training Loss: WD={selected_wd} across Learning Rate Values'
)

# %% Example 3: All LR x WD combinations (Single Metric)
print("\n=== Example 3: All LR x WD combinations (Single Metric) ===")

# Show all combinations with systematic color/linestyle control
all_lr_wd_metrics, all_lr_wd_descriptions, all_lr_wd_hparams = lr_wd_analysis.get_plotting_data(['train_loss', 'val_loss', 'epoch'])

print(f"Showing all {len(all_lr_wd_metrics)} LR x WD combinations:")
for key, desc in all_lr_wd_descriptions.items():
    print(f"  {key}: {desc}")

plot_metric_group(
    all_lr_wd_metrics,
    x_metric='epoch',
    y_metrics='train_loss',
    db=db,
    group_descriptions=all_lr_wd_descriptions,
    group_hparams=all_lr_wd_hparams,
    color_by='optim.lr',  # Colors distinguish learning rates
    linestyle_by='optim.weight_decay',  # Linestyles distinguish weight decay
    figsize=(12, 6),
    title='Training Loss: All LR x WD Combinations'
)

# %% Example 4: Fixed Learning Rate across Weight Decay values (Multiple Metrics)
print("\n=== Example 4: Fixed LR across WD values (Multiple Metrics) ===")

# Use the same selected LR from Example 1
print(f"Analyzing LR={selected_lr} across WD values - Train vs Validation Loss")

fixed_lr_analysis = lr_wd_analysis.filter(
    include={'optim.lr': [selected_lr]}
)

# Get plotting data with hyperparameter information
fixed_lr_metrics, fixed_lr_descriptions, fixed_lr_hparams = fixed_lr_analysis.get_plotting_data(['train_loss', 'val_loss', 'epoch'])

plot_metric_group(
    fixed_lr_metrics,
    x_metric='epoch',
    y_metrics=['train_loss', 'val_loss'],
    db=db,
    group_descriptions=fixed_lr_descriptions,
    group_hparams=fixed_lr_hparams,
    color_by='optim.weight_decay',  # Colors distinguish metrics (train vs val)
    linestyle_by='metric',  # Linestyles distinguish WD values
    figsize=(12, 6),
    ylabel='Loss',
    title=f'Train vs Val Loss: LR={selected_lr} across Weight Decay Values'
)

# %% Example 5: Fixed Weight Decay across Learning Rate values (Multiple Metrics)
print("\n=== Example 5: Fixed WD across LR values (Multiple Metrics) ===")

# Use the same selected WD from Example 2
print(f"Analyzing WD={selected_wd} across LR values - Train vs Validation Loss")

fixed_wd_analysis = lr_wd_analysis.filter(
    include={'optim.weight_decay': [selected_wd]}
)

# Get plotting data with hyperparameter information
fixed_wd_metrics, fixed_wd_descriptions, fixed_wd_hparams = fixed_wd_analysis.get_plotting_data(['train_loss', 'val_loss', 'epoch'])

plot_metric_group(
    fixed_wd_metrics,
    x_metric='epoch',
    y_metrics=['train_loss', 'val_loss'],
    db=db,
    group_descriptions=fixed_wd_descriptions,
    group_hparams=fixed_wd_hparams,
    color_by='optim.lr',  # Colors distinguish metrics (train vs val)
    linestyle_by='metric',  # Linestyles distinguish LR values
    figsize=(12, 6),
    ylabel='Loss',
    title=f'Train vs Val Loss: WD={selected_wd} across Learning Rate Values'
)

# %% Example 6: All LR x WD combinations (Multiple Metrics)
print("\n=== Example 6: All LR x WD combinations (Multiple Metrics) ===")

print(f"Showing train vs validation loss for all {len(all_lr_wd_metrics)} LR x WD combinations")

plot_metric_group(
    all_lr_wd_metrics,
    x_metric='epoch',
    y_metrics=['train_loss', 'val_loss'],
    db=db,
    show_individual_runs=False,
    group_descriptions=all_lr_wd_descriptions,
    group_hparams=all_lr_wd_hparams,
    color_by='group',  # Colors distinguish metrics (train vs val)
    linestyle_by='metric',  # Linestyles distinguish each LR x WD combination
    figsize=(14, 6),
    ylabel='Loss',
    title='Train vs Val Loss: All LR x WD Combinations',
    xlim=(20, 50),
    ylim=(0, 1.0),
    legend_loc='upper left',
    separate_legends=True,  # Must be explicitly set
    color_legend_title="Learning Rate",
    linestyle_legend_title="Weight Decay",
    color_legend_loc="center left",
    linestyle_legend_loc="upper left"
)

# %% ========== PLOTTING EXAMPLES SUMMARY ==========
# The examples above demonstrate systematic plotting patterns with NEW color/linestyle control:
# 
# 1. FILTERING STRATEGY:
#    - Filter to groups with ≥3 runs for statistical reliability
#    - Fix batch size + width multiplier to focus analysis
#    - Extract (lr, wd) combinations for systematic comparison
#
# 2. PLOTTING PATTERNS & AESTHETIC STRATEGY:
#    - Example 1: Fixed LR, varying WD (single metric)
#      → color_by='optim.weight_decay' (colors distinguish WD values)
#    - Example 2: Fixed WD, varying LR (single metric)  
#      → color_by='optim.lr' (colors distinguish LR values)
#    - Example 3: All LR x WD combinations (single metric)
#      → color_by='optim.lr', linestyle_by='optim.weight_decay' (systematic 2D mapping)
#    - Example 4: Fixed LR, varying WD (multiple metrics)
#      → color_by='metric', linestyle_by='optim.weight_decay' (colors for train/val, lines for WD)
#    - Example 5: Fixed WD, varying LR (multiple metrics)
#      → color_by='metric', linestyle_by='optim.lr' (colors for train/val, lines for LR)
#    - Example 6: All LR x WD combinations (multiple metrics)
#      → color_by='metric', linestyle_by='group' (colors for train/val, lines for each combination)
#
# 3. COLOR/LINESTYLE CONTROL OPTIONS:
#    NEW: color_by and linestyle_by parameters allow flexible visual mapping:
#    - 'metric': Default, distinguishes different metrics
#    - 'group': Distinguishes different hyperparameter groups
#    - 'optim.lr', 'optim.weight_decay': Specific hyperparameter mapping
#    - 'none': Single color or linestyle for simplified plots
#
# 4. ADDITIONAL AESTHETIC CONTROL OPPORTUNITIES:
#    - figsize: Control plot dimensions (used in examples)
#    - title: Custom titles with dynamic values (used in examples)
#    - ylabel: Specify axis labels (used in examples)
#    - color_scheme: Different color palettes ('colorblind', 'paul_tol', 'categorical')
#    - x/yscale: Log scale options ('linear', 'log')
#    - xlim/ylim: Axis limits
#    - individual_alpha: Transparency for individual runs
#    - std_band: Show/hide standard deviation bands
#
# Each example is in its own cell for easy iteration on aesthetic options!

# %% Log scale with custom labels using GroupedRuns
# Even with custom labels, "(log scale)" is appended if not already present
plot_metric_group(
    metrics_dict,  # Use the single group dict
    x_metric='epoch',
    y_metrics='train_loss',
    db=db,
    group_descriptions=single_grouped.describe_groups(),
    figsize=(10, 6),
    xscale='log',
    yscale='log',
    xlabel='Training Progress',  # Custom label - will get "(log scale)" appended
    ylabel='Cross-Entropy Loss',  # Custom label - will get "(log scale)" appended
    xlim=(1, 50),
    ylim=(0.01, 2.5)
)

# %% ========== NEW GROUPEDRUNS API DEMONSTRATION ==========
# Clean workflow using the new GroupedRuns class

# Step 1: Create initial grouping from database
print("=== Step 1: Initial Grouping ===")
grouped = GroupedRuns.from_db(db)
print(f"Initial grouping: {grouped}")
print(f"Hyperparameters: {grouped.hpm_names}")

# Show some group information
info = grouped.get_group_info()
print(f"Group info: {info}")

# %%
# Step 2: Filter to specific hyperparameter combinations
print("\n=== Step 2: Filter to (lr, wd) combinations for specific config ===")
base_hpms = {
    'batch_size': 512,
    'model.width_mult': 1.0,
}

# Get all (lr, wd) combinations for this configuration
lr_wd_grouped = grouped.matching(
    base_hpms=base_hpms,
    varying_hpms=['optim.lr', 'optim.weight_decay']
)

print(f"Filtered to lr/wd combinations: {lr_wd_grouped}")
print(f"Found {len(lr_wd_grouped)} combinations")

# Show all combinations with descriptions
descriptions = lr_wd_grouped.describe_groups()
print("\nAll (lr, wd) combinations:")
for i, (key, desc) in enumerate(descriptions.items()):
    print(f"  {i+1}. {key}: {desc}")

# %%
# Step 3: Apply value-based filtering
print("\n=== Step 3: Filter by hyperparameter values ===")
filtered_grouped = lr_wd_grouped.filter(
    include={'optim.lr': [0.001, 0.003, 0.01, 0.03]},  # Only reasonable LRs
    exclude={'optim.weight_decay': [0.01]},  # Exclude high WD
)

print(f"After value filtering: {filtered_grouped}")

# Show what remains
remaining_descriptions = filtered_grouped.describe_groups()
print("Remaining combinations:")
for key, desc in remaining_descriptions.items():
    print(f"  {key}: {desc}")

# %%
# Step 4: Remove specific problematic pairs
print("\n=== Step 4: Remove specific pairs ===")
final_grouped = filtered_grouped.filter(
    exclude_pairs=[(0.03, 0.0003), (0.01, 0.0001)]  # Remove unstable combinations
)

print(f"Final selection: {final_grouped}")

# %%
# Step 5: Convert to metric DataFrames for analysis/plotting
print("\n=== Step 5: Extract metrics for analysis ===")
metrics_to_extract = ['train_loss', 'val_loss', 'train_acc', 'val_acc', 'epoch']
metric_dfs = final_grouped.to_metric_dfs(metrics_to_extract)

print(f"Extracted metrics for {len(metric_dfs)} groups")
print(f"Available metrics: {metrics_to_extract}")

# Show example of accessing data
first_group_key = list(metric_dfs.keys())[0]
first_group_data = metric_dfs[first_group_key]
print(f"\nExample - Group {first_group_key}:")
print(f"  Train loss shape: {first_group_data['train_loss'].shape}")
print(f"  Columns (runs): {list(first_group_data['train_loss'].columns)}")
print(f"  First few train loss values:\n{first_group_data['train_loss'].head(3)}")

# %%
# Step 6: Plot using the extracted DataFrames
print("\n=== Step 6: Plotting with metric DataFrames ===")

# Use the existing plot_metric_group function
group_descriptions = final_grouped.describe_groups()

# Plot single metric across groups
plot_metric_group(
    metric_dfs,
    x_metric='epoch',
    y_metrics='train_loss',
    db=db,
    group_descriptions=group_descriptions,
    figsize=(12, 6),
    title='Training Loss - Final Filtered Selection'
)

# %%
# Plot multiple metrics for comparison
plot_metric_group(
    metric_dfs,
    x_metric='epoch',
    y_metrics=['train_loss', 'val_loss'],
    db=db,
    group_descriptions=group_descriptions,
    figsize=(12, 6),
    ylabel='Loss',
    title='Train vs Validation Loss - Final Selection'
)

# %%
# Demonstrate additional GroupedRuns features
print("\n=== Additional GroupedRuns Features ===")

# Get hyperparameter dictionaries for programmatic access
hpm_dicts = final_grouped.group_keys_to_dicts()
display_dicts = final_grouped.group_keys_to_dicts(use_display_names=True)

print("Technical names:")
for key, hpm_dict in list(hpm_dicts.items())[:2]:  # Show first 2
    print(f"  {key}: {hpm_dict}")

print("\nDisplay names:")
for key, display_dict in list(display_dicts.items())[:2]:  # Show first 2
    print(f"  {key}: {display_dict}")

# Iterate over groups
print(f"\nIterating over {len(final_grouped)} groups:")
for i, (group_key, runs) in enumerate(final_grouped):
    if i >= 3:  # Only show first 3
        print("  ...")
        break
    print(f"  Group {group_key}: {len(runs)} runs")

# %%















# %% ------------------- IGNORE EVERYTHING BELOW THIS POINT -------------------

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
