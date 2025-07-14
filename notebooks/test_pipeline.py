#!/usr/bin/env python3
"""
Test script to validate the integrated plotting pipeline works correctly.
"""

from pathlib import Path
import pandas as pd
import polars as pl
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from dr_gen.analyze.database import ExperimentDB
from dr_gen.analyze.schemas import AnalysisConfig
from typing import Any, Dict

def group_metric_by_hpms_v2(
    db: ExperimentDB,
    metric: str,
    **hpm_filters: Any
) -> Dict[tuple, pd.DataFrame]:
    """Group metric time series by main_hpms (excluding 'seed'), filtered by hpm_filters."""
    # Get the filtered runs based on hpm filters
    filtered_runs_df = db._runs_df
    
    # Apply filters
    for k, v in hpm_filters.items():
        col_name = k.replace('__', '.')
        if col_name in filtered_runs_df.columns:
            filtered_runs_df = filtered_runs_df.filter(pl.col(col_name) == v)
        else:
            print(f"Warning: Column '{col_name}' not found in runs dataframe")
    
    # Get run_ids from filtered runs
    run_ids = filtered_runs_df['run_id'].to_list()
    
    if not run_ids:
        print("No runs found matching the filters")
        return {}
    
    print(f"Found {len(run_ids)} runs matching filters")
    
    # Get metrics for these runs
    metrics_df = db._metrics_df.filter(
        (pl.col('run_id').is_in(run_ids)) & 
        (pl.col('metric') == metric)
    )
    
    print(f"Found {len(metrics_df)} metric entries")
    
    # Join metrics with run info to get hpms
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
    
    print(f"Created {len(result)} groups")
    return result

def main():
    """Test the pipeline functionality."""
    
    # Initialize configuration
    config_path = Path("../configs/").absolute()
    
    with initialize_config_dir(config_dir=str(config_path), version_base="1.3"):
        cfg = compose(config_name="config", overrides=["paths=mac"])
    OmegaConf.resolve(cfg)
    print("✓ Configuration loaded successfully")
    
    # Set up experiment directory and analysis config
    exp_dir = Path(f"{cfg.paths.data}/loss_slope/exps_v1/experiments/test_sweep/")
    
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
    
    # Load experiments into database
    db = ExperimentDB(config=analysis_cfg, lazy=False)
    db.load_experiments()
    print(f"✓ Loaded {len(db.all_runs)} total runs, {len(db.active_runs)} active runs")
    
    # Show available hyperparameter combinations
    print("\nAvailable hyperparameter combinations:")
    unique_hpms = db.active_runs_df.select([
        'model.architecture', 'model.width_mult', 'optim.lr', 
        'optim.weight_decay', 'batch_size'
    ]).unique().limit(5)
    print(unique_hpms.to_pandas())
    
    # Test the group_metric_by_hpms_v2 function
    print("\n--- Testing group_metric_by_hpms_v2 function ---")
    test_results = group_metric_by_hpms_v2(
        db, 
        'train_loss', 
        batch_size=128, 
        model__width_mult=1.0
    )
    
    if test_results:
        print(f"✓ Successfully created {len(test_results)} groups")
        
        # Show details of first group
        first_key = list(test_results.keys())[0]
        first_df = test_results[first_key]
        
        print(f"\nFirst group key: {first_key}")
        print(f"First group shape: {first_df.shape}")
        print(f"Seeds in first group: {sorted(first_df['seed'].unique())}")
        print(f"Epochs in first group: {first_df['epoch'].min()} to {first_df['epoch'].max()}")
        print(f"\nFirst few rows:")
        print(first_df.head())
        
        # Validate data structure
        required_cols = ['epoch', 'seed', 'train_loss']
        has_required = all(col in first_df.columns for col in required_cols)
        print(f"\n✓ Data has required columns: {has_required}")
        
        # Check if we can aggregate by seed
        avg_by_epoch = first_df.groupby('epoch')['train_loss'].mean()
        print(f"✓ Can aggregate by epoch: {len(avg_by_epoch)} unique epochs")
        
    else:
        print("✗ No results found - check filters")
    
    print("\n--- Pipeline validation complete ---")

if __name__ == "__main__":
    main()