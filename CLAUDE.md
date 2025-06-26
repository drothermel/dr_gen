# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

dr_gen is a PyTorch-based deep learning training framework focused on generative models and computer vision tasks. It provides a complete pipeline for training, evaluating, and analyzing neural networks with extensive experiment management capabilities.

## Common Development Commands

### Setup and Dependencies
```bash
# Sync dependencies
us  # (uv sync)

# Add new dependencies
uv add [package_name]
uv add --dev [dev_package]  # For development dependencies
```

### Code Quality
```bash
# Format code
format  # (uv run ruff format)

# Lint and auto-fix
lint_fix  # (uv run ruff check --fix)

# Just check linting
lint  # (uv run ruff check)

# IMPORTANT: Always run lint_fix after making code changes
```

### Type Checking
```bash
# Run type checking with mp command
mp src scripts tests

# IMPORTANT: The mp command behavior:
# - ALWAYS writes output to .mypy_errors.jsonl (never to stdout)
# - Returns "Error" exit status when type errors are found (this is EXPECTED)
# - An "Error" status means "type errors found", NOT "tool failed"
# - First line of .mypy_errors.jsonl contains {"type": "metadata", "error_count": N}
# - Subsequent lines contain individual error details

# Correct workflow:
mp src scripts tests  # Run type checking (may show "Error")
# Then immediately read .mypy_errors.jsonl to see results
# Do NOT retry the command or assume it failed
```

### Testing
```bash
# Run all tests
pt -v  # (pytest)

# Run specific test file
pt tests/train/test_model.py -v

# Run tests matching pattern
pt -k "test_create_optim" -v

# Run tests without parallelization (needed for MonkeyType)
pt tests/ -v -n0
```

### Training and Evaluation
```bash
# Basic training run
uvrp scripts/train.py

# Training with custom parameters (Hydra overrides)
uvrp scripts/train.py model.name=resnet50 optim.lr=0.01 train.batch_size=128

# Run evaluation only
uvrp scripts/test_eval.py eval.run=true train.run=false val.run=false

# Parallel seed sweep (8 processes, seeds 0-19)
uvrp scripts/parallel_runs.py -p 8 --start_seed 0 --max_seed 19
```

### Analysis
```bash
# Analyze training logs
uvrp scripts/analyze.py log_file=path/to/log.jsonl

# Convert pkl results to JSON
uvrp scripts/pkl_to_json.py input.pkl output.json
```

### Git Workflow
```bash
# Use these shortcuts ALWAYS (never full git commands):
gst         # git status
gd_agent    # git --no-pager diff
glo         # git log --oneline -10
ga .        # git add .
gc -m "msg" # git commit -m "msg"

# Standard workflow:
glo         # See recent history
# Make changes
lint_fix    # Verify quality and auto-fix
gst         # Check status
gd_agent    # Review changes
ga .        # Stage files
gc -m "type: description"  # Commit with proper message format
```

## Development Guidelines

### Code Quality Rules
- **Use type hints** on all functions
- **Place imports at top** with existing imports, never mid-file
- **Show full modified functions** when editing, not just diffs
- **Run lint_fix after changes** to verify quality

### Performance Requirements (ML-specific)
```python
# YES - Use asserts (fast for training loops)
assert len(inputs) == len(targets), f"Mismatch: {len(inputs)} vs {len(targets)}"

# NO - Don't use exceptions in hot paths (slow)
if len(inputs) != len(targets):
    raise ValueError("Mismatch")
```

### Hydra Configuration
- All scripts use Hydra for configuration
- Override parameters with dot notation: `optim.lr=0.01`
- Config files in `configs/` directory

## High-Level Architecture

### Core Components

1. **Training Pipeline** (`src/dr_gen/train/`)
   - `loops.py`: Implements the main training loop with epoch-based training, validation, and evaluation phases
   - `model.py`: Handles model creation (via timm), optimizer setup, learning rate scheduling, and checkpointing
   - `evaluate.py`: Contains evaluation logic for computing accuracy and loss metrics

2. **Data Management** (`src/dr_gen/data/`)
   - `load_data.py`: Provides unified interface for loading datasets (CIFAR-10/100) with transforms
   - Integrates with Hydra configs for data augmentation and preprocessing

3. **Metrics System** (`src/dr_gen/utils/metrics.py`)
   - `GenMetrics` class: Central metrics tracking system that handles:
     - Epoch-level metrics aggregation
     - JSON logging for experiment tracking
     - Integration with Hydra's structured logging

4. **Analysis Tools** (`src/dr_gen/analyze/`)
   - `run_data.py`: Parses JSON logs and extracts metrics across training runs
   - `metric_curves.py`: Generates plots for training dynamics
   - `ks_stats.py`: Performs statistical analysis between different runs
   - `bootstrapping.py`: Computes confidence intervals for metrics

### Configuration System

The project uses Hydra for configuration management with a hierarchical structure:
- Main config: `configs/config.yaml`
- Modular configs: `data/`, `paths/`, `transform/` subdirectories
- Schema validation via dataclasses in `schemas.py`

Key configuration groups:
- `model`: Model architecture and initialization
- `optim`: Optimizer, learning rate, and scheduler settings
- `train/val/eval`: Phase-specific settings (batch size, frequency)
- `metrics`: Logging configuration and output formats

### External Dependencies

- **timm**: Primary source for pre-trained models and training utilities
- **dr_util**: Custom utility library for experiment management
- **Hydra**: Configuration and experiment tracking
- **PyTorch ecosystem**: Core deep learning functionality

### SLURM Integration

The project includes SLURM support for HPC environments:
- Template scripts in `slurm/` for batch job submission
- Parallel training scripts that handle multiple seeds/configurations
- Archive system in `sbatch_archive/` for tracking submitted jobs

## Key Design Patterns

1. **Modular Training Loop**: Clear separation between training, validation, and evaluation phases
2. **Metrics Aggregation**: Centralized metrics tracking with automatic averaging and logging
3. **Configuration-Driven**: All hyperparameters and settings controlled via Hydra configs
4. **Reproducibility**: Explicit seed control and deterministic training options
5. **Analysis Pipeline**: Separate tools for post-hoc analysis without re-running experiments