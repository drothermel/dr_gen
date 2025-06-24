# Typing Experiment Status Update

**Date**: June 24, 2025  
**Status**: Ready for clean experiment retry with improved methodology

## Key Learnings About Error Measurement

### The Cross-Directory Error Phenomenon
We discovered that mypy error counts are **not additive** across directories:
- Individual directory checks: 643 total errors (src: 230, scripts: 158, tests: 255)
- Combined check: 1261 errors
- **618 "cross-directory" errors** only appear when checking all directories together

This happens because:
1. Test files importing untyped src functions generate additional "usage context" errors
2. Type mismatches across module boundaries only surface with full visibility
3. The same file can generate different errors depending on how it's checked

### Implications
- Always measure against combined directory baseline for accurate comparisons
- Fixing src/ types has cascade effects, eliminating downstream test errors
- Previous experiments may have underestimated improvements by not accounting for this

## New Tooling Available

### 1. Type Checking Script
We created `check_types.sh` that:
- Runs mypy on each directory individually
- Runs mypy on all directories combined
- Shows the cross-directory error delta
- Self-cleaning (removes old error files)

```bash
./check_types.sh
# Output:
# 🧹 Cleaning up old error files...
# === Checking src ===
# Errors in src: 230
# === Checking scripts ===  
# Errors in scripts: 158
# === Checking tests ===
# Errors in tests: 255
# === TOTAL ERRORS (individual): 643 ===
# === Checking all directories together ===
# Errors (combined): 1261
# ⚠️  Cross-directory errors detected: 618 additional errors
```

### 2. Updated Commands
- `mp src scripts tests` - Standard combined check
- Error count extraction: `head -1 .mypy_errors.jsonl | jq -r '.error_count'`
- Per-directory analysis now available via `.src_mypy_errors.jsonl`, etc.

## Corrected Baselines

### True Baseline (All Directories)
```
Clean repo: 1245 errors (combined check of src, scripts, tests)
```

### Tool Effectiveness (Preliminary)
Based on combined directory checks:
- **Autotyping** (`--safe --none-return` on all dirs): -170 errors (13.7% reduction)
- **Infer-types**: Estimated ~45 errors (3.6% reduction)
- **MonkeyType**: Demonstrated -28 to -29 additional errors in limited testing

## Critical Insights for Clean Experiment

### 1. Measurement Protocol
- Always run autotyping on ALL directories: `src/ scripts/ tests/`
- Always measure errors using combined check: `mp src scripts tests`
- Use `check_types.sh` to understand error distribution

### 2. Tool Application Order
Recommended sequence:
1. **Baseline**: Clean repo, combined check
2. **Autotyping**: Apply to all directories with `--safe --none-return`
3. **Infer-types**: Apply after autotyping (may have limited additional impact)
4. **MonkeyType**: Apply after static tools, focusing on high-impact modules

### 3. MonkeyType Considerations
- Must use `pytest -n0` to disable parallel execution
- Set `MONKEYTYPE_TRACE_MODULES="dr_gen"`
- Clean database between runs: `rm -f monkeytype.sqlite3`
- Focus on heavily-imported modules for cascade benefits

## Automated Experimental Tools

We now have a suite of tools to run systematic typing experiments:

### Core Tools

1. **`./run_typing_experiment.sh`** - Main experiment orchestrator
   ```bash
   # Run single tool experiments
   ./run_typing_experiment.sh autotyping
   ./run_typing_experiment.sh infer-types --only return
   ./run_typing_experiment.sh monkeytype --modules utils,metrics
   
   # Run combined experiment
   ./run_typing_experiment.sh combined --sequence autotyping,infer-types,monkeytype
   ```

2. **`./measure_typing_baseline.sh`** - Comprehensive error measurement
   ```bash
   # Outputs JSON with individual and combined error counts
   ./measure_typing_baseline.sh > baseline.json
   ```

3. **`./compare_typing_results.sh`** - Compare before/after states
   ```bash
   # Generate markdown comparison report
   ./compare_typing_results.sh baseline.json after.json
   ```

4. **`./analyze_import_graph.sh`** - Find high-impact modules
   ```bash
   # Show top 10 most imported modules
   ./analyze_import_graph.sh
   
   # Show what imports a specific module
   ./analyze_import_graph.sh --module metrics
   ```

5. **`./setup_monkeytype_trace.sh`** - MonkeyType-specific setup
   ```bash
   # Trace default high-impact modules
   ./setup_monkeytype_trace.sh
   
   # Trace specific modules
   ./setup_monkeytype_trace.sh --modules utils,display
   ```

### Experimental Protocol

#### Single Tool Evaluation
```bash
# 1. Clean state
git checkout -- .

# 2. Run experiment (automatically handles baseline, branching, measurement)
./run_typing_experiment.sh autotyping

# 3. Review results in experiments/TIMESTAMP_TOOL/
cat experiments/*/summary.md
```

#### Combined Tool Evaluation
```bash
# Test tools in sequence with intermediate measurements
./run_typing_experiment.sh combined --sequence autotyping,infer-types,monkeytype

# Results show impact after each tool
ls experiments/*/after_*.json
```

#### Options for Experiments
- `--skip-baseline` - Use existing baseline measurement
- `--no-branch` - Run in current branch without creating experiment branch
- `--keep-changes` - Don't reset after experiment (useful for debugging)

### Results Structure

Each experiment creates:
```
experiments/
└── 20240624_143022_autotyping/
    ├── baseline.json          # Initial state
    ├── after.json            # Final state
    ├── summary.md            # Detailed comparison
    ├── report.md             # Quick overview
    ├── application.log       # Tool output
    ├── changes.diff          # Git diff
    └── metadata.json         # Experiment details
```

## Next Steps for Clean Experiment

1. **Install Required Tools**:
   ```bash
   uv add --dev autotyping infer-types monkeytype
   ```

2. **Run Baseline Analysis**:
   ```bash
   # Measure current state
   ./measure_typing_baseline.sh > initial_baseline.json
   
   # Find high-impact modules
   ./analyze_import_graph.sh --top 20
   ```

3. **Run Individual Tool Experiments**:
   ```bash
   ./run_typing_experiment.sh autotyping
   ./run_typing_experiment.sh infer-types
   ./run_typing_experiment.sh monkeytype --modules utils,metrics,model
   ```

4. **Run Combined Experiment**:
   ```bash
   ./run_typing_experiment.sh combined
   ```

5. **Analyze Results**:
   ```bash
   # Compare all experiments
   for exp in experiments/*/summary.md; do
     echo "=== $exp ==="
     grep "Total Errors:" "$exp"
   done
   ```

## References

- `docs/mypy-reference.md` - Updated with cross-directory error section
- `docs/autotyping-reference.md` - Tool capabilities and limitations
- `docs/infer-types-reference.md` - Comparison with autotyping
- `docs/monkeytype-reference.md` - Runtime type inference guide
- `check_types.sh` - New measurement tool

## Summary

We now understand that:
1. Error counts are context-dependent and non-additive
2. Previous measurements may have been misleading
3. Fixing source types has amplified benefits
4. We have proper tooling to measure accurately

The experiment can now proceed with confidence that we're measuring the right things in the right way.