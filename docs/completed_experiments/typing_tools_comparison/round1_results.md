<!-- METHODOLOGICAL NOTES:
- Baseline measurement: Shows 1046 mypy errors using mp command on src/ directory
- Tool application scope: All tools applied only to src/ directory
- Measurement methodology: Used mp command which writes to .mypy_errors.jsonl file
- Measurement discrepancy: This experiment shows 1046 baseline errors while 
  automated_typing_experiment_results.md shows 221 baseline errors (5x difference)
- Possible explanations for discrepancy:
  a) Different mypy configuration or version between experiments
  b) Code changes between experiments 
  c) Different measurement commands (mp vs direct mypy output parsing)
- Execution completeness: Tested 4 configurations as shown
- File outputs: Generated .mypy_errors_baseline.jsonl and stage-specific measurements
-->

# Round 1 Typing Tool Experiment Results

**Date:** June 23, 2025  
**Baseline:** 1046 mypy errors, 0 ruff errors  
**Primary error type:** 757 `no-untyped-def` errors (72% of total)

## Configuration Results

### 🏆 autotyping-safe (WINNER)
**Command:** `autotyping --safe --none-return src/`  
**Results:**
- Mypy: 1046 → 959 (-87 errors, -8.3%)
- Ruff: 0 → 0 (no regressions)
- Primary improvement: `no-untyped-def` 757 → 670 (-87)
- **Mechanism:** Added `-> None` return annotations to functions

### ❌ autotyping-aggressive (POOR)
**Command:** `autotyping src/` 
**Results:**
- Mypy: 1046 → 1046 (essentially no change)
- Ruff: 0 → 0 (clean)
- **Issue:** Tool was surprisingly inactive without explicit flags

### ❌ infer-types-default (PROBLEMATIC)
**Command:** `python -m infer_types src/`
**Results:**
- Mypy: 1046 → 963 (+52 net after auto-fixes, initially worse)
- Ruff: 0 → 12 (import ordering, undefined names)
- **Problems:** 
  - Created 193 new `type-arg` errors
  - Import placement issues (E402)
  - Undefined `ndarray` references (F821)
- **Root cause:** Over-aggressive generic type inference

### ❌ infer-types-conservative (ALSO PROBLEMATIC)  
**Command:** `python -m infer_types --no-assumptions src/`
**Results:**
- Mypy: 1046 → 968 (+23 net)
- Ruff: 0 → 11 (similar import issues)
- **Issue:** Conservative flag insufficient to prevent core problems

## Key Insights

1. **Incremental typing works:** Small, safe changes (like `-> None`) provide substantial benefits
2. **Aggressive inference fails:** Complex type inference introduces more problems than it solves
3. **Import management matters:** Tools that add imports mid-file create ruff violations
4. **Safe flags are crucial:** The difference between autotyping modes was dramatic

## Files Generated
- `.mypy_errors_baseline.jsonl` - Baseline measurement (1046 errors)
- Individual stage measurements saved for each configuration

## Next Steps
Need to test incremental improvements building on autotyping-safe success.