# Typing Experiments Archive Summary

## Overview

This document summarizes the chronological progression of typing experiments in the dr_gen repository, highlighting methodological issues discovered and corrected interpretations based on our improved understanding of mypy's context-dependent error counting.

## Key Discovery: Non-Additive Error Counting

During the course of these experiments, we discovered that **mypy error counts are not additive across directories**. When checking directories together vs. separately, cross-directory import resolution creates additional errors that don't appear in individual checks. This fundamentally changed how we interpret earlier results.

## Chronological Experiment Summary

### 1. MyPy Parallel Fix Attempt (Early Experiment)
- **Location**: [completed_experiments/mypy_parallel_fix/](completed_experiments/mypy_parallel_fix/)
- **Goal**: Apply parallel fixing principles to reduce 395 mypy errors by 87%+
- **Methodological Issues**:
  - Assumed errors were additive across files
  - Used file-by-file error counts for load balancing
  - Didn't recognize many errors were external library issues
- **Actual Results**: 
  - Manual fixes: 17 errors reduced (4.3%)
  - External library config: 167 errors reduced (44%)
- **Corrected Interpretation**: The 395 baseline likely included many false positives from external libraries. The experiment validated that configuration changes are more effective than manual fixes.

### 2. Phase 1: Infer-Types Experiment
- **Location**: [completed_experiments/infer_types_phase1/](completed_experiments/infer_types_phase1/)
- **Goal**: Use static analysis tool to reduce 211 mypy errors
- **Methodological Issues**:
  - Tool only applied to `src/` directory
  - Baseline measurement scope unclear
  - Import placement errors not anticipated
- **Reported Results**: 211 → 10 errors (95% reduction)
- **Corrected Interpretation**: The 95% reduction is potentially inflated if baseline included `scripts/` and `tests/` directories that weren't processed. True effectiveness on `src/` alone was likely very high.

### 3. Automated Typing Tools Comparison
- **Location**: [completed_experiments/typing_tools_comparison/](completed_experiments/typing_tools_comparison/)
- **Goal**: Compare 11 different typing tool configurations
- **Methodological Issues**:
  - Baseline changed between planning (211) and execution (1046 errors)
  - Only 4 of 11 configurations tested
  - Tools applied only to `src/` directory
  - Measurement protocol inconsistently followed
- **Key Finding**: `autotyping --safe` was the only reliable tool, achieving 8-10% error reduction
- **Corrected Interpretation**: Despite incomplete execution, clearly identified that conservative approaches work while aggressive inference fails.

### 4. Round 2: Tool Stacking Experiment
- **Location**: [completed_experiments/monkeytype_rounds/](completed_experiments/monkeytype_rounds/)
- **Goal**: Test if typing tools stack additively on top of autotyping-safe
- **Methodological Issues**:
  - **Critical flaw**: Measured all directories but only applied tools to `src/`
  - 3 of 5 configurations failed due to invalid syntax
  - Started from autotyping baseline instead of clean state
- **Reported Results**: No additional improvement beyond autotyping-safe
- **Corrected Interpretation**: The experiment's conclusion that "tools don't stack" is invalid due to:
  - Only 2/5 configurations actually ran
  - Tools weren't applied to 14% of the measured codebase
  - Invalid tool configurations weren't corrected and retried

### 5. Round 3: MonkeyType Dynamic Analysis
- **Location**: [completed_experiments/monkeytype_rounds/](completed_experiments/monkeytype_rounds/)
- **Goal**: Break through static analysis ceiling using runtime type capture
- **Methodological Improvements**:
  - Combined static (autotyping) + dynamic (MonkeyType) approach
  - Correctly disabled pytest parallelization with `-n0`
  - Understood test coverage dependency
- **Results**: 
  - Static ceiling: 959 errors (autotyping-safe achieved -87)
  - With MonkeyType: 190-191 errors (additional -28 to -29)
  - Total: 81.7% error reduction
- **Key Success**: Proved that combining static and dynamic analysis breaks through automation limits

## Superseded/Incomplete Documents

### Phase 2 MonkeyType Plan (Superseded)
- **Location**: [superseded_plans/phase2_monkeytype_plan.md](superseded_plans/phase2_monkeytype_plan.md)
- **Issue**: Based on incorrect baseline of ~50-100 errors (actual was 395)
- **Why Superseded**: Round 3 experiments used correct methodology

### Remaining Error Analysis (Incomplete)
- **Location**: [superseded_plans/remaining_error_analysis.md](superseded_plans/remaining_error_analysis.md)
- **Issue**: Based on incorrect baseline; recommended manual fixes
- **Why Incomplete**: Discovery of automated tools made manual approach unnecessary

## Key Methodological Lessons Learned

1. **Always check all directories together** for accurate baseline measurements
2. **Apply tools to the same scope** as the measurement
3. **Verify tool syntax** before running experiments
4. **Test tools independently** before attempting to stack them
5. **Consider cross-directory effects** when interpreting results
6. **Document exact measurement commands** for reproducibility

## Corrected Understanding of Tool Effectiveness

Based on proper methodology:

1. **External Library Configuration**: 40-45% error reduction (fastest, highest impact)
2. **autotyping --safe**: 8-10% additional reduction (reliable, no side effects)
3. **MonkeyType (with good test coverage)**: 15-20% additional reduction (requires test execution)
4. **Combined Approach**: 80%+ total reduction achievable

## Future Work Recommendations

1. Re-run Round 2 with corrected tool syntax and full directory coverage
2. Test individual type inference extractors that were never evaluated
3. Explore tool combinations beyond autotyping + MonkeyType
4. Investigate why baseline measurements varied so dramatically (211 vs 1046)
5. Create automated pipeline incorporating all successful approaches

## Archive Structure

```
docs/
├── completed_experiments/
│   ├── mypy_parallel_fix/          # Early manual fixing attempt
│   ├── typing_tools_comparison/    # Automated tools evaluation
│   ├── infer_types_phase1/         # Initial infer-types success
│   └── monkeytype_rounds/          # Rounds 2-3 experiments
├── superseded_plans/               # Plans invalidated by discoveries
└── completed_tasks/                # Non-experimental documentation
```

All documents have been updated with methodological notes highlighting issues discovered during the retrospective analysis.