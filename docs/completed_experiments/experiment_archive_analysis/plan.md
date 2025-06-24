# Experiment Archive Analysis & Documentation Cleanup Task

## Context

Over the course of running automated typing experiments in this repository, we discovered significant methodological issues that affected our measurements and conclusions. Most critically, we learned that mypy error counts are **not additive** across directories - checking `src/`, `scripts/`, and `tests/` together produces different (higher) error counts than the sum of checking them individually due to cross-directory type dependencies.

## Your Mission

Analyze all experimental documents in the `docs/` directory to:
1. Identify methodological issues in past experiments
2. Flag these issues in the documents before archiving
3. Organize documents to clearly show plan → results → retrospective relationships
4. Mark superseded plans appropriately

## Key Methodological Issues to Check For

### 1. **Directory Coverage**
- ❌ **Bad**: Only checking `src/` directory for errors
- ❌ **Bad**: Running autotyping only on `src/` when baseline includes all directories
- ✅ **Good**: Checking `src scripts tests` together for true baseline
- ✅ **Good**: Applying tools to all directories that are included in baseline

### 2. **Error Measurement Consistency**
- ❌ **Bad**: Comparing individual directory counts to combined baseline
- ❌ **Bad**: Using different mypy commands for baseline vs. post-treatment
- ✅ **Good**: Using `mp src scripts tests` consistently for all measurements
- ✅ **Good**: Using the same error extraction method throughout

### 3. **Cross-Directory Effects**
- ❌ **Bad**: Not accounting for cascade effects (fixing src/ reduces test errors)
- ❌ **Bad**: Treating error counts as simple arithmetic
- ✅ **Good**: Understanding that fixing types in imported modules has ripple effects
- ✅ **Good**: Measuring combined directory impact

### 4. **Tool Application Scope**
- ❌ **Bad**: Applying typing tools to subset of directories in baseline
- ✅ **Good**: Applying tools to same directory set used for measurement

## Documents to Analyze

### Plans with Results & Retrospectives:
1. `mypy_parallel_fix_plan.md` → `mypy_parallel_fix_retrospective.md`
2. `phase1_infer_types_plan.md` → `phase1_results.md`
3. `automated_typing_experiment_plan.md` → `automated_typing_experiment_results.md`
4. `round2_experiment_plan.md` → `round2_experiment_results.md`
5. `round3_monkeytype_plan.md` → `round3_experiment_results.md`

### Superseded/Incomplete Plans:
1. `phase2_monkeytype_plan.md` (superseded by round3)
2. `remaining_error_analysis.md` (point-in-time snapshot)

### Completed Tasks:
1. `lint_fixes_todo.md`

## Required Actions

### 1. For Each Experiment Set:
- Read the plan, results, and retrospective (if exists)
- Identify any methodological issues based on our current understanding
- Add a "METHODOLOGICAL NOTES" section to the top of each document highlighting:
  - What measurement approach was used
  - Any issues that might affect validity
  - Whether results should be considered preliminary/flawed

Example header to add:
```markdown
<!-- METHODOLOGICAL NOTES
- Baseline: Only checked src/ directory (should have included scripts/ tests/)
- Tool application: Correctly applied to same scope as measurement
- Results validity: FLAWED - undercounts true baseline by ~600 errors
-->
```

### 2. Create Archive Structure:
```
docs/
├── completed_experiments/
│   ├── mypy_parallel_fix/
│   │   ├── plan.md
│   │   └── retrospective.md
│   ├── typing_tools_comparison/
│   │   ├── plan.md
│   │   ├── results.md
│   │   └── round1_results.md
│   ├── infer_types_phase1/
│   │   ├── plan.md
│   │   └── results.md
│   └── monkeytype_rounds/
│       ├── round2_plan.md
│       ├── round2_results.md
│       ├── round3_plan.md
│       └── round3_results.md
├── superseded_plans/
│   ├── phase2_monkeytype_plan.md
│   └── remaining_error_analysis.md
└── completed_tasks/
    └── lint_fixes_todo.md
```

### 3. Create Summary Document:
Create `docs/experiment-archive-summary.md` that:
- Lists all experiments chronologically
- Notes which had methodological issues
- Provides corrected interpretations where possible
- Links to the archived documents

## Reference Documents

Use these to understand correct methodology:
- `docs/mypy-reference.md` - Section on "Context-Dependent Error Counting"
- `docs/typing-experiment-status.md` - Current understanding of proper measurement
- `check_types.sh` - Tool that correctly measures individual vs. combined errors

## Important Context

The discovery of non-additive error counting fundamentally changes how we interpret earlier results. Many experiments that showed "modest" improvements may have actually had larger impacts when cascade effects are considered. Conversely, some baselines may have been incorrectly low if they only measured `src/` directory.

## Deliverables

1. All experiment documents updated with methodological notes
2. Documents organized into clear archive structure
3. `experiment-archive-summary.md` providing overview and corrections
4. Clear marking of superseded documents

This cleanup will ensure future agents and developers understand both what was learned and what methodological improvements were discovered along the way.