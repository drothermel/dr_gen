# Phase 2: Runtime Type Collection with MonkeyType - Detailed Execution Plan

## Pre-Execution State
- **Expected starting point**: ~50-100 mypy errors (after Phase 1 `infer-types`)
- **Hypothesis**: MonkeyType can capture complex ML pipeline types that static analysis missed
- **Target**: Additional 60% reduction on remaining errors (~50 → ~20 errors)

## MonkeyType Strategy for ML Codebase

### Key Advantages for Our Use Case
- **ML Pipeline Types**: Can capture complex `torch.Tensor`, `DataLoader`, `Dataset` types from actual usage
- **Generic Types**: Runtime data shows actual generic parameters (e.g., `list[float]` vs `list[Any]`)
- **Union Types**: Discovers when functions actually handle multiple types in practice
- **Test Coverage**: Our codebase likely has training/evaluation test coverage

## Execution Steps

### Step 1: Install MonkeyType
```bash
uv add --dev monkeytype
```

**Expected outcome**: Tool added to dev dependencies
**Requirements**: Python 3.9+ (✅ we use 3.12), libcst (should install automatically)

### Step 2: Baseline Measurement
```bash
ckdr
```

**Purpose**: Confirm Phase 1 results before Phase 2
**Expected state**: ~50-100 errors remaining after `infer-types`

### Step 3: Run Test Suite Under MonkeyType Tracing
```bash
# Check if we have tests
ls tests/ 

# Run test suite under MonkeyType tracing  
uv run monkeytype run pytest
```

**Alternative if no pytest**:
```bash
# Run training script under tracing
uv run monkeytype run python scripts/train.py --help

# Or run specific modules
uv run monkeytype run -m dr_gen.train.loops
```

**What this does**:
- Instruments all function calls during execution
- Records argument types and return types
- Stores data in `monkeytype.sqlite3` database
- Captures real runtime type usage patterns

### Step 4: Explore Collected Data
```bash
# List modules that have trace data
uv run monkeytype list-modules

# Generate stub for specific high-error modules  
uv run monkeytype stub dr_gen.utils.metrics
uv run monkeytype stub dr_gen.analyze.common_plots
uv run monkeytype stub dr_gen.train.loops
```

**Purpose**: Preview what MonkeyType discovered before applying changes

### Step 5: Apply Runtime-Based Annotations
```bash
# Apply annotations to high-impact modules
uv run monkeytype apply dr_gen.utils.metrics
uv run monkeytype apply dr_gen.analyze.common_plots  
uv run monkeytype apply dr_gen.train.loops
uv run monkeytype apply dr_gen.data.load_data
```

**What this does**:
- Modifies source files in-place with runtime-discovered types
- Only adds annotations where none exist (preserves existing annotations)
- Uses actual observed types from execution

### Step 6: Immediate Verification
```bash
ckdr
```

**Purpose**: Measure MonkeyType's impact on remaining errors
**Expected outcome**: Additional error reduction from runtime type data

### Step 7: Manual Review and Cleanup
```bash
# Review MonkeyType changes
git diff

# Check for overly specific types that need generalization
# Example: List[int] → Sequence[int], torch.Tensor specifics
```

**Important**: MonkeyType documentation warns that generated annotations are "a starting point" and may need adjustment for overly specific types.

### Step 8: Error Analysis
```bash
ckdr 2>&1 | grep "error:" | cut -d: -f4- | sort | uniq -c | sort -nr > phase2_remaining_errors.txt

# Compare Phase 1 vs Phase 2 results
diff phase1_remaining_errors.txt phase2_remaining_errors.txt
```

**Purpose**: Document which additional errors MonkeyType resolved

### Step 9: Git Commit for Phase 2
```bash
git add .
git commit -m "Phase 2: Apply MonkeyType runtime-based type annotations

- Collected runtime types from test suite execution
- Applied annotations based on observed type usage  
- Additional reduction from X to Y errors
- Focus on ML pipeline types and complex generics

🤖 Generated with MonkeyType runtime analysis
"
```

### Step 10: Results Documentation
Create summary in `phase2_results.md`:
- Error reduction metrics (Phase 1 result → Phase 2 result)
- Types of annotations MonkeyType successfully added
- Comparison with `infer-types` results
- Quality assessment of runtime-discovered types

## Success Criteria

### Quantitative Targets
- **Minimum success**: 30% additional reduction on remaining errors
- **Expected success**: 60% additional reduction (50 → 20 errors)
- **Optimistic success**: 80% additional reduction (50 → 10 errors)

### Qualitative Assessment
- **Type accuracy**: Runtime types should be more precise than static inference
- **ML-specific types**: Should discover `torch.Tensor`, `DataLoader`, etc.
- **Generic parameters**: Should fill in `Any` with actual observed types

## Risk Mitigation

### Potential Issues
1. **No test coverage**: Limited runtime data if tests don't exercise code paths
2. **Overly specific types**: May generate `List[int]` instead of `Sequence[int]`
3. **Performance impact**: Tracing may be slow on large test suites

### Mitigation Strategies
1. **Alternative execution**: Run training scripts if tests insufficient
2. **Manual review**: Generalize overly specific types after application
3. **Selective application**: Target specific modules with highest error counts

## Expected Timeline
- **Tool installation**: 30 seconds
- **Test suite execution under tracing**: 2-5 minutes  
- **Type application**: 1-2 minutes
- **Verification and review**: 2-3 minutes
- **Documentation**: 2-3 minutes
- **Total**: 8-13 minutes

## MonkeyType Limitations to Watch For

1. **Starting point only**: "MonkeyType annotations are rarely suitable exactly as generated"
2. **Overly concrete**: May suggest `List[int]` where `Sequence[int]` is better
3. **Execution-dependent**: Only captures types seen during traced runs
4. **Function scope**: Can target specific functions if full module is too broad

## Ready for Execution

This plan provides:
✅ **Accurate MonkeyType syntax** based on web search research  
✅ **ML-specific considerations** for our training pipeline codebase  
✅ **Flexible execution options** if test coverage is insufficient  
✅ **Quality control** with manual review expectations  
✅ **Clear success metrics** and risk mitigation

**This plan will execute after Phase 1 completion.**