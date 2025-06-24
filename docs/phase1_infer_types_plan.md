# Phase 1: Static Inference with `infer-types` - Detailed Execution Plan

## Pre-Execution State
- **Current errors**: 211 mypy errors (after external library config)
- **Error breakdown**: 165 mechanical + 34 semi-contextual + 12 contextual
- **Hypothesis**: `infer-types` can handle both mechanical and semi-contextual errors (~199 errors)
- **Expected outcome**: 211 → ~50 errors (75% reduction)

## Execution Steps

### Step 1: Install `infer-types`
```bash
uv add --dev infer-types
```

**Expected outcome**: Tool added to dev dependencies in `pyproject.toml`

### Step 2: Baseline Measurement
```bash
ckdr
```

**Purpose**: Confirm starting point of 211 errors before any changes
**Expected outcome**: Verify 211 mypy errors across 15 files

### Step 3: Run `infer-types` on Source Code
```bash
uv run python -m infer_types src/
```

**What this does**:
- Analyzes all Python files in `src/` directory
- Uses typeshed to understand external library types
- Applies sophisticated heuristics for type inference
- Modifies files in-place with inferred type annotations
- Handles inheritance, function name patterns, and contextual inference

**Expected changes**:
- Add `-> None` to void functions (42 errors)
- Add return type annotations based on context (e.g., `len()` → `int`)
- Add parameter type annotations based on usage patterns
- Copy annotations from base classes in inheritance hierarchies
- Infer `Iterator` types for generator functions

### Step 4: Immediate Verification
```bash
ckdr
```

**Purpose**: Measure impact of `infer-types` transformation
**Expected outcome**: Significant error reduction from 211 to ~50-100 errors

### Step 5: Post-Processing (if needed)
```bash
# Fix import organization (infer-types may add duplicate imports)
uv run python -m isort src/

# Add future annotations for Python <3.10 compatibility (our project uses 3.12)
# This step may not be needed, but good to know about
```

**Purpose**: Clean up any import issues from infer-types transformation

### Step 6: Error Analysis and Documentation
```bash
ckdr 2>&1 | grep "error:" | cut -d: -f4- | sort | uniq -c | sort -nr > phase1_remaining_errors.txt
```

**Purpose**: Categorize and count remaining error types
**Analysis**: Document which types of errors `infer-types` successfully handled vs. what remains

### Step 7: Git Commit for Phase 1
```bash
git add .
git commit -m "Phase 1: Apply infer-types automatic type annotations

- Reduced mypy errors from 211 to X
- Added return type annotations for void functions  
- Inferred parameter types from usage context
- Applied inheritance-based type annotations

🤖 Generated with infer-types tool
"
```

**Purpose**: Create checkpoint before Phase 2 (MonkeyType)

### Step 8: Results Documentation
Create summary in `phase1_results.md`:
- Error reduction metrics (211 → X)
- Success rate by error category
- Types of annotations successfully added
- Analysis of remaining error patterns
- Assessment for Phase 2 planning

## Success Criteria

### Quantitative Targets
- **Minimum success**: 30% error reduction (211 → 147 errors)
- **Expected success**: 75% error reduction (211 → 53 errors)  
- **Optimistic success**: 85% error reduction (211 → 32 errors)

### Qualitative Assessment
- **High-quality annotations**: Types should be accurate and helpful
- **No new errors**: Tool shouldn't introduce type inconsistencies
- **Code readability**: Annotations should improve code clarity

## Risk Mitigation

### Potential Issues
1. **Tool failure**: `infer-types` might not run on our codebase
2. **Type errors**: New annotations might introduce mypy conflicts
3. **Over-annotation**: Tool might add incorrect or overly specific types

### Mitigation Strategies
1. **Backup**: Git checkpoint before running tool
2. **Incremental verification**: Check errors immediately after tool run
3. **Rollback plan**: `git reset --hard HEAD~1` if tool causes issues

## Time Estimate
- **Tool installation**: 30 seconds
- **Tool execution**: 1-2 minutes
- **Verification and analysis**: 2-3 minutes
- **Documentation**: 2-3 minutes
- **Total**: 5-8 minutes

## Ready for Execution

This plan provides:
✅ **Clear steps** with specific commands  
✅ **Expected outcomes** for each step  
✅ **Measurement strategy** for success assessment  
✅ **Risk mitigation** with backup plan  
✅ **Time bounds** for efficient execution

**Awaiting approval to proceed with Phase 1 execution.**