<!-- METHODOLOGICAL NOTES
- Baseline: Unclear which directories were checked (likely all based on 395 errors)
- Tool application: Manual MultiEdit approach on identified files
- Error counting: Assumed errors were additive across files (now known to be incorrect)
- Results validity: FLAWED - mypy errors are context-dependent and non-additive
- Key issue: File-by-file error distribution used for load balancing doesn't account for cross-file dependencies
-->

# MyPy Parallel Fix Planning Process

## Initial Assessment

### Error Discovery
- Ran `ckdr` and found **395 mypy errors across 16 files**
- All ruff/format checks passed - this is purely a typing issue
- No lint conflicts to worry about during the fixing process

### Error Distribution Analysis

**File-Level Error Counts (estimated from output):**
- `utils/metrics.py`: ~50+ errors (heaviest file)
- `utils/display.py`: ~40+ errors (second heaviest)
- `analyze/ks_stats.py`: ~30+ errors
- `analyze/common_plots.py`: ~25+ errors
- `train/loops.py`: ~25+ errors
- `data/load_data.py`: ~20+ errors
- `train/model.py`: ~10+ errors
- `schemas.py`: ~5+ errors
- `utils/utils.py`: ~1-2 errors

**Key Insight**: Heavy concentration in utils/ and analyze/ modules, with substantial work in train/ and data/ modules.

## Error Type Categorization

### Mechanical Fixes (Low Complexity - ~40% of errors)
**Pattern**: Direct, unambiguous type additions
- `no-untyped-def` for simple functions → Add `-> None` for procedures
- `no-untyped-def` with obvious return types → Add `-> int`, `-> str`, `-> bool`
- Basic parameter types when usage is clear from context

**Justification**: These require no domain knowledge, just pattern matching. Can be batched efficiently.

### Semi-Contextual Fixes (Medium Complexity - ~35% of errors)
**Pattern**: Requires code reading but follows patterns
- `no-untyped-def` needing parameter type inference from function body
- `no-untyped-call` → Decision between typing the called function vs `# type: ignore`
- `attr-defined` → Check if legitimate issue or mypy configuration problem
- `no-any-return` where return path is traceable but not obvious

**Justification**: These need some understanding but follow predictable patterns. Can be grouped by error type.

### Contextual Fixes (High Complexity - ~25% of errors)
**Pattern**: Requires deep code understanding
- Complex `no-any-return` requiring full code flow analysis
- `arg-type` mismatches needing domain knowledge of the ML training pipeline
- `union-attr` errors requiring understanding of conditional logic
- `return` errors needing comprehensive path analysis
- Generic type parameters and complex inheritance issues

**Justification**: These need to stay with file context to maintain coherent understanding of the codebase's ML training logic.

## Load Balancing Strategy

### Round-Robin Assignment Rationale
Applied the position paper's round-robin strategy:

1. **Sorted files by error count (descending)**:
   - A: `utils/metrics.py` (~50)
   - B: `utils/display.py` (~40) 
   - C: `analyze/ks_stats.py` (~30)
   - D: `analyze/common_plots.py` (~25)
   - E: `train/loops.py` (~25)
   - F: `data/load_data.py` (~20)
   - G: `train/model.py` (~10)
   - H: `schemas.py` (~5)
   - I: `utils/utils.py` (~2)

2. **Round-robin distribution**:
   - **Agent 1**: A(50) + E(25) + I(2) = **77 errors**
   - **Agent 2**: B(40) + F(20) = **60 errors**
   - **Agent 3**: C(30) + G(10) = **40 errors**
   - **Agent 4**: D(25) + H(5) = **30 errors**

**Justification**: This achieves reasonable load balancing while maintaining file coherence. Agent 1 gets the heaviest file but balanced with smaller ones.

## Phase Design

### Phase 1: Parallel Analysis & Generation
**Rationale**: Following the position paper's output-only approach to prevent coordination failures.

**Agent Constraints**:
- **CRITICAL**: Explicit output-only instructions using the proven template
- NO file modification tools allowed
- Generate structured output for later batch application
- Categorize each fix by complexity level

**Output Format**:
```markdown
## File: [filename]

### Mechanical Fixes
- Line X: Function `foo()` → Add `-> None`
- Line Y: Parameter `bar` → Add `: int`

### Semi-Contextual Fixes  
- Line Z: Function `baz()` → Need to analyze usage, likely `-> list[str]`

### Contextual Fixes
- Line W: Complex return type analysis needed for ML metrics aggregation
```

### Phase 2: Coordinated Application
**Rationale**: Apply fixes in complexity order to catch issues early.

1. **Mechanical fixes first**: Low risk, high confidence
2. **Semi-contextual fixes**: Moderate risk, pattern-based
3. **Contextual fixes**: High risk, requires careful review

Use MultiEdit for atomic file changes to maintain consistency.

### Phase 3: Verification & Iteration
**Rationale**: Progressive verification allows course correction.

- Run `ckdr` after each complexity category
- Identify patterns in remaining errors
- Launch targeted fixes for residual issues

## Expected Challenges & Mitigations

### Challenge 1: ML Domain Complexity
**Issue**: This codebase has deep ML/training concepts that affect typing
**Mitigation**: Keep contextual fixes with their file context, leverage existing type hints in related functions

### Challenge 2: External Library Typing
**Issue**: `timm`, `torch`, etc. may have incomplete type hints
**Mitigation**: Use `# type: ignore` strategically rather than over-typing

### Challenge 3: Generic Types in ML Pipelines
**Issue**: Dataset, DataLoader, Metrics classes use complex generics
**Mitigation**: Start with concrete types, evolve to generics only when necessary

## Success Metrics

### Quantitative Goals
- **Target**: 395 → <50 errors (87%+ reduction)
- **Time**: 8-12 minutes (vs ~90 minutes sequential)
- **Success Rate**: >85% of errors resolved without manual intervention

### Qualitative Goals
- Maintain code readability and ML pipeline clarity
- No introduction of type errors that break functionality
- Preserve existing type annotations and patterns

## Key Learnings Expected

1. **ML Codebase Specifics**: How training pipelines affect typing strategies
2. **External Library Integration**: Best practices for typing around `timm`/`torch`
3. **Complexity Categorization**: Whether our 40/35/25 split holds for typing errors
4. **Tool Integration**: How mypy errors differ from ruff/lint errors in parallel fixing

## Retrospective Framework

Post-execution analysis will examine:
- **Accuracy of error categorization**: Did complexity estimates match reality?
- **Load balancing effectiveness**: Were agents optimally utilized?
- **Output-only compliance**: Did agents follow instructions correctly?
- **Time efficiency**: Actual vs predicted timing
- **Quality of fixes**: Type safety improvements vs readability trade-offs

This planning document serves as our baseline for measuring the effectiveness of applying parallel fixing principles to mypy type errors in a real ML codebase.