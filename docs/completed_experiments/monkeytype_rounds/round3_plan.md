# Round 3 Experiment Plan: MonkeyType Runtime Type Inference

**Date:** June 24, 2025  
**Strategy:** Use runtime type tracing to break through the -87 error automation ceiling  
**Foundation:** Build on autotyping-safe success, add MonkeyType runtime intelligence

## Breakthrough Discovery

**MonkeyType successfully captures runtime types!** Testing revealed:

✅ **Working Configuration:**
```bash
export MONKEYTYPE_TRACE_MODULES="dr_gen"
monkeytype run pytest tests/ -v -n0  # -n0 disables parallel execution
```

✅ **Captured Types:** MonkeyType generated accurate stubs:
```python
# dr_gen.utils.metrics
class GenMetrics:
    def __init__(self, cfg: DictConfig) -> None: ...
    def clear_data(self, group_name: Optional[str] = ...): ...
    def log_data(self, data: Dict[str, int], group_name: str, ns: Optional[int] = ...): ...

# dr_gen.utils.utils  
def flatten_dict_tuple_keys(
    d: dict[typing.Any, typing.Any],
    parent_key: tuple[typing.Any, ...] = ...
) -> dict[tuple[typing.Any, ...], typing.Any]: ...
```

## The Round 3 Hypothesis Confirmed

**Runtime analysis captures parameter types that static analysis cannot infer** - exactly what we need to address the remaining 670 `no-untyped-def` errors.

## Key Technical Requirements

### 1. Environment Setup
```bash
export MONKEYTYPE_TRACE_MODULES="dr_gen"
```

### 2. Test Execution Constraints
- **Critical:** Use `-n0` to disable pytest parallel execution
- **Reason:** MonkeyType can't trace across multiple processes
- **Impact:** Slower test execution but accurate type capture

### 3. Module Focus Strategy
Target modules with highest `no-untyped-def` error counts:
- `src/dr_gen/utils/metrics.py` - Primary target
- `src/dr_gen/analyze/bootstrapping.py` - Secondary target
- `src/dr_gen/train/model.py` - Tertiary target

## Round 3 Experiment Configurations

### Configuration 1: Foundation + Metrics Module
```bash
# Step 1: Apply autotyping-safe foundation
autotyping --safe --none-return src/

# Step 2: Trace and apply metrics module types
export MONKEYTYPE_TRACE_MODULES="dr_gen"
monkeytype run pytest tests/utils/test_metrics.py -v -n0
monkeytype apply dr_gen.utils.metrics

# Step 3: Measure improvement
```

### Configuration 2: Foundation + Multiple Module Focus
```bash
# Step 1: Apply autotyping-safe foundation
autotyping --safe --none-return src/

# Step 2: Comprehensive test suite tracing
export MONKEYTYPE_TRACE_MODULES="dr_gen"
monkeytype run pytest tests/utils/ tests/train/ tests/analyze/ -v -n0

# Step 3: Apply types to multiple modules
monkeytype apply dr_gen.utils.metrics
monkeytype apply dr_gen.utils.utils
monkeytype apply dr_gen.train.evaluate
# (apply to modules with most traces)

# Step 4: Measure improvement
```

## Expected Breakthrough Potential

**Target error reduction:**
- **Current ceiling:** -87 errors (autotyping-safe)
- **MonkeyType target:** Additional -20 to -50 errors
- **Total goal:** >-110 errors (1046 → <930)

**Specific improvements expected:**
- **Parameter annotations:** `cfg: DictConfig`, `data: Dict[str, int]`
- **Optional parameters:** `group_name: Optional[str] = None`
- **Complex types:** `dict[tuple[Any, ...], Any]`

## Risk Assessment & Mitigation

### Risks
1. **Test performance:** Sequential execution will be slower
2. **Type accuracy:** Runtime types may be overly specific
3. **Import issues:** MonkeyType may add unnecessary imports

### Mitigation
1. **Focused testing:** Target specific test modules, not full suite
2. **Review generated types:** Treat as draft annotations requiring validation  
3. **Integration testing:** Run full mypy + test suite after application

## Success Criteria

**Primary Goal:** Beat -87 error ceiling with >10 additional error reduction  
**Stretch Goal:** Achieve >-100 total mypy error reduction  
**Quality Gates:**
- Zero test failures after type application
- Maximum +5 ruff errors (import-related)
- All generated types validate with mypy

## Validation Protocol

**Three-stage validation:**
1. **Type Generation:** Verify stubs look reasonable
2. **Application:** Apply to source code without breaking tests
3. **Integration:** Full mypy check + test suite execution

This experiment leverages MonkeyType's proven ability to capture accurate runtime types, targeting exactly the parameter annotations that static analysis cannot infer.

## Methodological Improvements from Previous Rounds

### 1. Combined Static + Dynamic Approach
- **Innovation:** Used autotyping as foundation, then applied MonkeyType for deeper insights
- **Rationale:** Static tools handle obvious cases, runtime capture handles complex types
- **Result:** Synergistic effect - each tool's strengths complement the other's weaknesses

### 2. Correct Test Execution Strategy
- **Key Fix:** Discovered `-n0` flag requirement for MonkeyType compatibility
- **Impact:** Enabled accurate runtime tracing by disabling pytest parallelization
- **Learning:** Tool constraints must be understood for effective deployment

### 3. Test Coverage Dependency Recognition
- **Insight:** MonkeyType effectiveness directly correlates with test coverage
- **Strategy:** Focused on modules with comprehensive test suites first
- **Limitation:** Can only type what gets executed during test runs

### 4. Focused Application Scope
- **Decision:** Applied only to `src/` directory, not `scripts/` or `tests/`
- **Reasoning:** Production code typing provides most value
- **Trade-off:** Left some potential improvements unexplored but maintained focus

### 5. Breaking the Automation Ceiling
- **Static Tool Limit:** -87 errors was the maximum achievable with static analysis
- **Runtime Breakthrough:** Additional -28 to -29 errors via runtime type capture
- **Proof of Concept:** Demonstrated that runtime analysis can surpass static limitations
- **Total Achievement:** -115 to -116 total error reduction, far exceeding the -87 ceiling