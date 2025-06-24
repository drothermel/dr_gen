# MonkeyType Reference: Practical Guide for Type Automation

**Purpose:** Runtime type inference to complement static typing tools  
**Key Strength:** Captures parameter types that static analysis cannot infer

## Critical Setup Requirements

### 1. Environment Variable (Essential)
```bash
export MONKEYTYPE_TRACE_MODULES="your_package_name"
```
**Why:** Without this, MonkeyType excludes site-packages and may capture nothing  
**Format:** Comma-separated list: `"package1,package2.module"`

### 2. Sequential Execution (Essential for pytest)
```bash
monkeytype run pytest tests/ -v -n0  # -n0 disables parallel execution
```
**Why:** MonkeyType cannot trace across multiple processes  
**Impact:** Slower tests but accurate type capture

## Basic Workflow

```bash
# 1. Trace runtime execution
export MONKEYTYPE_TRACE_MODULES="mypackage"
monkeytype run pytest tests/ -v -n0

# 2. Verify traces were captured
monkeytype list-modules

# 3. Generate stub file (preview)
monkeytype stub mypackage.module

# 4. Apply types to source code
monkeytype apply mypackage.module
```

## Integration with Static Tools

**Ideal pattern:** Static foundation + MonkeyType enhancement
```bash
# Step 1: Apply static typing (safe baseline)
autotyping --safe --none-return src/

# Step 2: Add runtime intelligence
export MONKEYTYPE_TRACE_MODULES="mypackage" 
monkeytype run pytest tests/ -v -n0
monkeytype apply mypackage.module
```

**Why this works:** 
- Static tools handle obvious cases (return types)
- MonkeyType captures complex cases (parameter types)
- By default, MonkeyType respects existing annotations

## Common Issues & Solutions

### Empty Database (No Traces)
**Symptoms:** `sqlite3 monkeytype.sqlite3 "SELECT COUNT(*) FROM monkeytype_call_traces;"` returns 0

**Causes & Solutions:**
- **Missing env var:** Set `MONKEYTYPE_TRACE_MODULES="your_package"`
- **Parallel execution:** Add `-n0` to pytest
- **Script vs module:** MonkeyType can't trace the main script, only imported modules

### Performance Considerations
- **Overhead:** ~20x slower execution during tracing
- **Mitigation:** Trace only focused test subsets, not full suites
- **Database:** SQLite file grows with traces (safe to delete between runs)

### Type Quality
- **Strength:** Captures actual runtime types with high accuracy
- **Limitation:** May be overly specific (e.g., `List[int]` vs `Sequence[int]`)
- **Best practice:** Treat generated annotations as high-quality drafts requiring review

## Advanced Configuration

### Focused Module Tracing
```bash
# Target specific modules for better performance
export MONKEYTYPE_TRACE_MODULES="mypackage.utils,mypackage.models"
monkeytype run pytest tests/utils/ tests/models/ -v -n0
```

### Annotation Behavior
```bash
# Default: respect existing annotations
monkeytype apply mypackage.module

# Override existing annotations (use cautiously)
monkeytype apply --ignore-existing-annotations mypackage.module
```

## Value Proposition

**MonkeyType excels at:**
- Parameter type inference (`def func(cfg: DictConfig, data: Dict[str, int])`)
- Optional parameter detection (`param: Optional[str] = None`)
- Complex return types (`dict[tuple[Any, ...], Any]`)
- Types that static analysis cannot safely infer

**Limitations:**
- Requires good test coverage to see all code paths
- Performance overhead during tracing
- May miss edge cases not exercised in tests

## Integration Strategy

**For existing codebases:**
1. Start with static tools for safe, obvious improvements
2. Use MonkeyType to capture parameter types from test execution  
3. Validate with mypy and test suite
4. Apply incrementally, module by module

**Success pattern:** Static + Dynamic = Maximum safe automation