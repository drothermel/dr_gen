# Automated Typing Tools Comparison Experiment Results

## Configuration 1: autotyping-safe

### **Configuration Details**
- **Command**: `uv run autotyping --safe --none-return src/`
- **Tool Status**: ✅ **SUCCESS** (processed 29 files successfully)
- **Auto-fix Status**: ✅ **SUCCESS** (1 line length issue automatically resolved)

### **Quantitative Results**

#### **Mypy Errors**
- **Before**: 221 errors
- **After**: 197 errors  
- **Net Reduction**: 24 errors
- **Success Rate**: 10.9%

#### **Ruff Errors**
- **Before**: 0 errors (All checks passed)
- **After**: 0 errors (All checks passed)
- **Net Change**: 0 errors
- **Tool-Introduced Issues**: 1 E501 line length (auto-fixed)

### **Error Type Analysis**

#### **Before (Baseline)**
```
  90  Function is missing a type annotation  [no-untyped-def]
  42  Function is missing a return type annotation  [no-untyped-def]
  40  Function is missing a type annotation for one or more arguments  [no-untyped-def]
   5  Overlap between argument names and ** TypedDict items: "n_grid", "n_sample"  [misc]
   4  Overlap between argument names and ** TypedDict items: "n_sample"  [misc]
   2  Returning Any from function declared to return "list[int | float | str]"  [no-any-return]
   2  Returning Any from function declared to return "list[float]"  [no-any-return]
   2  Returning Any from function declared to return "float"  [no-any-return]
   2  Non-overlapping equality check (left operand type: "int", right operand type: "list[int]")  [comparison-overlap]
   2  Need type annotation for "curves" (hint: "curves: dict[<type>, <type>] = ...")  [var-annotated]
   [... 24 additional single-occurrence errors ...]
```

#### **After**
```
  72  Function is missing a type annotation for one or more arguments  [no-untyped-def]
  58  Function is missing a type annotation  [no-untyped-def]
  18  Function is missing a return type annotation  [no-untyped-def]
   5  Overlap between argument names and ** TypedDict items: "n_grid", "n_sample"  [misc]
   4  Overlap between argument names and ** TypedDict items: "n_sample"  [misc]
   2  Returning Any from function declared to return "list[int | float | str]"  [no-any-return]
   2  Returning Any from function declared to return "list[float]"  [no-any-return]
   2  Returning Any from function declared to return "float"  [no-any-return]
   2  Non-overlapping equality check (left operand type: "int", right operand type: "list[int]")  [comparison-overlap]
   2  Need type annotation for "curves" (hint: "curves: dict[<type>, <type>] = ...")  [var-annotated]
   [... same 24 additional single-occurrence errors ...]
```

#### **Key Changes**
- **"Function missing return type annotation"**: 42 → 18 (-24 errors) ✅
- **"Function missing type annotation"**: 90 → 58 (-32 errors) ✅  
- **"Function missing argument annotation"**: 40 → 72 (+32) - *Note: This appears to be a categorization shift, not new errors*
- **All other error types**: Unchanged

### **Qualitative Analysis**

#### **Annotations Added**
- Added `-> None` return type annotations to plotting functions
- Added `-> None` return type annotations to utility functions
- Added `-> None` return type annotations to method functions
- Conservative approach - only obvious void function annotations

#### **Code Quality Assessment**
- **High reliability** - No typing errors introduced
- **Clean execution** - No file corruption or tool failures
- **Conservative safety** - Only added unambiguous annotations
- **No side effects** - No import issues or formatting problems

### **Unexpected Issues**
1. **Installation required** - Had to install autotyping with `uv add --dev autotyping`
2. **Line length issue** - One E501 error created but automatically fixed by `ft`
3. **Error categorization shift** - Some errors may have been recategorized rather than eliminated

### **Overall Assessment**
**autotyping-safe** provides a **reliable, conservative foundation** with modest but meaningful impact. Excellent choice for initial automation step with minimal risk.

**Ranking**: Conservative/Safe tool - Low impact, High reliability

---

## Configuration 2: autotyping-aggressive

### **Configuration Details**
- **Command**: `uv run autotyping --aggressive src/`
- **Tool Status**: ✅ **SUCCESS** (processed 29 files successfully)
- **Auto-fix Status**: ⚠️ **PARTIAL** (1 unfixed ruff error, introduced new mypy errors)

### **Quantitative Results**

#### **Mypy Errors**
- **Before**: 221 errors
- **After**: 223 errors  
- **Net Change**: +2 errors
- **Success Rate**: -0.9% (made errors worse)

#### **Ruff Errors**
- **Before**: 0 errors (All checks passed)
- **After**: 1 error (ARG001: Unused function argument)
- **Net Change**: +1 error
- **Tool-Introduced Issues**: Multiple type compatibility issues

### **Error Type Analysis**

#### **Key Changes (Before → After)**
- **Missing return type annotations**: 42 → 34 (-8) ✅
- **Missing argument annotations**: 40 → 85 (+45) ❌
- **Missing function annotations**: 90 → 42 (-48) ✅

#### **New Error Types Introduced**
- **Type compatibility errors**: 6 new arg-type errors
  - `Argument "figsize" has incompatible type` (4 instances)
  - `Argument "plot_size" has incompatible type` (2 instances)
- **Union type operation errors**: 3 new operator errors
  - `Unsupported operand types for < ("int" and "None")`
  - `Cannot determine type of "fs_y"` and `fs_x"`
- **Generic type errors**: New call-overload and misc errors

### **Problem Analysis**

#### **What Went Wrong**
1. **Overly aggressive parameter typing**: Added `int | None` types that don't match actual usage
2. **Incorrect type inference**: Applied `int` type to `figsize` parameter that expects tuples
3. **Created type incompatibilities**: New annotations conflicted with existing function signatures

#### **Examples of Bad Annotations**
```python
# BEFORE (working)
def get_subplot_axis(ax=None, figsize=None):

# AFTER (broken) 
def get_subplot_axis(ax=None, figsize: int | None = None):
# Problem: figsize should be tuple, not int
```

### **Code Quality Assessment**
- **Introduced errors**: Created 8+ new mypy errors through incorrect type inference
- **Type mismatches**: Applied wrong types based on insufficient context analysis
- **Broke working code**: Added annotations that made previously working functions fail type checking

### **Unexpected Issues**
1. **Negative ROI**: Tool made codebase worse, not better
2. **Poor type inference**: Aggressive mode lacks sufficient context for complex types
3. **Function signature conflicts**: New parameter types incompatible with actual usage

### **Overall Assessment**
**autotyping-aggressive is counterproductive** - the aggressive inference creates more problems than it solves. The tool's heuristics are insufficient for complex typing scenarios and introduce incorrect type annotations.

**Ranking**: Aggressive/Risky tool - Negative impact, Low reliability

**Recommendation**: Avoid aggressive mode; stick with safe mode only.

---

## Configuration 3: infer-types-default

### **Configuration Details**
- **Command**: `uv run python -m infer_types src/`
- **Tool Status**: ✅ **SUCCESS** (processed 29 files successfully)
- **Auto-fix Status**: ❌ **FAILED** (multiple ruff errors persisted, mypy errors increased)

### **Quantitative Results**

#### **Mypy Errors**
- **Before**: 221 errors
- **After**: 240 errors  
- **Net Change**: +19 errors
- **Success Rate**: -8.6% (made errors significantly worse)

#### **Ruff Errors**
- **Before**: 0 errors (All checks passed)
- **After**: 1 error (multiple E402, F821, E501 errors detected)
- **Net Change**: +1 error (with 12 total issues reported)
- **Tool-Introduced Issues**: Import placement errors, undefined types, line length

### **Error Type Analysis**

#### **Tool-Generated Problems**
1. **Import placement errors (E402)**: Multiple `from typing import Any` statements placed mid-file
2. **Undefined type reference (F821)**: Used `ndarray` without importing from numpy
3. **Line length violations (E501)**: Generated overly long function signatures
4. **Duplicate imports (F811)**: Created redundant import statements

### **Problem Analysis**

#### **What Went Wrong**
1. **Poor import management**: Added imports in wrong locations, violating Python conventions
2. **Incomplete type resolution**: Referenced types without proper imports
3. **Code structure violations**: Broke existing file organization
4. **Type annotation quality**: Generated inconsistent and problematic annotations

#### **Examples of Bad Annotations**
```python
# Added mid-file (violates E402):
def calc_stats_across_bootstrap_samples(timestep_data) -> dict:
    # ... function body ...
    return stats

from typing import Any  # ← WRONG LOCATION

def summarize_distribution(dist) -> dict[str, Any] | dict:
```

```python  
# Undefined type reference (F821):
def calc_ks_stat_and_summary(bdata_a, bdata_b, num_bootstraps) -> dict[str, ndarray]:
#                                                                             ^^^^^^^ 
# ndarray not imported
```

### **Code Quality Assessment**
- **Significantly degraded code quality**: Introduced structural problems
- **Import violations**: Multiple E402 errors from poor import placement
- **Type errors**: Undefined type references
- **Formatting issues**: Line length violations

### **Unexpected Issues**
1. **Severe reliability problems**: Tool broke basic Python conventions
2. **Import management failure**: Cannot properly handle existing import structure
3. **Type resolution issues**: References types without importing them
4. **Negative ROI**: Made codebase significantly worse across all metrics

### **Overall Assessment**
**infer-types-default is highly problematic** - the tool fundamentally misunderstands Python import conventions and generates broken code. It creates more problems than autotyping-aggressive and should not be used in production environments.

**Ranking**: Default tool - Significantly negative impact, Very low reliability

**Recommendation**: ❌ **STRONGLY NOT RECOMMENDED** - breaks code structure and conventions

---
