# Round 2 Typing Tool Experiment Plan

## Hypothesis
**Incremental typing improvements can be stacked** - building upon autotyping-safe's success with additional conservative typing annotations.

## Strategy
**Two-stage approach:** Apply autotyping-safe first (proven winner), then test additional tools on the improved codebase.

## Baseline for Round 2
- **Start with:** autotyping-safe results (959 mypy errors after -87 improvement)
- **Remaining errors:** 670 `no-untyped-def`, 97 `var-annotated`, 56 `no-any-return`, etc.
- **Clean slate:** 0 ruff errors

## Experiment Configurations

### Stage 1: Enhanced autotyping (building on --safe --none-return)

**1. autotyping-enhanced-conservative**
```bash
autotyping --safe --none-return --scalar-return src/
```
- **Target:** Simple return types (int, str, bool) from obvious cases
- **Risk:** Low - only adds when very confident

**2. autotyping-enhanced-moderate** 
```bash
autotyping --safe --none-return --scalar-return --none-param src/
```
- **Target:** Default None parameters (`param: None = None`)
- **Risk:** Low - syntactically obvious cases

**3. autotyping-enhanced-aggressive**
```bash
autotyping --safe --none-return --scalar-return --none-param --guess-simple src/
```
- **Target:** Basic variable type inference from assignments
- **Risk:** Medium - more inference involved

### Stage 2: Targeted infer-types (on autotyping base)

**4. autotyping-safe + infer-inherit**
```bash
# Step 1: autotyping --safe --none-return src/
# Step 2: python -m infer_types --only inherit src/
```
- **Target:** Method signatures from base classes
- **Risk:** Low-medium - inheritance relationships are explicit

**5. autotyping-safe + infer-names**
```bash
# Step 1: autotyping --safe --none-return src/
# Step 2: python -m infer_types --only name src/
```
- **Target:** Types from function/variable naming patterns
- **Risk:** Medium - relies on naming conventions

## Success Criteria

**Primary Goal:** Additional mypy error reduction without ruff regressions  
**Acceptable:** Net mypy improvement ≥10 errors, ruff increase ≤5  
**Ideal:** Net mypy improvement ≥20 errors, ruff increase = 0  

## Measurement Protocol

**Three-stage measurement:**
1. **Baseline:** Current state (959 mypy, 0 ruff)
2. **After autotyping-safe:** Verify reproducibility of -87 improvement  
3. **After additional tool:** Measure incremental benefit
4. **After auto-fixes:** Final state with lint/format applied

**Key metrics:**
- **Total improvement:** Baseline → Final
- **Incremental improvement:** Post-autotyping → Final  
- **Error type analysis:** Which categories improve
- **Regression analysis:** Any new error types introduced

## Expected Outcomes

**Most promising:** autotyping-enhanced-conservative (scalar returns)  
**Wildcard:** infer-inherit (could handle method overrides well)  
**Risky:** infer-names (naming patterns may not be consistent)

## Risk Mitigation

- Test on autotyping-safe foundation (proven safe)
- Incremental flag addition (one at a time)
- Detailed error type tracking
- Auto-fix integration in measurement

This experiment tests whether **conservative typing improvements can be compounded** for greater benefit than any single tool alone.