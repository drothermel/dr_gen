# Automated Python Type Annotation Tools Research

## Research Overview

Through multiple web searches, I investigated alternatives to manual string matching and complex AST manipulation tools like `libcst` for automatically adding type annotations to Python code.

## Tools Discovered

### 1. `autotyping` - Simple Type Annotation Tool

**Source**: [GitHub - JelleZijlstra/autotyping](https://github.com/JelleZijlstra/autotyping)

**Purpose**: Tool for autoadding simple type annotations that are obvious from context.

**Key Features**:
- `--none-return`: Add `-> None` to functions without any return, yield, or raise
- `--scalar-return`: Add return annotations for functions returning literal bool, str, bytes, int, or float
- `--int-param`, `--float-param`, `--str-param`, `--bytes-param`: Add annotations to parameters with literal defaults
- `--safe`: Conservative mode with high success rate
- `--aggressive`: More transformations, higher risk
- `--guess-common-names`: Infer parameter types from common naming patterns (e.g., `verbose` → `bool`)

**Usage**:
```bash
python -m autotyping --safe --none-return src/
python -m autotyping --aggressive src/
```

**Philosophy**: Conservative, focuses on "tedious" obvious cases that are safe to infer automatically.

### 2. `infer-types` - Comprehensive Type Inference Tool ⭐

**Source**: [GitHub - orsinium-labs/infer-types](https://github.com/orsinium-labs/infer-types)

**Purpose**: "Must have tool for annotating existing code" - designed for big and old codebases.

**Installation**: 
```bash
python3 -m pip install infer-types
# For our project: uv add --dev infer-types
```

**Usage Syntax**:
```bash
python3 -m infer_types ./example/
# For our project: uv run python -m infer_types src/
```

**Key Features**:
- **Sophisticated heuristics**: Actively uses typeshed to find annotations for unannotated dependencies
- **Inheritance-aware**: If same method is defined in base class, copies type annotations from there
- **Context inference**: Can infer that `len(self.users)` returns `int` because `len()` always returns `int`
- **Function name patterns**: `is_open` function assumed to return `bool` because it starts with `is_`
- **Yield handling**: If there's a yield statement, return type is `typing.Iterator`
- **Modern syntax**: Uses Python 3.10+ syntax for type annotations (`str | None` instead of `Optional[str]`)
- **External knowledge**: Leverages typeshed for understanding external library types

**Post-Processing Required**:
```bash
# Fix import organization (infer-types may add duplicate imports)
python3 -m isort ./example/

# Add future annotations for Python <3.10 compatibility  
python3 -m isort --add-import 'from __future__ import annotations' ./example/
```

**Example Transformation**:
```python
# Before
def users_count(self):
    return len(self.users)

# After  
def users_count(self) -> int:
    return len(self.users)
```

**Philosophy**: Comprehensive tool designed specifically for legacy codebase annotation with advanced inference.

### 3. `MonkeyType` - Runtime Type Collection

**Source**: [GitHub - Instagram/MonkeyType](https://github.com/Instagram/MonkeyType)

**Purpose**: Generates static type annotations by collecting runtime types during code execution.

**Installation**:
```bash
pip install monkeytype
# For our project: uv add --dev monkeytype
```

**Requirements**: Python 3.9+, libcst library

**Core Workflow**:
```bash
# 1. Run code under tracing (collect runtime types)
monkeytype run pytest                    # Run test suite
monkeytype run your_script.py           # Run specific script
monkeytype run -m your_module           # Run module as script

# 2. Generate stub files
monkeytype stub some.module              # Generate stub for module
monkeytype stub some.module:SomeClass    # Generate stub for specific class
monkeytype stub some.module > module.pyi # Save stub to file

# 3. Apply annotations directly to source code
monkeytype apply some.module             # Apply annotations in-place
```

**CLI Options**:
- `--config`: Location of config object for trace store
- `--verbose, -v`: Show verbose output
- `--limit, -l`: How many traces to return (default: 2000)
- `--diff`: Show diff between existing and generated annotations
- `--ignore-existing-annotations`: Replace existing annotations

**Data Storage**: 
- Default: SQLite database (`monkeytype.sqlite3` in current directory)
- Configurable storage backends available

**Key Features**:
- **Runtime analysis**: Instruments code during execution to collect actual type usage
- **Test-driven**: Works well with codebases that have good test coverage
- **Accurate types**: Uses real runtime data instead of static inference
- **Direct application**: Can modify source files in-place with `monkeytype apply`
- **Selective annotation**: Can target specific modules, classes, or functions

**Important Limitations**:
- "MonkeyType annotations are rarely suitable exactly as generated; they are a starting point"
- May generate overly specific types (e.g., `List[int]` instead of `Sequence[int]`)
- Requires manual review and adjustment
- Only captures types seen during execution

**Best for**: Codebases with comprehensive test suites and complex runtime type patterns.

### 4. Other Tools Identified

**LibCST**: Concrete syntax tree parser for code transformation (complex, preserves formatting)
**Bowler**: Refactoring tool with fluent API (middle complexity)
**PyAnnotate**: Similar to MonkeyType, runtime type collection
**merge-pyi**: Applies stub files to source code

## Tool Comparison Analysis

| Tool | Complexity | Scope | Best For | Success Rate |
|------|------------|--------|----------|--------------|
| `autotyping` | Low | Simple cases | Obvious annotations | High (conservative) |
| `infer-types` | Medium | Comprehensive | Legacy codebases | Very High |
| `MonkeyType` | Medium | Runtime-based | Test-covered code | Highest (real data) |
| `libcst` | High | Full control | Complex transformations | Highest (but complex) |

## Key Research Findings

### 1. `infer-types` is Superior to `autotyping` for Our Use Case

**Reasons**:
- **Designed for legacy codebases**: "The main scenario for using the tool is to help you with annotating a big and old codebase"
- **More sophisticated inference**: Uses typeshed, inheritance, and contextual analysis
- **ML/Scientific friendly**: Understands `len()`, `np.array()` patterns through typeshed
- **Higher potential impact**: Can handle both mechanical AND semi-contextual cases

### 2. Runtime Collection (`MonkeyType`) Complements Static Inference

**Advantages**:
- **Real type data**: Based on actual execution, not guesswork
- **Complex types**: Can discover generic types, unions, complex ML pipeline types
- **Test validation**: If tests pass, the types are likely correct

**Requirements**:
- Good test coverage
- Ability to run comprehensive test suite

### 3. Two-Tool Strategy is Optimal

**Phase 1**: `infer-types` for comprehensive static inference
**Phase 2**: `MonkeyType` for runtime-validated types on remaining errors

## Updated Experimental Plan

### Context
- **Starting point**: 395 mypy errors → 211 errors (after external library config)
- **Target**: Minimize manual typing effort through automation
- **Error breakdown**: 165 mechanical + 34 semi-contextual + 12 contextual

### Phase 1: Static Inference with `infer-types`
**Hypothesis**: `infer-types` can handle both mechanical (165) and semi-contextual (34) errors

**Steps**:
1. Install: `uv add --dev infer-types`
2. Run: `infer-types src/`
3. Measure: `ckdr` to count remaining errors
4. Document: Success rate and types of remaining errors

**Expected outcome**: 211 → ~50 errors (75% reduction)

### Phase 2: Runtime Collection with `MonkeyType`
**Hypothesis**: MonkeyType can capture complex ML pipeline types that static analysis missed

**Steps**:
1. Install: `uv add --dev monkeytype`
2. Instrument: Run comprehensive test suite with MonkeyType collection
3. Apply: Generate and apply annotations from collected runtime data
4. Measure: `ckdr` to count final remaining errors
5. Document: Additional reduction and type quality

**Expected outcome**: ~50 → ~20 errors (additional 60% reduction on remaining)

### Phase 3: Manual Review
**Remaining**: Complex architectural issues, legitimate bugs, edge cases

**Steps**:
1. Categorize remaining errors
2. Apply manual fixes for high-impact cases
3. Strategic `# type: ignore` for edge cases
4. Document final state

### Success Metrics
- **Quantitative**: Error reduction from 395 to <20 (95% reduction)
- **Qualitative**: Minimize manual typing effort through automation
- **Time**: <30 minutes total vs. hours of manual annotation
- **Learning**: Validate automated typing tools for future projects

## Next Steps

Execute Phase 1 with `infer-types` following the detailed plan below.