# Plotting Module Simplification Plan

## Overview
Dramatically simplify the plotting infrastructure by eliminating overengineering while maintaining core functionality. Target: 75% code reduction (400+ lines → ~100 lines).

## Current State Analysis

### Files to Modify:
- `src/dr_gen/analyze/plot_utils.py` (441 lines) - DELETE
- `src/dr_gen/analyze/common_plots.py` (464 lines) - REPLACE with simplified version

### Problems with Current Implementation:
1. **Overengineered config system** - 50+ lines of OmegaConf for basic matplotlib parameters
2. **Unnecessary abstraction layers** - 3-4 function call chains for simple plots  
3. **Complex data handling** - Custom utilities for nested lists instead of pandas/numpy
4. **Grid plotting complexity** - 100+ lines of wrappers around matplotlib subplots

## Implementation Plan

### Phase 1: Create Simplified plots.py

#### 1.1 File Structure
```
src/dr_gen/analyze/
├── plots.py (NEW - replaces both files)
├── plot_utils.py (DELETE)
└── common_plots.py (DELETE)
```

#### 1.2 Core Design Principles
- **Direct matplotlib usage** - eliminate abstraction layers
- **Simple kwargs with defaults** - replace OmegaConf config system
- **Pandas/numpy for statistics** - eliminate custom summary calculations
- **Built-in matplotlib subplots** - eliminate custom grid system

#### 1.3 API Design (10 core functions max)
```python
# Main plotting functions
def plot_lines(data, sample=None, **kwargs)
def plot_histogram(data, **kwargs) 
def plot_cdf(data1, data2, **kwargs)
def plot_splits(split_data, splits=['train', 'val'], **kwargs)
def plot_summary(data_list, **kwargs)  # mean + std/sem bands

# Grid versions  
def plot_grid(plot_func, data_list, **kwargs)

# Utilities
def sample_data(data, n=None)
def set_plot_defaults(**kwargs)
```

### Phase 2: Implementation Strategy

#### 2.1 Replace Config System
```python
# Instead of 50+ lines of OmegaConf
DEFAULT_STYLE = {
    'figsize': (8, 6),
    'alpha': 0.7,
    'linewidth': 2,
    'grid': True
}

def merge_defaults(**kwargs):
    return {**DEFAULT_STYLE, **kwargs}
```

#### 2.2 Direct Matplotlib Implementation
```python
# Replace 3-layer abstraction with direct calls
def plot_lines(data, **kwargs):
    style = merge_defaults(**kwargs)
    fig, ax = plt.subplots(figsize=style['figsize'])
    
    if isinstance(data[0], list):  # multiple curves
        for i, curve in enumerate(data):
            ax.plot(curve, label=f"Curve {i}", **style)
    else:  # single curve
        ax.plot(data, **style)
    
    ax.grid(style['grid'])
    if style.get('legend'): ax.legend()
    plt.show()
```

#### 2.3 Use Pandas/Numpy for Statistics
```python
# Replace custom get_multi_curve_summary_stats()
def plot_summary(data_list, **kwargs):
    import pandas as pd
    
    df = pd.DataFrame(data_list).T  # curves as columns
    mean = df.mean(axis=1)
    std = df.std(axis=1)
    
    fig, ax = plt.subplots(**merge_defaults(**kwargs))
    ax.plot(mean, linewidth=2, label='Mean')
    ax.fill_between(range(len(mean)), mean-std, mean+std, alpha=0.3)
    plt.show()
```

#### 2.4 Simplify Grid Plotting
```python
# Replace 100+ lines of grid wrapper functions
def plot_grid(plot_func, data_list, subplot_shape=None, **kwargs):
    n_plots = len(data_list)
    if subplot_shape is None:
        cols = int(np.ceil(np.sqrt(n_plots)))
        rows = int(np.ceil(n_plots / cols))
    else:
        rows, cols = subplot_shape
        
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3))
    axes = np.atleast_1d(axes).flatten()
    
    for i, data in enumerate(data_list):
        plot_func(data, ax=axes[i], **kwargs)
    
    plt.tight_layout()
    plt.show()
```

### Phase 3: Update Dependencies

#### 3.1 Find All Imports
- Search for `import dr_gen.analyze.plot_utils`
- Search for `import dr_gen.analyze.common_plots`
- Search for `from dr_gen.analyze import plot_utils, common_plots`

#### 3.2 Update Import Statements
```python
# OLD
import dr_gen.analyze.plot_utils as pu
import dr_gen.analyze.common_plots as cp

# NEW  
import dr_gen.analyze.plots as plots
```

#### 3.3 Update Function Calls
```python
# OLD
cp.line_plot(data, **kwargs)
pu.make_summary_plot(data_list, **kwargs)

# NEW
plots.plot_lines(data, **kwargs)
plots.plot_summary(data_list, **kwargs)
```

### Phase 4: Preserve Core Functionality

#### 4.1 Essential Features to Maintain
- ✅ Line plots for training curves
- ✅ Histograms for distributions
- ✅ CDF plots with KS statistics  
- ✅ Split plots (train/val/eval with different line styles)
- ✅ Summary plots (mean + confidence bands)
- ✅ Grid plotting for multiple experiments
- ✅ Random sampling to avoid overcrowded plots
- ✅ Type hints and documentation

#### 4.2 Features to Eliminate
- ❌ OmegaConf configuration system
- ❌ Custom statistical summary calculations
- ❌ Multi-layer function call abstractions
- ❌ Complex grid wrapper functions
- ❌ Custom data structure utilities

### Phase 5: Testing and Validation

#### 5.1 Test Core Functions
```python
# Create test data
curves = [np.random.randn(100) + i for i in range(5)]
split_data = [curves, [c + 0.5 for c in curves]]  # train, val

# Test all functions
plots.plot_lines(curves[0])  # single curve
plots.plot_lines(curves[:3], sample=2)  # multiple curves, sampled
plots.plot_histogram([c[-10:] for c in curves[:2]])  # final values
plots.plot_splits(split_data, splits=['train', 'val'])
plots.plot_summary(curves)
plots.plot_grid(plots.plot_lines, [curves[:2], curves[2:4]])
```

#### 5.2 Verify No Broken Imports
- Run `mp src` to check for import errors
- Run `lint_fix` to ensure code quality
- Test remaining notebooks/scripts that use plotting

## Expected Outcomes

### Quantitative Improvements
- **75% code reduction**: 905 lines → ~225 lines
- **File consolidation**: 2 files → 1 file  
- **Dependency reduction**: Remove OmegaConf dependency for plotting
- **Performance improvement**: Fewer function call layers

### Qualitative Improvements
- **Easier maintenance**: Direct matplotlib usage, clearer code paths
- **Better debugging**: Simpler call stacks, fewer abstractions
- **Improved readability**: Standard pandas/matplotlib patterns
- **Reduced cognitive load**: Single file instead of cross-file navigation

### Maintained Benefits
- All existing plot types and functionality
- Type hints and documentation
- Grid plotting capabilities
- Statistical analysis integration
- Clean, user-friendly API

## Success Criteria
1. All original plotting functionality works
2. No broken imports in remaining codebase
3. Significant reduction in lines of code (target: 75%)
4. Improved performance and maintainability
5. Passes all linting and type checking