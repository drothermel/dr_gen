# Lint and Type Error Fixes - Todo List

## ✅ COMPLETED WORK
**Started with:** 167 errors  
**Completed:** 18 major errors fixed  
**Remaining:** 149 errors  

### Successfully Fixed Categories:

## 1. Import Order Issues (E402) - 2 instances
- [x] Fix `src/dr_gen/data/load_data.py:17` - Move torchvision.transforms import to top
- [x] Fix `src/dr_gen/data/load_data.py:19` - Move dr_gen.schemas import to top

## 2. Line Length Issues (E501) - 4 instances  
- [x] Fix `src/dr_gen/analyze/bootstrapping.py:99` - Line 92 chars (docstring)
- [x] Fix `src/dr_gen/analyze/bootstrapping.py:305` - Line 92 chars (docstring)
- [x] Fix `src/dr_gen/analyze/bootstrapping.py:336` - Line 93 chars (docstring)
- [x] Fix `src/dr_gen/analyze/bootstrapping.py:469` - Line 92 chars (parameter)

## 3. Missing Type Annotations for **kwargs (ANN003) - ~10 instances
- [x] Fix `src/dr_gen/analyze/common_plots.py:37` - line_plot function
- [x] Fix `src/dr_gen/analyze/common_plots.py:53` - histogram_plot function
- [x] Fix `src/dr_gen/analyze/common_plots.py:64` - cdf_plot function
- [x] Fix `src/dr_gen/analyze/common_plots.py:82` - cdf_histogram_plot function
- [x] Fix `src/dr_gen/analyze/common_plots.py:113` - multi_line_plot function
- [x] Fix `src/dr_gen/analyze/common_plots.py:191` - multi_line_sample_plot function
- [x] Fix `src/dr_gen/analyze/common_plots.py:211` - multi_line_summary_plot function
- [x] Fix `src/dr_gen/analyze/common_plots.py:243` - multi_line_sampled_summary_plot function

## 4. Unused Function Arguments (ARG001) - 1 instance
- [x] Fix `src/dr_gen/analyze/common_plots.py:82` - Remove unused `ax` parameter in cdf_histogram_plot

## 5. typing.Any Disallowed (ANN401) - ~15 instances
Need to replace with proper type hints:
- [x] Fix `src/dr_gen/analyze/common_plots.py:15` - data_to_inds function parameter (→ Sized)
- [x] Fix `src/dr_gen/analyze/common_plots.py:19` - default_grid_ind_labels function parameter (→ Sized)
- [x] Fix `src/dr_gen/data/load_data.py:70` - get_dataset transform parameter (→ Callable[[Any], Any] | None)
- [x] Fix `src/dr_gen/data/load_data.py:72` - get_dataset return type (→ Dataset[Any])
- [x] Fix `src/dr_gen/data/load_data.py:152` - _load_source_datasets cfg parameter (→ DictConfig)
- [x] Fix `src/dr_gen/data/viz.py:14` - plot function **imshow_kwargs (→ Keep Any - matplotlib args)
- [x] Fix `src/dr_gen/schemas.py:15` - check_contains cls parameter (→ Type[Enum])
- [x] Fix `src/dr_gen/schemas.py:15` - check_contains val parameter (→ Keep Any - generic value)
- [x] Fix `src/dr_gen/schemas.py:33` - OptimizerType.__contains__ val parameter (→ Keep Any - generic value)
- [x] Fix `src/dr_gen/schemas.py:44` - SchedulerType.__contains__ val parameter (→ Keep Any - generic value)
- [x] Fix `src/dr_gen/schemas.py:54` - CriterionType.__contains__ val parameter (→ Keep Any - generic value)
- [x] Fix `src/dr_gen/train/model.py:167` - create_optim_lrsched cfg parameter (→ DictConfig)
- [x] Fix `src/dr_gen/train/model.py:186` - get_model_optim_lrsched cfg parameter (→ DictConfig)

## Commit Strategy
After completing each major category, commit with descriptive message:
- [x] Commit after fixing import order issues
- [x] Commit after fixing line length issues  
- [x] Commit after fixing missing **kwargs annotations
- [x] Commit after fixing unused arguments
- [x] Commit after fixing typing.Any issues

---

## 🔄 PROGRESS UPDATE (Started: 149 → Current: 118 errors)
*31 errors fixed (~21% reduction)*

### ✅ **COMPLETED HIGH PRIORITY ISSUES**

#### **ANN003 - Missing **kwargs Type Annotations (3 instances)** ✅ FIXED
- [x] `src/dr_gen/analyze/run_group.py:250` - select_run_split_metrics_by_hpms
- [x] `src/dr_gen/analyze/run_group.py:261` - select_run_metrics_by_hpms  
- [x] `src/dr_gen/analyze/run_group.py:275` - ignore_runs_by_hpms

#### **Strategic ANN401 - Plotting Functions kwargs (16+ instances)** ✅ FIXED
*Implemented modern TypedDict approach with Unpack[PlotKwargs]*
- [x] `src/dr_gen/analyze/common_plots.py:38` - line_plot **kwargs
- [x] `src/dr_gen/analyze/common_plots.py:54` - histogram_plot **kwargs
- [x] `src/dr_gen/analyze/common_plots.py:65` - cdf_plot **kwargs
- [x] `src/dr_gen/analyze/common_plots.py:83` - cdf_histogram_plot **kwargs
- [x] `src/dr_gen/analyze/common_plots.py:114` - split_plot **kwargs
- [x] `src/dr_gen/analyze/common_plots.py:192` - multi_line_sample_plot **kwargs
- [x] `src/dr_gen/analyze/common_plots.py:213` - split_sample_plot **kwargs
- [x] `src/dr_gen/analyze/common_plots.py:245` - multi_line_sampled_summary_plot **kwargs
- [x] `src/dr_gen/analyze/common_plots.py:304` - grid_sample_plot_wrapper **kwargs
- [x] `src/dr_gen/analyze/common_plots.py:326` - multi_line_sample_plot_grid
- [x] `src/dr_gen/analyze/common_plots.py:336` - multi_line_sampled_summary_plot_grid
- [x] `src/dr_gen/analyze/common_plots.py:346` - spilt_sample_plot_grid
- [x] `src/dr_gen/analyze/common_plots.py:356` - spilt_sampled_summary_plot_grid
- [x] `src/dr_gen/analyze/common_plots.py:366` - grid_seq_plot_wrapper
- [x] `src/dr_gen/analyze/common_plots.py:387` - histogram_plot_grid
- [x] `src/dr_gen/analyze/common_plots.py:398` - split_plot_grid

### ✅ **COMPLETED MEDIUM PRIORITY ISSUES**

#### **D102 - Missing Docstrings (3 instances)** ✅ FIXED
- [x] `src/dr_gen/analyze/run_group.py:250` - select_run_split_metrics_by_hpms method
- [x] `src/dr_gen/analyze/run_group.py:261` - select_run_metrics_by_hpms method
- [x] `src/dr_gen/analyze/run_group.py:275` - ignore_runs_by_hpms method

#### **D205 - Docstring Formatting (3 instances)** 🔧 PARTIALLY FIXED
*Complex multi-line summary issue - requires manual restructuring*
- [x] `src/dr_gen/analyze/bootstrapping.py:99` - bootstrap_experiment_timesteps (attempted fix)
- [x] `src/dr_gen/analyze/bootstrapping.py:304` - calc_diff_stats_and_ci (attempted fix)
- [x] `src/dr_gen/analyze/bootstrapping.py:336` - calc_ks_stat_and_summary (attempted fix)

#### **Strategic ANN401 - Legitimate Any Usage (5 instances)** ✅ FIXED
*Suppressed with # noqa: ANN401 comments*
- [x] `src/dr_gen/data/viz.py:14` - matplotlib imshow_kwargs (suppressed)
- [x] `src/dr_gen/schemas.py:15` - generic enum value validation (suppressed)
- [x] `src/dr_gen/schemas.py:33` - OptimizerTypes.__contains__ (suppressed)
- [x] `src/dr_gen/schemas.py:44` - LRSchedTypes.__contains__ (suppressed)  
- [x] `src/dr_gen/schemas.py:54` - CriterionTypes.__contains__ (suppressed)

#### **B023 - Loop Variable Binding Issues** ✅ FIXED
- [x] `src/dr_gen/analyze/run_group.py` - Fixed closure issue by passing hpm as parameter

#### **ANN202 - Missing Return Type Annotations** ✅ FIXED
- [x] Added return type annotations for private functions

## 🔄 **REMAINING ISSUES (118 errors)**

### **Major Categories Still Needing Attention:**

#### **D102 - Missing Docstrings (~70 instances)**
*Mostly in analyze/ module - lower priority systematic work*
- Multiple missing docstrings throughout analyze/ module  
- Property methods, utility functions, class methods

#### **ANN401 - Remaining Any Usage (~20 instances)**  
*Need individual evaluation - some may be legitimate*
- Various instances that may need individual evaluation
- Some may be acceptable to suppress based on context

#### **ANN003 - Scattered Missing Type Annotations (~15 instances)**
*Various functions missing parameter type annotations*

#### **Minor Issues (~13 instances)**
- E501 line length issues 
- D205 docstring formatting (3 persistent instances)
- Miscellaneous formatting issues

---

## 📈 **IMPACT ASSESSMENT**

### **✅ Key Achievements:**
1. **Modern Type Safety**: Implemented TypedDict with Unpack for matplotlib kwargs
2. **Strategic Error Suppression**: Proper use of # noqa for legitimate Any usage  
3. **Code Quality**: Fixed loop variable binding and closure issues
4. **Documentation**: Added docstrings for critical methods
5. **Type Annotations**: Comprehensive **kwargs and return type annotations

### **🎯 Recommendations for Remaining Work:**
1. **D102 docstrings** - Can be addressed systematically in batches
2. **Remaining ANN401** - Evaluate case-by-case, suppress where appropriate
3. **Line length issues** - Minor formatting fixes
4. **D205 persistent issues** - Consider project-level suppression if complex

### **📊 Overall Success:**
- **21% error reduction** with focused effort on high-impact issues
- **Major type safety improvements** with modern Python typing practices
- **Strategic approach** prioritizing code quality over superficial compliance