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

## 🔄 REMAINING ERRORS (149 total)
*Analysis of remaining lint and type errors by category and priority*

### High Priority Issues (12 errors)

#### **ANN003 - Missing **kwargs Type Annotations (3 instances)**
- [ ] `src/dr_gen/analyze/run_group.py:250` - select_run_split_metrics_by_hpms
- [ ] `src/dr_gen/analyze/run_group.py:261` - select_run_metrics_by_hpms  
- [ ] `src/dr_gen/analyze/run_group.py:275` - ignore_runs_by_hpms

#### **Strategic ANN401 - Plotting Functions kwargs (9 instances)**
*These are currently flagged but may be acceptable as-is for matplotlib compatibility*
- [ ] `src/dr_gen/analyze/common_plots.py:38` - line_plot **kwargs
- [ ] `src/dr_gen/analyze/common_plots.py:54` - histogram_plot **kwargs
- [ ] `src/dr_gen/analyze/common_plots.py:65` - cdf_plot **kwargs
- [ ] `src/dr_gen/analyze/common_plots.py:83` - cdf_histogram_plot **kwargs
- [ ] `src/dr_gen/analyze/common_plots.py:114` - split_plot **kwargs
- [ ] `src/dr_gen/analyze/common_plots.py:192` - multi_line_sample_plot **kwargs
- [ ] `src/dr_gen/analyze/common_plots.py:213` - split_sample_plot **kwargs
- [ ] `src/dr_gen/analyze/common_plots.py:245` - multi_line_sampled_summary_plot **kwargs
- [ ] `src/dr_gen/analyze/common_plots.py:304` - grid_sample_plot_wrapper **kwargs

### Medium Priority Issues (18 errors)

#### **D102 - Missing Docstrings (3 instances)**
- [ ] `src/dr_gen/analyze/run_group.py:250` - select_run_split_metrics_by_hpms method
- [ ] `src/dr_gen/analyze/run_group.py:261` - select_run_metrics_by_hpms method
- [ ] `src/dr_gen/analyze/run_group.py:275` - ignore_runs_by_hpms method

#### **D205 - Docstring Formatting (3 instances)**
*Need blank line between summary and description*
- [ ] `src/dr_gen/analyze/bootstrapping.py:99` - bootstrap_experiment_timesteps
- [ ] `src/dr_gen/analyze/bootstrapping.py:304` - calc_diff_stats_and_ci
- [ ] `src/dr_gen/analyze/bootstrapping.py:336` - calc_ks_stat_and_summary

#### **Strategic ANN401 - Legitimate Any Usage (5 instances)**
*These should likely remain as Any for valid reasons*
- [ ] `src/dr_gen/data/viz.py:14` - matplotlib imshow_kwargs (keep Any)
- [ ] `src/dr_gen/schemas.py:15` - generic enum value validation (keep Any)
- [ ] `src/dr_gen/schemas.py:33` - OptimizerTypes.__contains__ (keep Any)
- [ ] `src/dr_gen/schemas.py:44` - LRSchedTypes.__contains__ (keep Any)  
- [ ] `src/dr_gen/schemas.py:54` - CriterionTypes.__contains__ (keep Any)

#### **Additional Plotting Function kwargs (7 instances)**
- [ ] `src/dr_gen/analyze/common_plots.py:326` - multi_line_sample_plot_grid
- [ ] `src/dr_gen/analyze/common_plots.py:336` - multi_line_sampled_summary_plot_grid
- [ ] `src/dr_gen/analyze/common_plots.py:346` - spilt_sample_plot_grid
- [ ] `src/dr_gen/analyze/common_plots.py:356` - spilt_sampled_summary_plot_grid
- [ ] `src/dr_gen/analyze/common_plots.py:366` - grid_seq_plot_wrapper
- [ ] `src/dr_gen/analyze/common_plots.py:387` - histogram_plot_grid
- [ ] `src/dr_gen/analyze/common_plots.py:398` - split_plot_grid

### Lower Priority Issues (119+ errors)

#### **B023 - Loop Variable Binding Issues**
*Multiple instances in run_group.py with lambda functions*

#### **Additional D102 - Missing Docstrings**  
*Multiple missing docstrings throughout analyze/ module*

#### **Additional ANN003 - Missing Type Annotations**
*Various functions missing parameter type annotations*

#### **Additional ANN401 - Other Any Usage**
*Various instances that may need individual evaluation*

### Approach Recommendations:

1. **High Priority**: Fix the 3 remaining **kwargs annotations in run_group.py
2. **Medium Priority**: Add missing docstrings and fix docstring formatting  
3. **Strategic Decision**: Evaluate whether to suppress ANN401 for legitimate matplotlib/enum cases
4. **Lower Priority**: Address remaining issues based on project priorities

### Notes:
- Many remaining ANN401 errors are for legitimate use cases (matplotlib kwargs, enum validation)
- Consider adding `# noqa: ANN401` for strategic Any usage
- B023 loop binding issues may require code restructuring
- Some errors may be acceptable to suppress based on project standards