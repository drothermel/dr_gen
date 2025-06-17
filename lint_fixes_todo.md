# Lint and Type Error Fixes - Todo List

Found 167 errors from `ckdr` run. Categories and plan:

## 1. Import Order Issues (E402) - 2 instances
- [x] Fix `src/dr_gen/data/load_data.py:17` - Move torchvision.transforms import to top
- [x] Fix `src/dr_gen/data/load_data.py:19` - Move dr_gen.schemas import to top

## 2. Line Length Issues (E501) - 4 instances  
- [ ] Fix `src/dr_gen/analyze/bootstrapping.py:99` - Line 92 chars (docstring)
- [ ] Fix `src/dr_gen/analyze/bootstrapping.py:305` - Line 92 chars (docstring)
- [ ] Fix `src/dr_gen/analyze/bootstrapping.py:336` - Line 93 chars (docstring)
- [ ] Fix `src/dr_gen/analyze/bootstrapping.py:469` - Line 92 chars (parameter)

## 3. Missing Type Annotations for **kwargs (ANN003) - ~10 instances
- [ ] Fix `src/dr_gen/analyze/common_plots.py:37` - line_plot function
- [ ] Fix `src/dr_gen/analyze/common_plots.py:53` - histogram_plot function
- [ ] Fix `src/dr_gen/analyze/common_plots.py:64` - cdf_plot function
- [ ] Fix `src/dr_gen/analyze/common_plots.py:82` - cdf_histogram_plot function
- [ ] Fix `src/dr_gen/analyze/common_plots.py:113` - multi_line_plot function
- [ ] Fix `src/dr_gen/analyze/common_plots.py:191` - multi_line_sample_plot function
- [ ] Fix `src/dr_gen/analyze/common_plots.py:211` - multi_line_summary_plot function
- [ ] Fix `src/dr_gen/analyze/common_plots.py:243` - multi_line_sampled_summary_plot function

## 4. Unused Function Arguments (ARG001) - 1 instance
- [ ] Fix `src/dr_gen/analyze/common_plots.py:82` - Remove unused `ax` parameter in cdf_histogram_plot

## 5. typing.Any Disallowed (ANN401) - ~15 instances
Need to replace with proper type hints:
- [ ] Fix `src/dr_gen/analyze/common_plots.py:15` - data_to_inds function parameter
- [ ] Fix `src/dr_gen/analyze/common_plots.py:19` - default_grid_ind_labels function parameter  
- [ ] Fix `src/dr_gen/data/load_data.py:70` - get_dataset transform parameter
- [ ] Fix `src/dr_gen/data/load_data.py:72` - get_dataset return type
- [ ] Fix `src/dr_gen/data/load_data.py:152` - _load_source_datasets cfg parameter
- [ ] Fix `src/dr_gen/data/viz.py:14` - plot function **imshow_kwargs
- [ ] Fix `src/dr_gen/schemas.py:15` - check_contains cls parameter
- [ ] Fix `src/dr_gen/schemas.py:15` - check_contains val parameter
- [ ] Fix `src/dr_gen/schemas.py:33` - OptimizerType.__contains__ val parameter
- [ ] Fix `src/dr_gen/schemas.py:44` - SchedulerType.__contains__ val parameter
- [ ] Fix `src/dr_gen/schemas.py:54` - CriterionType.__contains__ val parameter
- [ ] Fix `src/dr_gen/train/model.py:167` - create_optim_lrsched cfg parameter
- [ ] Fix `src/dr_gen/train/model.py:186` - get_model_optim_lrsched cfg parameter

## Commit Strategy
After completing each major category, commit with descriptive message:
- [ ] Commit after fixing import order issues
- [ ] Commit after fixing line length issues  
- [ ] Commit after fixing missing **kwargs annotations
- [ ] Commit after fixing unused arguments
- [ ] Commit after fixing typing.Any issues