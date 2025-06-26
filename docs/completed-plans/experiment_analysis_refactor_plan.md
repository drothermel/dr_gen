# Experiment Analysis System Refactor Implementation Plan

## Completion Details
- **Date Completed**: 2025-01-25
- **Status**: Plan Created and Archived
- **Implementation Status**: Not Started
- **Notes**: This plan outlines a comprehensive refactor of the experiment analysis system to use Pydantic v2 and Polars. The plan includes 28 detailed commits across 6 stages, designed to incrementally migrate the system while maintaining all existing functionality.

---

## Overview
This plan refactors the experiment analysis system to use Pydantic v2 for data validation and Polars for efficient data manipulation. Each commit is designed to be 15-30 lines of self-contained changes.

## Architecture Goals
1. Replace complex class hierarchy with Pydantic models + Polars DataFrames
2. Use functional approach with immutable data structures
3. Leverage Polars' performance for large-scale experiment analysis
4. Maintain all existing functionality while improving simplicity

## Implementation Stages

### Stage 1: Core Data Models (Commits 1-4)

**Commit 1: Add Pydantic models for hyperparameters**
- Create `src/dr_gen/analyze/models.py`
- Define `Hyperparameters` class with flatten/validation logic
- Run `lint_fix` and ensure no linting issues
- Commit message: `add: Pydantic model for hyperparameter management`

**Commit 2: Add test for Hyperparameters model**
- Create `tests/analyze/test_models.py`
- Test flatten logic, validation, and serialization
- Run `lint_fix` and `pt tests/analyze/test_models.py -v`
- Commit message: `test: Hyperparameters model validation and serialization`

**Commit 3: Add Run data model with computed fields**
- Add `Run` model to `models.py` with metrics/metadata
- Include computed fields for best metrics
- Run `lint_fix` and ensure no linting issues
- Commit message: `add: Run model with computed metric properties`

**Commit 4: Add comprehensive Run model tests**
- Test Run validation, computed fields, serialization
- Test error handling for invalid data
- Run `lint_fix` and `pt tests/analyze/test_models.py -v`
- Commit message: `test: Run model validation and computed fields`

### Stage 2: Data Loading (Commits 5-8)

**Commit 5: Add JSONL parsing utilities**
- Create `src/dr_gen/analyze/parsers.py`
- Add `parse_jsonl_file` function with error handling
- Run `lint_fix` and ensure no linting issues
- Commit message: `add: JSONL parsing utility with error collection`

**Commit 6: Test JSONL parsing with valid/invalid data**
- Create `tests/analyze/test_parsers.py`
- Test parsing valid files, corrupted files, missing data
- Run `lint_fix` and `pt tests/analyze/test_parsers.py -v`
- Commit message: `test: JSONL parser error handling and validation`

**Commit 7: Add run loading from directory**
- Add `load_runs_from_dir` function to `parsers.py`
- Return list of validated Run models
- Run `lint_fix` and ensure no linting issues
- Commit message: `add: Batch run loading from directory structure`

**Commit 8: Test directory-based run loading**
- Test loading multiple files, handling errors
- Verify partial success with some invalid files
- Run `lint_fix` and `pt tests/analyze/test_parsers.py -v`
- Commit message: `test: Directory-based run loading and error resilience`

### Stage 3: Polars Integration (Commits 9-14)

**Commit 9: Add Polars DataFrame builders**
- Create `src/dr_gen/analyze/dataframes.py`
- Add `runs_to_dataframe` function
- Run `lint_fix` and ensure no linting issues
- Commit message: `add: Convert Run models to Polars DataFrame`

**Commit 10: Add metrics DataFrame builder**
- Add `runs_to_metrics_df` for long-format metrics
- Handle nested metric dictionaries
- Run `lint_fix` and ensure no linting issues
- Commit message: `add: Long-format metrics DataFrame builder`

**Commit 11: Test DataFrame conversion**
- Create `tests/analyze/test_dataframes.py`
- Test both run and metrics DataFrame creation
- Run `lint_fix` and `pt tests/analyze/test_dataframes.py -v`
- Commit message: `test: Polars DataFrame conversion from Run models`

**Commit 12: Add hyperparameter analysis functions**
- Add `find_varying_hparams` to `dataframes.py`
- Add `group_by_hparams` function
- Run `lint_fix` and ensure no linting issues
- Commit message: `add: Hyperparameter variation analysis with Polars`

**Commit 13: Add metric query functions**
- Add `query_metrics` with filtering support
- Add `summarize_by_hparams` for aggregations
- Run `lint_fix` and ensure no linting issues
- Commit message: `add: Metric querying and aggregation functions`

**Commit 14: Test analysis functions**
- Test varying hparam detection, grouping, queries
- Test aggregation with multiple statistics
- Run `lint_fix` and `pt tests/analyze/test_dataframes.py -v`
- Commit message: `test: Hyperparameter and metric analysis functions`

### Stage 4: Configuration System (Commits 15-18)

**Commit 15: Add analysis configuration model**
- Create `AnalysisConfig` with Pydantic BaseSettings
- Support environment variables and config files
- Run `lint_fix` and ensure no linting issues
- Commit message: `add: Pydantic-based analysis configuration`

**Commit 16: Test configuration loading**
- Test loading from dict, env vars, files
- Test validation and defaults
- Run `lint_fix` and `pt tests/analyze/test_models.py -v`
- Commit message: `test: Analysis configuration validation and loading`

**Commit 17: Add display name mapping utilities**
- Add remapping functions using config mappings
- Support both key and value transformations
- Run `lint_fix` and ensure no linting issues
- Commit message: `add: Display name mapping utilities`

**Commit 18: Test display mapping functionality**
- Test key/value remapping with various inputs
- Test reverse mapping for queries
- Run `lint_fix` and `pt tests/analyze/test_dataframes.py -v`
- Commit message: `test: Display name mapping and reverse lookup`

### Stage 5: High-Level API (Commits 19-24)

**Commit 19: Add ExperimentDB class**
- Create `src/dr_gen/analyze/experiment_db.py`
- Initialize with config, lazy loading support
- Run `lint_fix` and ensure no linting issues
- Commit message: `add: ExperimentDB class for unified analysis API`

**Commit 20: Add data loading to ExperimentDB**
- Implement `load_experiments` method
- Store DataFrames, handle errors
- Run `lint_fix` and ensure no linting issues
- Commit message: `add: Experiment loading in ExperimentDB`

**Commit 21: Add query methods to ExperimentDB**
- Implement metric queries with filters
- Add summary statistics methods
- Run `lint_fix` and ensure no linting issues
- Commit message: `add: Query and analysis methods to ExperimentDB`

**Commit 22: Test ExperimentDB basic functionality**
- Create `tests/analyze/test_experiment_db.py`
- Test initialization, loading, basic queries
- Run `lint_fix` and `pt tests/analyze/test_experiment_db.py -v`
- Commit message: `test: ExperimentDB initialization and data loading`

**Commit 23: Add lazy evaluation support**
- Add lazy query building methods
- Support streaming for large datasets
- Run `lint_fix` and ensure no linting issues
- Commit message: `add: Lazy evaluation support for large experiments`

**Commit 24: Test advanced ExperimentDB features**
- Test complex queries, lazy evaluation
- Test memory efficiency with mock large data
- Run `lint_fix` and `pt tests/analyze/test_experiment_db.py -v`
- Commit message: `test: ExperimentDB advanced queries and lazy evaluation`

### Stage 6: Migration Utilities (Commits 25-28)

**Commit 25: Add legacy data converter**
- Create converter from old format to new
- Maintain backward compatibility
- Run `lint_fix` and ensure no linting issues
- Commit message: `add: Legacy format converter for migration`

**Commit 26: Test legacy converter**
- Test conversion preserves all data
- Test handling of edge cases
- Run `lint_fix` and `pt tests/analyze/test_parsers.py -v`
- Commit message: `test: Legacy data format conversion`

**Commit 27: Add usage examples**
- Create `examples/analyze_experiments.py`
- Show common analysis workflows
- Run `lint_fix` and ensure no linting issues
- Commit message: `docs: Add experiment analysis examples`

**Commit 28: Update analysis documentation**
- Update `docs/analysis_system_overview.md`
- Document new API and migration guide
- Run `lint_fix` on any code snippets
- Commit message: `docs: Update analysis system documentation`

## Success Criteria
- All tests pass after each commit
- No linting issues in any commit
- Each commit builds on previous functionality
- System maintains all original capabilities with simpler implementation