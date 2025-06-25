# Agent Prompt for Experiment Analysis Refactor

## Current Progress
Last Completed: Commit 19 - add: ExperimentDB class for unified analysis API
Next Up: Commit 20 - Add data loading to ExperimentDB

## Task Overview
You are implementing a refactor of the experiment analysis system according to the plan in `experiment_analysis_refactor_plan.md`. This refactor replaces the complex class hierarchy with Pydantic models and Polars DataFrames for better performance and maintainability.

## Your Current Task
1. First, read the implementation plan at `experiment_analysis_refactor_plan.md`
2. Check the git log with `glo` to see which commits have been completed
3. Find the next unimplemented commit in the plan
4. Implement ONLY that specific commit, following the plan precisely

## Implementation Requirements

### For Each Commit:
1. **Start by announcing**: "Starting work on Commit N: [commit description]"
2. **Implement exactly** what the plan specifies - no more, no less
3. **Keep changes to 15-30 lines** of meaningful code (excluding imports/docstrings)
4. **Run `lint_fix`** before committing and fix ALL linting issues
5. **Run tests** if the commit adds tests: `pt tests/analyze/test_[module].py -v`
6. **Fix all test failures** before committing
7. **Use the exact commit message** specified in the plan
8. **Announce completion**: "Completed Commit N: [commit message]"

### Code Quality Standards:
- Use type hints on all functions
- Include docstrings for classes and public functions
- Follow existing import patterns (group by standard/third-party/local)
- Use meaningful variable names
- Prefer functional patterns over stateful classes
- Leverage Polars' native operations over Python loops

### Important Technical Notes:

#### Pydantic v2:
- Use `BaseModel` not dataclasses unless specified
- Use `model_dump()` not `dict()`
- Use `model_validate()` not `parse_obj()`
- Use `ConfigDict` for configuration
- Use `@field_validator` with `@classmethod`
- Use `@computed_field` for derived properties

#### Polars:
- Use `pl.read_ndjson()` for JSONL files
- Use lazy operations (`scan_*`) when possible
- Use expression API: `pl.col("name")` not `df["name"]`
- Cast string columns to `pl.Categorical` for efficiency
- Use `unnest()` for struct columns from JSON

### Error Handling:
If you encounter any issues that require deviation from the plan:
1. **Document at the top** of `experiment_analysis_refactor_plan.md` in a new section "## Implementation Notes"
2. Include: what diverged, why, and what you did instead
3. Continue with the implementation using your best judgment

### Progress Tracking:
After completing each commit, update your status at the top of this prompt file:
```
## Current Progress
Last Completed: Commit N - [commit message]
Next Up: Commit N+1 - [commit description]
```

## Getting Started
1. Run `glo` to see completed commits
2. Find your starting point in the plan
3. Begin implementation of the next commit
4. Follow all requirements above

Remember: Quality over speed. Each commit should be correct, tested, and lint-free before moving on.