# Agent Prompt: Run Typing Experiments and Report Results

## Task Overview
You are tasked with running a comprehensive typing experiment using the automated tooling in this repository. Follow the protocol below to test typing tools, measure their effectiveness, and provide a detailed analysis.

## Prerequisites Check
Before starting experiments, verify:

1. **Repository is clean**: Run `gst` to confirm no uncommitted changes
2. **Tools are installed**: Verify autotyping, infer-types, and monkeytype are available
3. **Baseline measurement works**: Test `./measure_typing_baseline.sh` produces valid JSON

## Experimental Protocol

### Phase 1: Establish Baseline
```bash
# Create initial baseline measurement
./measure_typing_baseline.sh > baseline_measurement.json

# Analyze import relationships to identify high-impact modules
./analyze_import_graph.sh --top 10 > import_analysis.txt
```

### Phase 2: Individual Tool Experiments
Run each typing tool independently to measure their individual effectiveness:

```bash
# Test autotyping (conservative approach)
./run_typing_experiment.sh autotyping

# Test infer-types (static analysis)
./run_typing_experiment.sh infer-types

# Test monkeytype (runtime analysis) - focus on high-impact modules
./run_typing_experiment.sh monkeytype --modules utils,metrics,model
```

### Phase 3: Combined Tool Experiment
Test if tools work synergistically:

```bash
# Run tools in sequence with intermediate measurements
./run_typing_experiment.sh combined --sequence autotyping,infer-types,monkeytype
```

### Phase 4: Results Analysis
```bash
# Generate summary of all experiments
for exp in experiments/*/summary.md; do
    echo "=== $(dirname "$exp") ==="
    grep -A 5 "## Summary" "$exp"
    echo ""
done > experiment_summary.txt
```

## Expected Deliverables

### 1. Baseline Analysis
Report the initial state:
- Total errors: Individual (src + scripts + tests) vs Combined
- Cross-directory errors and their implications
- High-impact modules identified for targeted improvement

### 2. Individual Tool Results
For each tool (autotyping, infer-types, monkeytype):
- **Error Reduction**: Absolute numbers and percentages
- **Cascade Effects**: How fixing one directory affected others
- **Quality Assessment**: Types of annotations added
- **Side Effects**: Any new errors or issues introduced

### 3. Combined Tool Analysis
- **Synergy Effects**: Whether tools complement each other
- **Diminishing Returns**: Point where additional tools don't help
- **Optimal Sequence**: Best order for applying multiple tools

### 4. Recommendations
Based on results, provide:
- **Best Single Tool**: Most effective individual approach
- **Best Combined Approach**: Optimal tool sequence if beneficial
- **ROI Analysis**: Time invested vs. errors eliminated
- **Next Steps**: Manual fixes needed for remaining errors

## Analysis Framework

### Key Metrics to Report
1. **Total Error Reduction**: Baseline → Final combined errors
2. **Per-Directory Impact**: How each directory improved
3. **Cross-Directory Benefits**: Cascade effects quantified
4. **Tool Effectiveness Ranking**: Best → Worst performing tools
5. **Time Investment**: Runtime for each tool

### Success Criteria
- **Significant**: >20% total error reduction
- **Moderate**: 10-20% total error reduction  
- **Minimal**: <10% total error reduction

### Quality Indicators
- **Clean JSON Output**: All measurements produce valid structured data
- **No Regressions**: Tools don't introduce new error categories
- **Reproducible Results**: Consistent measurements across runs

## Error Handling

If experiments fail:
1. **Check Prerequisites**: Ensure all tools are properly installed
2. **Repository State**: Verify clean starting state
3. **Disk Space**: Ensure sufficient space for experiment artifacts
4. **Log Analysis**: Check `experiments/*/application.log` for specific errors

## Final Report Template

Structure your final report as:

```markdown
# Typing Automation Experiment Results

## Executive Summary
- **Baseline**: X total errors (Y cross-directory)
- **Best Result**: Z total errors (W% reduction) 
- **Recommended Approach**: [Tool/sequence]

## Detailed Results
[Table showing each experiment's before/after/improvement]

## Key Findings
1. **Most Effective Tool**: [Analysis]
2. **Cascade Effects**: [Impact quantification]
3. **Tool Synergies**: [Combined effectiveness]

## Recommendations
1. **Immediate Actions**: [High-impact, low-effort fixes]
2. **Strategic Approach**: [Long-term typing improvement plan]
3. **Tooling Gaps**: [Areas needing manual intervention]

## Technical Details
- **Methodology**: Cross-directory error measurement
- **Tools Tested**: [Versions and configurations]
- **Reproducibility**: [Commands to replicate results]
```

## Notes
- All experiment results are saved in `experiments/` directory
- Each experiment includes JSON data for programmatic analysis
- Use `compare_typing_results.sh` for detailed before/after comparisons
- The `--keep-changes` flag can be used to inspect specific tool effects
- Experiments automatically create branches and clean up unless specified otherwise

This protocol ensures systematic, reproducible evaluation of typing automation tools with clear metrics for decision-making.