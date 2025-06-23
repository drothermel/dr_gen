#!/bin/bash

# Enhanced Typing Experiment Helper Functions
# Focused on the 4-configuration minimal experiment

# ============================================================================
# Core Measurement Functions with Better Error Tracking
# ============================================================================

measure_baseline() {
    echo "=== BASELINE MEASUREMENT ==="
    
    # Capture mypy errors with count and types
    mp src scripts tests > /dev/null 2>&1
    cp .mypy_errors.jsonl .mypy_errors_baseline.jsonl
    local mypy_count=$(head -1 .mypy_errors_baseline.jsonl | jq -r '.error_count // 0')
    echo "Mypy errors: $mypy_count"
    
    # Show mypy error types
    echo "Mypy error types:"
    tail -n +2 .mypy_errors_baseline.jsonl | jq -r '.error_code' | sort | uniq -c | sort -nr
    
    # Capture ruff errors (actual error count, not all output lines)
    local ruff_output=$(lint 2>&1)
    local ruff_count=$(echo "$ruff_output" | grep -E "^\S+\.py:[0-9]+:[0-9]+:" | wc -l)
    echo "Ruff errors: $ruff_count"
    
    # Show ruff error codes
    echo "Ruff error codes:"
    echo "$ruff_output" | grep -E "^\S+\.py:[0-9]+:[0-9]+:" | grep -oE "[A-Z][0-9]{3,4}" | sort | uniq -c | sort -nr
}

measure_after_typing() {
    echo "=== AFTER TYPING TOOL ==="
    
    # Same as baseline but labeled differently
    mp src scripts tests > /dev/null 2>&1
    cp .mypy_errors.jsonl .mypy_errors_after_typing.jsonl
    local mypy_count=$(head -1 .mypy_errors_after_typing.jsonl | jq -r '.error_count // 0')
    echo "Mypy errors: $mypy_count"
    
    echo "Mypy error types:"
    tail -n +2 .mypy_errors_after_typing.jsonl | jq -r '.error_code' | sort | uniq -c | sort -nr
    
    local ruff_output=$(lint 2>&1)
    local ruff_count=$(echo "$ruff_output" | grep -E "^\S+\.py:[0-9]+:[0-9]+:" | wc -l)
    echo "Ruff errors: $ruff_count"
    
    echo "Ruff error codes:"
    echo "$ruff_output" | grep -E "^\S+\.py:[0-9]+:[0-9]+:" | grep -oE "[A-Z][0-9]{3,4}" | sort | uniq -c | sort -nr
}

measure_final() {
    echo "=== FINAL MEASUREMENTS (after auto-fixes) ==="
    
    # Same measurement logic
    mp src scripts tests > /dev/null 2>&1
    cp .mypy_errors.jsonl .mypy_errors_final.jsonl
    local mypy_count=$(head -1 .mypy_errors_final.jsonl | jq -r '.error_count // 0')
    echo "Mypy errors: $mypy_count"
    
    echo "Mypy error types:"
    tail -n +2 .mypy_errors_final.jsonl | jq -r '.error_code' | sort | uniq -c | sort -nr
    
    local ruff_output=$(lint 2>&1)
    local ruff_count=$(echo "$ruff_output" | grep -E "^\S+\.py:[0-9]+:[0-9]+:" | wc -l)
    echo "Ruff errors: $ruff_count"
    
    echo "Ruff error codes:"
    echo "$ruff_output" | grep -E "^\S+\.py:[0-9]+:[0-9]+:" | grep -oE "[A-Z][0-9]{3,4}" | sort | uniq -c | sort -nr
}

# Show specific mypy errors if count increased
analyze_mypy_increase() {
    echo "=== ANALYZING MYPY ERROR INCREASE ==="
    echo "New/changed mypy errors:"
    tail -n +2 .mypy_errors.jsonl | jq -r '.message' | head -10
}

apply_autofixes() {
    echo "=== APPLYING AUTO-FIXES ==="
    # Run ruff check with --fix to auto-fix what it can
    lint_fix
    # Run formatter
    format
}

reset_repo() {
    echo "=== RESETTING REPOSITORY ==="
    git checkout .
}

# ============================================================================
# Tool Application Functions (just the 4 we need)
# ============================================================================

apply_autotyping_safe() {
    echo "=== APPLYING TOOL: autotyping-safe ==="
    uv run autotyping --safe --none-return src/
}

apply_autotyping_aggressive() {
    echo "=== APPLYING TOOL: autotyping-aggressive ==="
    uv run autotyping src/
}

apply_infer_types_default() {
    echo "=== APPLYING TOOL: infer-types-default ==="
    uv run python -m infer_types src/
}

apply_infer_types_conservative() {
    echo "=== APPLYING TOOL: infer-types-conservative ==="
    uv run python -m infer_types --no-assumptions src/
}

# ============================================================================
# Enhanced Experiment Runner
# ============================================================================

run_enhanced_experiment() {
    local config_name="$1"
    local tool_function="$2"
    
    echo ""
    echo "=========================================="
    echo "CONFIGURATION: $config_name"
    echo "=========================================="
    
    # Store baseline counts for comparison
    mp src scripts tests > /dev/null 2>&1
    local baseline_mypy=$(head -1 .mypy_errors.jsonl | jq -r '.error_count')
    local baseline_ruff=$(lint 2>&1 | grep -E "^\S+\.py:[0-9]+:[0-9]+:" | wc -l)
    
    # Baseline measurement
    measure_baseline
    
    # Apply tool
    eval "$tool_function"
    
    # Measure after typing tool (before fixes)
    measure_after_typing
    
    # Check if mypy errors increased
    mp src scripts tests > /dev/null 2>&1
    local after_typing_mypy=$(head -1 .mypy_errors.jsonl | jq -r '.error_count // 0')
    if [[ ${after_typing_mypy:-0} -gt ${baseline_mypy:-0} ]]; then
        echo "⚠️  MYPY ERRORS INCREASED! Analyzing..."
        analyze_mypy_increase
    fi
    
    # Apply auto-fixes
    apply_autofixes
    
    # Final measurement
    measure_final
    
    # Summary
    mp src scripts tests > /dev/null 2>&1
    local final_mypy=$(head -1 .mypy_errors.jsonl | jq -r '.error_count')
    local final_ruff=$(lint 2>&1 | grep -E "^\S+\.py:[0-9]+:[0-9]+:" | wc -l)
    
    echo ""
    echo "=== SUMMARY: $config_name ==="
    echo "Mypy: ${baseline_mypy:-0} → ${final_mypy:-0} (change: $(( ${final_mypy:-0} - ${baseline_mypy:-0} )))"
    echo "Ruff: ${baseline_ruff:-0} → ${final_ruff:-0} (change: $(( ${final_ruff:-0} - ${baseline_ruff:-0} )))"
    echo ""
}

# ============================================================================
# Run All 4 Experiments
# ============================================================================

run_minimal_experiment_suite() {
    echo "Running minimal 4-configuration experiment suite..."
    echo "Starting at: $(date)"
    
    # Configuration 1
    run_enhanced_experiment "autotyping-safe" "apply_autotyping_safe"
    reset_repo
    
    # Configuration 2
    run_enhanced_experiment "autotyping-aggressive" "apply_autotyping_aggressive"
    reset_repo
    
    # Configuration 3
    run_enhanced_experiment "infer-types-default" "apply_infer_types_default"
    reset_repo
    
    # Configuration 4
    run_enhanced_experiment "infer-types-conservative" "apply_infer_types_conservative"
    reset_repo
    
    echo "Experiment suite completed at: $(date)"
}

# ============================================================================
# Usage
# ============================================================================

show_usage() {
    cat << 'EOF'
Enhanced Typing Experiment Helper Functions

BASIC USAGE:
  source typing_experiment_enhanced.sh

RUN ALL 4 EXPERIMENTS:
  run_minimal_experiment_suite

RUN SINGLE EXPERIMENT:
  run_enhanced_experiment "config-name" "apply_function_name"

INDIVIDUAL FUNCTIONS:
  measure_baseline           # Detailed baseline measurement
  measure_after_typing      # Measure after typing tool
  measure_final             # Measure after auto-fixes
  analyze_mypy_increase     # Show why mypy errors increased
  apply_autofixes           # Run lint_fix && ft
  reset_repo                # git checkout .

TOOL FUNCTIONS:
  apply_autotyping_safe
  apply_autotyping_aggressive  
  apply_infer_types_default
  apply_infer_types_conservative

EOF
}

# Show usage if script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    show_usage
fi