#!/bin/bash

# Typing Experiment Helper Functions
# Source this file to get access to experiment functions

# ============================================================================
# Core Measurement Functions
# ============================================================================

measure_baseline() {
    echo "=== BASELINE MEASUREMENT ==="
    local mypy_errors=$(mp 2>&1 | grep "Found [0-9]* error" | head -1)
    local ruff_errors=$(lint --quiet 2>&1 | wc -l | awk '{print "Found " $1 " ruff error(s)"}')
    echo "$mypy_errors"
    echo "$ruff_errors"
}

measure_final() {
    echo "=== FINAL MEASUREMENTS ==="
    local mypy_errors=$(mp 2>&1 | grep "Found [0-9]* error" | head -1)
    local ruff_errors=$(lint --quiet 2>&1 | wc -l | awk '{print "Found " $1 " ruff error(s)"}')
    echo "$mypy_errors"
    echo "$ruff_errors"
}

apply_autofixes() {
    echo "=== APPLYING AUTO-FIXES ==="
    lint && ft
}

reset_repo() {
    echo "=== RESETTING REPOSITORY ==="
    git checkout .
}

# ============================================================================
# Tool Application Functions
# ============================================================================

apply_autotyping_safe() {
    echo "=== APPLYING TOOL: autotyping-safe ==="
    uv run autotyping --safe --none-return src/
}

apply_autotyping_aggressive() {
    echo "=== APPLYING TOOL: autotyping-aggressive ==="
    uv run autotyping src/
}

apply_autotyping_comprehensive() {
    echo "=== APPLYING TOOL: autotyping-comprehensive ==="
    uv run autotyping --safe --none-return --guess-simple src/
}

apply_infer_types_default() {
    echo "=== APPLYING TOOL: infer-types-default ==="
    uv run python -m infer_types src/
}

apply_infer_types_conservative() {
    echo "=== APPLYING TOOL: infer-types --no-assumptions (conservative) ==="
    uv run python -m infer_types --no-assumptions src/
}

apply_infer_types_no_imports() {
    echo "=== APPLYING TOOL: infer-types --no-imports ==="
    uv run python -m infer_types --no-imports src/
}

apply_infer_types_no_methods() {
    echo "=== APPLYING TOOL: infer-types --no-methods ==="
    uv run python -m infer_types --no-methods src/
}

apply_infer_types_no_functions() {
    echo "=== APPLYING TOOL: infer-types --no-functions ==="
    uv run python -m infer_types --no-functions src/
}

apply_infer_types_only_none() {
    echo "=== APPLYING TOOL: infer-types --only none ==="
    uv run python -m infer_types --only none src/
}

apply_infer_types_only_yield() {
    echo "=== APPLYING TOOL: infer-types --only yield ==="
    uv run python -m infer_types --only yield src/
}

# ============================================================================
# Complete Experiment Workflows
# ============================================================================

run_experiment() {
    local config_name="$1"
    local tool_function="$2"
    
    echo "=================================="
    echo "CONFIGURATION: $config_name"
    echo "=================================="
    
    # Baseline measurement
    measure_baseline
    
    # Apply tool
    eval "$tool_function"
    
    # Auto-fixes
    apply_autofixes
    
    # Final measurement
    measure_final
    
    echo "=== EXPERIMENT COMPLETE: $config_name ==="
    echo ""
}

# ============================================================================
# Quick Test Functions
# ============================================================================

# Quick test of a single configuration without reset
quick_test() {
    local tool_function="$1"
    echo "=== QUICK TEST ==="
    measure_baseline
    eval "$tool_function"
    apply_autofixes
    measure_final
}

# ============================================================================
# Batch Experiment Runner
# ============================================================================

run_remaining_experiments() {
    echo "Running all remaining experiments..."
    
    # Configuration 5
    run_experiment "infer-types-no-imports" "apply_infer_types_no_imports"
    reset_repo
    
    # Configuration 6
    run_experiment "infer-types-no-methods" "apply_infer_types_no_methods"
    reset_repo
    
    # Configuration 7
    run_experiment "infer-types-no-functions" "apply_infer_types_no_functions"
    reset_repo
    
    # Configuration 8
    run_experiment "infer-types-only-none" "apply_infer_types_only_none"
    reset_repo
    
    # Configuration 9
    run_experiment "infer-types-only-yield" "apply_infer_types_only_yield"
    reset_repo
    
    # Configuration 10
    run_experiment "autotyping-comprehensive" "apply_autotyping_comprehensive"
    reset_repo
    
    echo "All remaining experiments completed!"
}

# ============================================================================
# Usage Examples
# ============================================================================

show_usage() {
    cat << 'EOF'
Typing Experiment Helper Functions

BASIC USAGE:
  source typing_experiment_helpers.sh

INDIVIDUAL FUNCTIONS:
  measure_baseline                 # Check current mypy/ruff errors
  measure_final                   # Check final mypy/ruff errors
  apply_autofixes                 # Run lint && ft
  reset_repo                      # git checkout .

TOOL FUNCTIONS:
  apply_autotyping_safe           # autotyping --safe --none-return
  apply_autotyping_aggressive     # autotyping (default)
  apply_autotyping_comprehensive  # autotyping with all safe flags
  apply_infer_types_default       # infer_types (default)
  apply_infer_types_conservative  # infer_types --no-assumptions
  apply_infer_types_no_imports    # infer_types --no-imports
  apply_infer_types_no_methods    # infer_types --no-methods
  apply_infer_types_no_functions  # infer_types --no-functions
  apply_infer_types_only_none     # infer_types --only none
  apply_infer_types_only_yield    # infer_types --only yield

EXPERIMENT WORKFLOWS:
  run_experiment "name" "tool_function"    # Full experiment workflow
  quick_test "tool_function"               # Quick test without reset
  run_remaining_experiments                # Run all remaining configs

EXAMPLES:
  run_experiment "infer-types-no-imports" "apply_infer_types_no_imports"
  quick_test "apply_autotyping_safe"
  run_remaining_experiments

EOF
}

# Show usage if script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    show_usage
fi