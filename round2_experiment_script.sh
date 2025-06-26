#!/bin/bash

# Round 2 Typing Experiment - Building on autotyping-safe Success
# Tests incremental improvements by stacking typing tools

# ============================================================================
# Round 2 Tool Functions
# ============================================================================

apply_autotyping_enhanced_conservative() {
    echo "=== APPLYING TOOL: autotyping-enhanced-conservative ==="
    echo "Step 1: Base autotyping-safe"
    uv run autotyping --safe --none-return src/
    echo "Step 2: Add scalar-return"
    uv run autotyping --safe --none-return --scalar-return src/
}

apply_autotyping_enhanced_moderate() {
    echo "=== APPLYING TOOL: autotyping-enhanced-moderate ==="
    echo "Step 1: Base autotyping-safe" 
    uv run autotyping --safe --none-return src/
    echo "Step 2: Add scalar-return and none-param"
    uv run autotyping --safe --none-return --scalar-return --none-param src/
}

apply_autotyping_enhanced_aggressive() {
    echo "=== APPLYING TOOL: autotyping-enhanced-aggressive ==="
    echo "Step 1: Base autotyping-safe"
    uv run autotyping --safe --none-return src/
    echo "Step 2: Add all safe enhancements"
    uv run autotyping --safe --none-return --scalar-return --none-param --guess-simple src/
}

apply_autotyping_safe_plus_infer_inherit() {
    echo "=== APPLYING TOOL: autotyping-safe + infer-inherit ==="
    echo "Step 1: Base autotyping-safe"
    uv run autotyping --safe --none-return src/
    echo "Step 2: Add inheritance-based inference"
    uv run python -m infer_types --only inherit src/
}

apply_autotyping_safe_plus_infer_names() {
    echo "=== APPLYING TOOL: autotyping-safe + infer-names ==="
    echo "Step 1: Base autotyping-safe"
    uv run autotyping --safe --none-return src/
    echo "Step 2: Add name-based inference"
    uv run python -m infer_types --only name src/
}

# ============================================================================
# Enhanced Measurement Functions (reuse from Round 1 with stage naming)
# ============================================================================

measure_round2_baseline() {
    echo "=== ROUND 2 BASELINE (original codebase) ==="
    
    mp src scripts tests > /dev/null 2>&1
    cp .mypy_errors.jsonl .mypy_errors_round2_baseline.jsonl
    local mypy_count=$(head -1 .mypy_errors_round2_baseline.jsonl | jq -r '.error_count // 0')
    echo "Mypy errors: $mypy_count"
    
    echo "Mypy error types:"
    tail -n +2 .mypy_errors_round2_baseline.jsonl | jq -r '.error_code' | sort | uniq -c | sort -nr
    
    local ruff_output=$(lint 2>&1)
    local ruff_count=$(echo "$ruff_output" | grep -E "^\S+\.py:[0-9]+:[0-9]+:" | wc -l)
    echo "Ruff errors: $ruff_count"
    
    echo "Ruff error codes:"
    echo "$ruff_output" | grep -E "^\S+\.py:[0-9]+:[0-9]+:" | grep -oE "[A-Z][0-9]{3,4}" | sort | uniq -c | sort -nr
}

measure_after_tools() {
    echo "=== AFTER TYPING TOOLS ==="
    
    mp src scripts tests > /dev/null 2>&1
    cp .mypy_errors.jsonl .mypy_errors_after_tools.jsonl
    local mypy_count=$(head -1 .mypy_errors_after_tools.jsonl | jq -r '.error_count // 0')
    echo "Mypy errors: $mypy_count"
    
    echo "Mypy error types:"
    tail -n +2 .mypy_errors_after_tools.jsonl | jq -r '.error_code' | sort | uniq -c | sort -nr
    
    local ruff_output=$(lint 2>&1)
    local ruff_count=$(echo "$ruff_output" | grep -E "^\S+\.py:[0-9]+:[0-9]+:" | wc -l)
    echo "Ruff errors: $ruff_count"
    
    echo "Ruff error codes:"
    echo "$ruff_output" | grep -E "^\S+\.py:[0-9]+:[0-9]+:" | grep -oE "[A-Z][0-9]{3,4}" | sort | uniq -c | sort -nr
}

measure_round2_final() {
    echo "=== ROUND 2 FINAL (after auto-fixes) ==="
    
    mp src scripts tests > /dev/null 2>&1
    cp .mypy_errors.jsonl .mypy_errors_round2_final.jsonl
    local mypy_count=$(head -1 .mypy_errors_round2_final.jsonl | jq -r '.error_count // 0')
    echo "Mypy errors: $mypy_count"
    
    echo "Mypy error types:"
    tail -n +2 .mypy_errors_round2_final.jsonl | jq -r '.error_code' | sort | uniq -c | sort -nr
    
    local ruff_output=$(lint 2>&1)
    local ruff_count=$(echo "$ruff_output" | grep -E "^\S+\.py:[0-9]+:[0-9]+:" | wc -l)
    echo "Ruff errors: $ruff_count"
    
    echo "Ruff error codes:"
    echo "$ruff_output" | grep -E "^\S+\.py:[0-9]+:[0-9]+:" | grep -oE "[A-Z][0-9]{3,4}" | sort | uniq -c | sort -nr
}

analyze_mypy_increase() {
    echo "=== ANALYZING MYPY ERROR INCREASE ==="
    echo "New/changed mypy errors:"
    tail -n +2 .mypy_errors_after_tools.jsonl | jq -r '.message' | head -10
}

apply_autofixes() {
    echo "=== APPLYING AUTO-FIXES ==="
    lint_fix
    format
}

reset_repo() {
    echo "=== RESETTING REPOSITORY ==="
    git checkout .
}

# ============================================================================
# Round 2 Experiment Runner  
# ============================================================================

run_round2_experiment() {
    local config_name="$1"
    local tool_function="$2"
    
    echo ""
    echo "=========================================="
    echo "ROUND 2 CONFIGURATION: $config_name"
    echo "=========================================="
    
    # Store baseline counts for comparison
    mp src scripts tests > /dev/null 2>&1
    local baseline_mypy=$(head -1 .mypy_errors.jsonl | jq -r '.error_count // 0')
    local baseline_ruff=$(lint 2>&1 | grep -E "^\S+\.py:[0-9]+:[0-9]+:" | wc -l)
    
    # Baseline measurement
    measure_round2_baseline
    
    # Apply tools (potentially multi-stage)
    eval "$tool_function"
    
    # Measure after tools (before fixes)
    measure_after_tools
    
    # Check if mypy errors increased
    mp src scripts tests > /dev/null 2>&1
    local after_tools_mypy=$(head -1 .mypy_errors.jsonl | jq -r '.error_count // 0')
    if [[ ${after_tools_mypy:-0} -gt ${baseline_mypy:-0} ]]; then
        echo "⚠️  MYPY ERRORS INCREASED! Analyzing..."
        analyze_mypy_increase
    fi
    
    # Apply auto-fixes
    apply_autofixes
    
    # Final measurement
    measure_round2_final
    
    # Summary with Round 1 comparison
    mp src scripts tests > /dev/null 2>&1
    local final_mypy=$(head -1 .mypy_errors.jsonl | jq -r '.error_count // 0')
    local final_ruff=$(lint 2>&1 | grep -E "^\S+\.py:[0-9]+:[0-9]+:" | wc -l)
    
    # Round 1 autotyping-safe achieved: 1046 → 959 (-87)
    local round1_result=959
    local total_improvement=$((1046 - final_mypy))
    local incremental_improvement=$((round1_result - final_mypy))
    
    echo ""
    echo "=== SUMMARY: $config_name ==="
    echo "Mypy: ${baseline_mypy:-0} → ${final_mypy:-0} (total change: $((final_mypy - baseline_mypy)))"
    echo "Ruff: ${baseline_ruff:-0} → ${final_ruff:-0} (change: $((final_ruff - baseline_ruff)))"
    echo ""
    echo "📊 ROUND 2 ANALYSIS:"
    echo "Total improvement from original: -$total_improvement errors"
    echo "Incremental vs Round 1 winner: $incremental_improvement errors"
    if [[ $incremental_improvement -lt 0 ]]; then
        echo "✅ BEATS Round 1 by ${incremental_improvement#-} errors!"
    elif [[ $incremental_improvement -eq 0 ]]; then
        echo "⚖️  MATCHES Round 1 performance"
    else
        echo "❌ WORSE than Round 1 by +$incremental_improvement errors"
    fi
    echo ""
}

# ============================================================================
# Run All Round 2 Experiments
# ============================================================================

run_round2_experiment_suite() {
    echo "Running Round 2 incremental typing experiment..."
    echo "Building on Round 1 autotyping-safe success (-87 errors)"
    echo "Starting at: $(date)"
    
    # Configuration 1: Enhanced Conservative
    run_round2_experiment "autotyping-enhanced-conservative" "apply_autotyping_enhanced_conservative"
    reset_repo
    
    # Configuration 2: Enhanced Moderate
    run_round2_experiment "autotyping-enhanced-moderate" "apply_autotyping_enhanced_moderate"
    reset_repo
    
    # Configuration 3: Enhanced Aggressive  
    run_round2_experiment "autotyping-enhanced-aggressive" "apply_autotyping_enhanced_aggressive"
    reset_repo
    
    # Configuration 4: autotyping + infer-inherit
    run_round2_experiment "autotyping-safe + infer-inherit" "apply_autotyping_safe_plus_infer_inherit"
    reset_repo
    
    # Configuration 5: autotyping + infer-names
    run_round2_experiment "autotyping-safe + infer-names" "apply_autotyping_safe_plus_infer_names"
    reset_repo
    
    echo "Round 2 experiment suite completed at: $(date)"
    echo ""
    echo "🎯 ROUND 2 GOAL: Beat Round 1's -87 error improvement"
    echo "📈 SUCCESS CRITERIA: ≥10 additional errors fixed, ≤5 ruff regressions"
}

# ============================================================================
# Usage
# ============================================================================

show_round2_usage() {
    cat << 'EOF'
Round 2 Typing Experiment - Building on Success

OVERVIEW:
  Tests incremental improvements by building on autotyping-safe foundation
  Round 1 winner: autotyping-safe achieved -87 mypy errors (1046 → 959)

RUN ALL EXPERIMENTS:
  run_round2_experiment_suite

RUN SINGLE EXPERIMENT:
  run_round2_experiment "config-name" "apply_function_name"

CONFIGURATIONS:
1. autotyping-enhanced-conservative  (adds --scalar-return)
2. autotyping-enhanced-moderate      (adds --none-param too)
3. autotyping-enhanced-aggressive    (adds --guess-simple too)
4. autotyping-safe + infer-inherit   (inheritance-based)
5. autotyping-safe + infer-names     (name-based inference)

MEASUREMENT FILES:
  .mypy_errors_round2_baseline.jsonl  - Original state
  .mypy_errors_after_tools.jsonl      - After typing tools
  .mypy_errors_round2_final.jsonl     - After auto-fixes

EOF
}

# Show usage if script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    show_round2_usage
fi