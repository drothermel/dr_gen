#!/bin/bash
# compare_typing_results.sh - Compare typing results between two measurements
set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Usage function
usage() {
    echo "Usage: $0 <baseline.json> <after.json>"
    echo ""
    echo "Compares two typing measurement JSON files and shows improvements"
    echo ""
    echo "Example:"
    echo "  $0 experiments/baseline.json experiments/after_autotyping.json"
    exit 1
}

# Check arguments
if [[ $# -ne 2 ]]; then
    usage
fi

BASELINE_FILE=$1
AFTER_FILE=$2

# Check files exist
if [[ ! -f "$BASELINE_FILE" ]]; then
    echo -e "${RED}❌ Error: Baseline file not found: $BASELINE_FILE${NC}"
    exit 1
fi

if [[ ! -f "$AFTER_FILE" ]]; then
    echo -e "${RED}❌ Error: After file not found: $AFTER_FILE${NC}"
    exit 1
fi

# Function to calculate percentage
calc_percentage() {
    local before=$1
    local after=$2
    if [[ $before -eq 0 ]]; then
        echo "0.0"
    else
        echo "scale=1; (($before - $after) * 100) / $before" | bc
    fi
}

# Function to format improvement
format_improvement() {
    local before=$1
    local after=$2
    local diff=$((before - after))
    local pct=$(calc_percentage $before $after)
    
    if [[ $diff -gt 0 ]]; then
        echo -e "${GREEN}-$diff errors ($pct% improvement)${NC}"
    elif [[ $diff -lt 0 ]]; then
        echo -e "${RED}+${diff#-} errors (${pct#-}% worse)${NC}"
    else
        echo -e "${YELLOW}No change${NC}"
    fi
}

# Extract data from JSON files
baseline_src=$(jq -r '.individual.src' "$BASELINE_FILE")
baseline_scripts=$(jq -r '.individual.scripts' "$BASELINE_FILE")
baseline_tests=$(jq -r '.individual.tests' "$BASELINE_FILE")
baseline_combined=$(jq -r '.combined' "$BASELINE_FILE")
baseline_cross=$(jq -r '.cross_directory_errors' "$BASELINE_FILE")

after_src=$(jq -r '.individual.src' "$AFTER_FILE")
after_scripts=$(jq -r '.individual.scripts' "$AFTER_FILE")
after_tests=$(jq -r '.individual.tests' "$AFTER_FILE")
after_combined=$(jq -r '.combined' "$AFTER_FILE")
after_cross=$(jq -r '.cross_directory_errors' "$AFTER_FILE")

# Extract metadata
baseline_timestamp=$(jq -r '.timestamp' "$BASELINE_FILE")
after_timestamp=$(jq -r '.timestamp' "$AFTER_FILE")
baseline_commit=$(jq -r '.git_commit' "$BASELINE_FILE")
after_commit=$(jq -r '.git_commit' "$AFTER_FILE")

# Generate markdown report
cat <<EOF
# Typing Results Comparison

## Metadata
- **Baseline**: $baseline_timestamp (commit: $baseline_commit)
- **After**: $after_timestamp (commit: $after_commit)

## Summary
- **Total Errors**: $baseline_combined → $after_combined $(format_improvement $baseline_combined $after_combined)
- **Cross-Directory Errors**: $baseline_cross → $after_cross

## Per-Directory Results

| Directory | Baseline | After | Change |
|-----------|----------|-------|--------|
| src       | $baseline_src | $after_src | $(format_improvement $baseline_src $after_src) |
| scripts   | $baseline_scripts | $after_scripts | $(format_improvement $baseline_scripts $after_scripts) |
| tests     | $baseline_tests | $after_tests | $(format_improvement $baseline_tests $after_tests) |
| **Combined** | **$baseline_combined** | **$after_combined** | **$(format_improvement $baseline_combined $after_combined)** |

## Analysis

### Direct vs. Cascade Effects
EOF

# Calculate direct and cascade effects
individual_baseline=$((baseline_src + baseline_scripts + baseline_tests))
individual_after=$((after_src + after_scripts + after_tests))
direct_improvement=$((individual_baseline - individual_after))
total_improvement=$((baseline_combined - after_combined))
cascade_improvement=$((total_improvement - direct_improvement))

cat <<EOF
- **Direct improvements**: $direct_improvement errors fixed in individual files
- **Cascade improvements**: $cascade_improvement errors eliminated through cross-directory effects
- **Total improvement**: $total_improvement errors eliminated

### Improvement Breakdown
EOF

# Show percentage improvements
if [[ $baseline_combined -gt 0 ]]; then
    total_pct=$(calc_percentage $baseline_combined $after_combined)
    echo "- **Overall reduction**: $total_pct%"
fi

if [[ $baseline_src -gt 0 ]]; then
    src_pct=$(calc_percentage $baseline_src $after_src)
    echo "- **src/ reduction**: $src_pct%"
fi

if [[ $baseline_scripts -gt 0 ]]; then
    scripts_pct=$(calc_percentage $baseline_scripts $after_scripts)
    echo "- **scripts/ reduction**: $scripts_pct%"
fi

if [[ $baseline_tests -gt 0 ]]; then
    tests_pct=$(calc_percentage $baseline_tests $after_tests)
    echo "- **tests/ reduction**: $tests_pct%"
fi

# Note about cross-directory effects
if [[ $cascade_improvement -gt 0 ]]; then
    cat <<EOF

### 🎯 Key Insight
The cascade effect eliminated $cascade_improvement additional errors beyond the direct fixes.
This demonstrates the importance of fixing type errors in core modules that are imported elsewhere.
EOF
elif [[ $cascade_improvement -lt 0 ]]; then
    cat <<EOF

### ⚠️ Warning
The changes introduced ${cascade_improvement#-} new cross-directory errors.
This might indicate that some type annotations are too restrictive or incorrect.
EOF
fi