#!/bin/bash
# measure_typing_baseline.sh - Comprehensive baseline measurement for typing experiments
set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to extract error count from mypy output
extract_error_count() {
    local file=$1
    if [[ -f "$file" ]]; then
        head -1 "$file" | jq -r '.error_count // 0'
    else
        echo "0"
    fi
}

# Clean up old error files
echo -e "${YELLOW}🧹 Cleaning up old error files...${NC}"
rm -f .mypy_errors*.jsonl .src_mypy_errors.jsonl .scripts_mypy_errors.jsonl .tests_mypy_errors.jsonl

# Run mypy on individual directories
echo -e "${BLUE}📊 Running individual directory checks...${NC}"

# Check src
echo -e "${BLUE}=== Checking src ===${NC}"
dr-typecheck --output-format jsonl --output-file ./.mypy_errors.jsonl src 2>&1 || true
src_errors=$(extract_error_count ".mypy_errors.jsonl")
mv .mypy_errors.jsonl .src_mypy_errors.jsonl 2>/dev/null || true

# Check scripts
echo -e "${BLUE}=== Checking scripts ===${NC}"
dr-typecheck --output-format jsonl --output-file ./.mypy_errors.jsonl scripts 2>&1 || true
scripts_errors=$(extract_error_count ".mypy_errors.jsonl")
mv .mypy_errors.jsonl .scripts_mypy_errors.jsonl 2>/dev/null || true

# Check tests
echo -e "${BLUE}=== Checking tests ===${NC}"
dr-typecheck --output-format jsonl --output-file ./.mypy_errors.jsonl tests 2>&1 || true
tests_errors=$(extract_error_count ".mypy_errors.jsonl")
mv .mypy_errors.jsonl .tests_mypy_errors.jsonl 2>/dev/null || true

# Calculate individual total
individual_total=$((src_errors + scripts_errors + tests_errors))

# Run combined check
echo -e "${BLUE}=== Checking all directories together ===${NC}"
dr-typecheck --output-format jsonl --output-file ./.mypy_errors.jsonl src scripts tests 2>&1 || true
combined_errors=$(extract_error_count ".mypy_errors.jsonl")

# Calculate cross-directory errors
cross_directory_errors=$((combined_errors - individual_total))

# Output JSON results
timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
cat <<EOF
{
  "timestamp": "$timestamp",
  "individual": {
    "src": $src_errors,
    "scripts": $scripts_errors,
    "tests": $tests_errors,
    "total": $individual_total
  },
  "combined": $combined_errors,
  "cross_directory_errors": $cross_directory_errors,
  "git_commit": "$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')",
  "git_branch": "$(git branch --show-current 2>/dev/null || echo 'unknown')"
}
EOF

# Also display human-readable summary to stderr
{
    echo -e "\n${GREEN}=== SUMMARY ===${NC}"
    echo -e "Individual checks: src=$src_errors, scripts=$scripts_errors, tests=$tests_errors (total=$individual_total)"
    echo -e "Combined check: $combined_errors errors"
    if [[ $cross_directory_errors -gt 0 ]]; then
        echo -e "${YELLOW}⚠️  Cross-directory errors: $cross_directory_errors additional errors${NC}"
    else
        echo -e "${GREEN}✅ No cross-directory errors${NC}"
    fi
} >&2