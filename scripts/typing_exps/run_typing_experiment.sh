#!/bin/bash
# run_typing_experiment.sh - Main orchestrator for typing experiments
set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Usage function
usage() {
    echo "Usage: $0 <tool> [tool-args]"
    echo ""
    echo "Tools:"
    echo "  autotyping              Apply autotyping with safe defaults"
    echo "  infer-types             Apply infer-types tool"
    echo "  monkeytype              Apply MonkeyType with runtime tracing"
    echo "  combined                Apply multiple tools in sequence"
    echo ""
    echo "Options:"
    echo "  --skip-baseline         Skip baseline measurement (use existing)"
    echo "  --no-branch             Don't create experiment branch"
    echo "  --keep-changes          Don't reset after experiment"
    echo ""
    echo "Examples:"
    echo "  $0 autotyping"
    echo "  $0 infer-types --only return"
    echo "  $0 monkeytype --modules utils,metrics"
    echo "  $0 combined --sequence autotyping,monkeytype"
    exit 1
}

# Check arguments
if [[ $# -lt 1 ]]; then
    usage
fi

# Parse options
TOOL=""
SKIP_BASELINE=false
CREATE_BRANCH=true
KEEP_CHANGES=false
TOOL_ARGS=""
SEQUENCE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-baseline)
            SKIP_BASELINE=true
            shift
            ;;
        --no-branch)
            CREATE_BRANCH=false
            shift
            ;;
        --keep-changes)
            KEEP_CHANGES=true
            shift
            ;;
        --sequence)
            SEQUENCE="$2"
            shift 2
            ;;
        autotyping|infer-types|monkeytype|combined)
            TOOL="$1"
            shift
            TOOL_ARGS="$@"
            break
            ;;
        *)
            echo -e "${RED}❌ Unknown option: $1${NC}"
            usage
            ;;
    esac
done

if [[ -z "$TOOL" ]]; then
    echo -e "${RED}❌ No tool specified${NC}"
    usage
fi

# Generate timestamp and experiment name
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME="${TIMESTAMP}_${TOOL}"
EXPERIMENT_DIR="experiments/${EXPERIMENT_NAME}"

# Create experiment directory
echo -e "${MAGENTA}🧪 Starting experiment: $EXPERIMENT_NAME${NC}"
mkdir -p "$EXPERIMENT_DIR"

# Save current branch
ORIGINAL_BRANCH=$(git branch --show-current)

# Function to cleanup on exit
cleanup() {
    if [[ "$KEEP_CHANGES" == "false" && "$CREATE_BRANCH" == "true" ]]; then
        echo -e "${YELLOW}🧹 Cleaning up...${NC}"
        git checkout "$ORIGINAL_BRANCH" &>/dev/null || true
        git branch -D "exp_${EXPERIMENT_NAME}" &>/dev/null || true
    fi
}

# Set trap for cleanup
trap cleanup EXIT

# Establish baseline
if [[ "$SKIP_BASELINE" == "false" ]]; then
    echo -e "${BLUE}📊 Establishing baseline...${NC}"
    "$SCRIPT_DIR/measure_typing_baseline.sh" > "$EXPERIMENT_DIR/baseline.json"
    echo -e "${GREEN}✅ Baseline saved to $EXPERIMENT_DIR/baseline.json${NC}"
else
    echo -e "${YELLOW}⚠️  Skipping baseline measurement${NC}"
fi

# Create experiment branch
if [[ "$CREATE_BRANCH" == "true" ]]; then
    echo -e "${BLUE}🌿 Creating experiment branch...${NC}"
    git checkout -b "exp_${EXPERIMENT_NAME}" &>/dev/null
fi

# Log experiment metadata
cat > "$EXPERIMENT_DIR/metadata.json" <<EOF
{
  "experiment_name": "$EXPERIMENT_NAME",
  "tool": "$TOOL",
  "tool_args": "$TOOL_ARGS",
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "git_commit": "$(git rev-parse HEAD)",
  "git_branch": "$(git branch --show-current)",
  "sequence": "$SEQUENCE"
}
EOF

# Apply tools based on selection
echo -e "${BLUE}🔧 Applying tool(s)...${NC}"

case $TOOL in
    autotyping)
        "$SCRIPT_DIR/apply_typing_tool.sh" autotyping $TOOL_ARGS 2>&1 | tee "$EXPERIMENT_DIR/application.log"
        ;;
        
    infer-types)
        "$SCRIPT_DIR/apply_typing_tool.sh" infer-types $TOOL_ARGS 2>&1 | tee "$EXPERIMENT_DIR/application.log"
        ;;
        
    monkeytype)
        "$SCRIPT_DIR/setup_monkeytype_trace.sh" $TOOL_ARGS 2>&1 | tee "$EXPERIMENT_DIR/application.log"
        ;;
        
    combined)
        if [[ -z "$SEQUENCE" ]]; then
            SEQUENCE="autotyping,infer-types,monkeytype"
        fi
        
        echo -e "${BLUE}📋 Running sequence: $SEQUENCE${NC}"
        IFS=',' read -ra TOOLS <<< "$SEQUENCE"
        
        for tool in "${TOOLS[@]}"; do
            echo -e "${BLUE}➡️  Applying $tool...${NC}"
            case $tool in
                autotyping)
                    "$SCRIPT_DIR/apply_typing_tool.sh" autotyping 2>&1 | tee -a "$EXPERIMENT_DIR/application.log"
                    ;;
                infer-types)
                    "$SCRIPT_DIR/apply_typing_tool.sh" infer-types 2>&1 | tee -a "$EXPERIMENT_DIR/application.log"
                    ;;
                monkeytype)
                    "$SCRIPT_DIR/setup_monkeytype_trace.sh" 2>&1 | tee -a "$EXPERIMENT_DIR/application.log"
                    ;;
                *)
                    echo -e "${RED}❌ Unknown tool in sequence: $tool${NC}"
                    ;;
            esac
            
            # Measure after each tool in sequence
            echo -e "${BLUE}📊 Measuring after $tool...${NC}"
            "$SCRIPT_DIR/measure_typing_baseline.sh" > "$EXPERIMENT_DIR/after_${tool}.json"
        done
        ;;
esac

# Measure final impact
echo -e "${BLUE}📊 Measuring final impact...${NC}"
"$SCRIPT_DIR/measure_typing_baseline.sh" > "$EXPERIMENT_DIR/after.json"

# Generate comparison report
if [[ -f "$EXPERIMENT_DIR/baseline.json" ]]; then
    echo -e "${BLUE}📈 Generating comparison report...${NC}"
    "$SCRIPT_DIR/compare_typing_results.sh" \
        "$EXPERIMENT_DIR/baseline.json" \
        "$EXPERIMENT_DIR/after.json" \
        > "$EXPERIMENT_DIR/summary.md"
fi

# Save git diff
echo -e "${BLUE}💾 Saving changes...${NC}"
git diff > "$EXPERIMENT_DIR/changes.diff"

# Generate final report
cat > "$EXPERIMENT_DIR/report.md" <<EOF
# Experiment Report: $EXPERIMENT_NAME

## Overview
- **Tool**: $TOOL
- **Arguments**: $TOOL_ARGS
- **Timestamp**: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
- **Git Commit**: $(git rev-parse HEAD)

## Files
- \`baseline.json\` - Initial typing errors
- \`after.json\` - Final typing errors
- \`summary.md\` - Detailed comparison
- \`application.log\` - Tool output
- \`changes.diff\` - Git diff of changes
- \`metadata.json\` - Experiment metadata

## Quick Summary
$(tail -n 20 "$EXPERIMENT_DIR/summary.md" 2>/dev/null || echo "No summary available")
EOF

# Reset if requested
if [[ "$KEEP_CHANGES" == "false" ]]; then
    echo -e "${YELLOW}🔄 Resetting changes...${NC}"
    git checkout -- .
fi

# Final output
echo -e "${GREEN}✅ Experiment complete!${NC}"
echo -e "${MAGENTA}📁 Results saved to: $EXPERIMENT_DIR/${NC}"
echo ""
echo -e "${BLUE}Key files:${NC}"
echo "  - Summary: $EXPERIMENT_DIR/summary.md"
echo "  - Report: $EXPERIMENT_DIR/report.md"
echo "  - Changes: $EXPERIMENT_DIR/changes.diff"
echo ""

# Show key metrics
if [[ -f "$EXPERIMENT_DIR/baseline.json" ]] && [[ -f "$EXPERIMENT_DIR/after.json" ]]; then
    baseline_errors=$(jq -r '.combined' "$EXPERIMENT_DIR/baseline.json")
    after_errors=$(jq -r '.combined' "$EXPERIMENT_DIR/after.json")
    improvement=$((baseline_errors - after_errors))
    
    if [[ $improvement -gt 0 ]]; then
        pct=$(echo "scale=1; ($improvement * 100) / $baseline_errors" | bc)
        echo -e "${GREEN}🎯 Result: $baseline_errors → $after_errors errors (-$improvement, $pct% improvement)${NC}"
    else
        echo -e "${YELLOW}📊 Result: $baseline_errors → $after_errors errors${NC}"
    fi
fi