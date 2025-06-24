#!/bin/bash
# apply_typing_tool.sh - Standardized application of typing tools
set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Usage function
usage() {
    echo "Usage: $0 <tool> [tool-specific-args]"
    echo ""
    echo "Tools:"
    echo "  autotyping    - Apply autotyping with --safe --none-return flags"
    echo "  infer-types   - Apply infer-types tool"
    echo "  monkeytype    - Apply MonkeyType (requires setup_monkeytype_trace.sh)"
    echo ""
    echo "Examples:"
    echo "  $0 autotyping"
    echo "  $0 infer-types --only return"
    echo "  $0 monkeytype --modules utils,metrics"
    exit 1
}

# Check arguments
if [[ $# -lt 1 ]]; then
    usage
fi

TOOL=$1
shift
ARGS=$@

# Default directories to apply tools to
DIRECTORIES="src scripts tests"

# Log file for command tracking
LOG_FILE="typing_tool_application.log"

# Function to log commands
log_command() {
    local cmd=$1
    echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] $cmd" >> "$LOG_FILE"
    echo -e "${BLUE}🔧 Running: $cmd${NC}"
}

# Function to check if tool is installed
check_tool() {
    local tool=$1
    local check_cmd=$2
    
    if ! eval "$check_cmd" &>/dev/null; then
        echo -e "${RED}❌ Error: $tool is not installed${NC}"
        echo -e "${YELLOW}Please install with: uv add --dev $tool${NC}"
        exit 1
    fi
}

# Apply the specified tool
case $TOOL in
    autotyping)
        check_tool "autotyping" "uv run python -c 'import autotyping'"
        
        # Apply autotyping with safe flags to all directories
        for dir in $DIRECTORIES; do
            if [[ -d "$dir" ]]; then
                cmd="uv run autotyping --safe --none-return $dir $ARGS"
                log_command "$cmd"
                eval "$cmd" || {
                    echo -e "${RED}❌ Failed to apply autotyping to $dir${NC}"
                    exit 1
                }
                echo -e "${GREEN}✅ Applied autotyping to $dir${NC}"
            fi
        done
        ;;
        
    infer-types)
        check_tool "infer-types" "uv run python -m infer_types --help"
        
        # Apply infer-types to all directories
        # Note: infer-types requires directories at the end after --
        cmd="uv run python -m infer_types $ARGS -- $DIRECTORIES"
        log_command "$cmd"
        eval "$cmd" || {
            echo -e "${RED}❌ Failed to apply infer-types${NC}"
            exit 1
        }
        echo -e "${GREEN}✅ Applied infer-types to all directories${NC}"
        ;;
        
    monkeytype)
        check_tool "monkeytype" "uv run python -c 'import monkeytype'"
        
        # MonkeyType requires special handling
        echo -e "${YELLOW}⚠️  MonkeyType requires setup_monkeytype_trace.sh${NC}"
        echo -e "${YELLOW}Run: ./setup_monkeytype_trace.sh $ARGS${NC}"
        exit 1
        ;;
        
    *)
        echo -e "${RED}❌ Unknown tool: $TOOL${NC}"
        usage
        ;;
esac

# Run auto-fixes after tool application
echo -e "${BLUE}🔧 Running auto-fixes...${NC}"
lint_fix_cmd="lint_fix"
log_command "$lint_fix_cmd"
eval "$lint_fix_cmd" || {
    echo -e "${YELLOW}⚠️  Warning: lint_fix reported issues${NC}"
}

echo -e "${GREEN}✅ Tool application complete${NC}"
echo -e "${BLUE}📋 Commands logged to: $LOG_FILE${NC}"