#!/bin/bash
# setup_monkeytype_trace.sh - Setup and run MonkeyType tracing for type inference
set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default modules to focus on
DEFAULT_MODULES="utils,metrics,model"

# Usage function
usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --modules <list>   Comma-separated list of module names to focus on"
    echo "                     (default: $DEFAULT_MODULES)"
    echo "  --all             Trace all modules (may be slower)"
    echo "  --no-apply        Only generate stubs, don't apply them"
    echo "  --help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                           # Use default modules"
    echo "  $0 --modules utils,display   # Focus on specific modules"
    echo "  $0 --all                     # Trace everything"
    exit 1
}

# Parse arguments
MODULES=$DEFAULT_MODULES
APPLY_STUBS=true
TRACE_ALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --modules)
            MODULES="$2"
            shift 2
            ;;
        --all)
            TRACE_ALL=true
            shift
            ;;
        --no-apply)
            APPLY_STUBS=false
            shift
            ;;
        --help|-h)
            usage
            ;;
        *)
            echo -e "${RED}❌ Unknown option: $1${NC}"
            usage
            ;;
    esac
done

# Check if MonkeyType is installed
if ! uv run python -c 'import monkeytype' &>/dev/null; then
    echo -e "${RED}❌ Error: monkeytype is not installed${NC}"
    echo -e "${YELLOW}Please install with: uv add --dev monkeytype${NC}"
    exit 1
fi

# Check if pytest is available
if ! uv run python -c 'import pytest' &>/dev/null; then
    echo -e "${RED}❌ Error: pytest is not installed${NC}"
    echo -e "${YELLOW}Please install with: uv add --dev pytest${NC}"
    exit 1
fi

# Clean up old database
echo -e "${YELLOW}🧹 Cleaning up old MonkeyType database...${NC}"
rm -f monkeytype.sqlite3

# Set up environment
export MONKEYTYPE_TRACE_MODULES="dr_gen"

echo -e "${BLUE}📊 Running tests with MonkeyType tracing...${NC}"
echo -e "${YELLOW}Note: This will run tests sequentially (no parallelization)${NC}"

# Run tests with MonkeyType
uv run monkeytype run -m pytest tests/ -v -n0 || {
    echo -e "${YELLOW}⚠️  Some tests failed, but continuing with type inference${NC}"
}

echo -e "${GREEN}✅ Test execution complete${NC}"

# Process modules
if [[ "$TRACE_ALL" == "true" ]]; then
    echo -e "${BLUE}🔍 Finding all traced modules...${NC}"
    # Get all modules that were traced
    TRACED_MODULES=$(uv run monkeytype list-modules | grep "^dr_gen" | head -20)
else
    # Convert comma-separated list to individual modules
    echo -e "${BLUE}🔍 Processing specified modules: $MODULES${NC}"
    TRACED_MODULES=""
    IFS=',' read -ra MODULE_ARRAY <<< "$MODULES"
    for module in "${MODULE_ARRAY[@]}"; do
        # Add dr_gen prefix if not present
        if [[ "$module" != dr_gen.* ]]; then
            full_module="dr_gen.$module"
        else
            full_module="$module"
        fi
        
        # Check if module was traced
        if uv run monkeytype list-modules | grep -q "^$full_module"; then
            TRACED_MODULES="$TRACED_MODULES$full_module\n"
        else
            echo -e "${YELLOW}⚠️  Module not traced: $full_module${NC}"
        fi
    done
fi

# Generate and optionally apply stubs
echo -e "${BLUE}🔧 Processing traced modules...${NC}"
echo -e "$TRACED_MODULES" | while read -r module; do
    if [[ -z "$module" ]]; then
        continue
    fi
    
    echo -e "${BLUE}Processing $module...${NC}"
    
    # Generate stub
    stub_file="${module//.//}_stub.py"
    uv run monkeytype stub "$module" > "$stub_file" 2>/dev/null || {
        echo -e "${YELLOW}⚠️  Could not generate stub for $module${NC}"
        continue
    }
    
    # Show stub size
    if [[ -f "$stub_file" ]]; then
        line_count=$(wc -l < "$stub_file")
        echo -e "${GREEN}✅ Generated stub: $stub_file ($line_count lines)${NC}"
        
        if [[ "$APPLY_STUBS" == "true" ]]; then
            # Apply the stub
            uv run monkeytype apply "$module" || {
                echo -e "${YELLOW}⚠️  Could not apply stub for $module${NC}"
            }
            echo -e "${GREEN}✅ Applied types to $module${NC}"
        fi
        
        # Clean up stub file
        rm -f "$stub_file"
    fi
done

# Run auto-fixes
if [[ "$APPLY_STUBS" == "true" ]]; then
    echo -e "${BLUE}🔧 Running auto-fixes...${NC}"
    lint_fix || {
        echo -e "${YELLOW}⚠️  Warning: lint_fix reported issues${NC}"
    }
fi

# Summary
echo -e "${GREEN}✅ MonkeyType tracing complete${NC}"
if [[ "$APPLY_STUBS" == "true" ]]; then
    echo -e "${BLUE}📋 Types have been applied to source files${NC}"
else
    echo -e "${BLUE}📋 Stubs generated but not applied (--no-apply mode)${NC}"
fi
echo -e "${YELLOW}💡 Tip: Run './measure_typing_baseline.sh' to see the improvement${NC}"