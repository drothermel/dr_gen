#!/bin/bash
# analyze_import_graph.sh - Analyze import relationships to find high-impact modules
set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Temporary file for import data
IMPORT_DATA=$(mktemp)

# Cleanup on exit
trap "rm -f $IMPORT_DATA" EXIT

# Usage function
usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --top N       Show top N most imported modules (default: 10)"
    echo "  --module M    Show what imports module M"
    echo "  --help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                    # Show top 10 most imported modules"
    echo "  $0 --top 20          # Show top 20"
    echo "  $0 --module metrics   # Show what imports metrics module"
    exit 1
}

# Parse arguments
TOP_N=10
SPECIFIC_MODULE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --top)
            TOP_N="$2"
            shift 2
            ;;
        --module)
            SPECIFIC_MODULE="$2"
            shift 2
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

echo -e "${BLUE}🔍 Analyzing import relationships...${NC}"

# Find all Python imports in the codebase
{
    # Look for 'from X import Y' statements
    rg --no-heading -o 'from\s+(dr_gen\.[a-zA-Z0-9_.]+)\s+import' -r '$1' --type py src/ scripts/ tests/ 2>/dev/null || true
    
    # Look for 'import X' statements
    rg --no-heading -o 'import\s+(dr_gen\.[a-zA-Z0-9_.]+)' -r '$1' --type py src/ scripts/ tests/ 2>/dev/null || true
    
    # Look for relative imports that can be resolved
    # This is more complex and would need directory context, so we'll skip for now
} | sort | uniq -c | sort -nr > "$IMPORT_DATA"

if [[ -z "$SPECIFIC_MODULE" ]]; then
    # Show most imported modules
    echo -e "${GREEN}Top $TOP_N most imported modules:${NC}"
    echo ""
    echo "Count | Module"
    echo "------|-------"
    
    head -n "$TOP_N" "$IMPORT_DATA" | while read -r count module; do
        # Convert module path to file path
        file_path="src/${module//./\/}.py"
        
        # Check if file exists to filter out sub-imports
        if [[ -f "$file_path" ]]; then
            printf "%5d | %s\n" "$count" "$module"
        fi
    done
    
    echo ""
    echo -e "${YELLOW}💡 Tip: Focus MonkeyType tracing on these high-impact modules${NC}"
    echo -e "${YELLOW}Example: ./setup_monkeytype_trace.sh --modules $(head -n 3 "$IMPORT_DATA" | awk '{print $2}' | sed 's/dr_gen\.//g' | tr '\n' ',' | sed 's/,$//')${NC}"
    
else
    # Show what imports a specific module
    echo -e "${GREEN}Files importing $SPECIFIC_MODULE:${NC}"
    echo ""
    
    # Search for imports of the specific module
    pattern="(from|import)\s+(dr_gen\.)?${SPECIFIC_MODULE}"
    rg -l "$pattern" --type py src/ scripts/ tests/ 2>/dev/null | while read -r file; do
        # Get the actual import line for context
        import_line=$(rg -m1 "$pattern" "$file" | sed 's/^[[:space:]]*//')
        echo "$file"
        echo "  └─ $import_line"
    done
fi

# Additional analysis - find circular dependencies
echo ""
echo -e "${BLUE}🔄 Checking for potential circular imports...${NC}"

# This is a simplified check - looks for files that import each other
CIRCULAR_FOUND=false
for file in $(find src -name "*.py" -type f); do
    module=$(echo "$file" | sed 's/src\///; s/\.py$//; s/\//./g')
    
    # Find what this module imports
    imports=$(rg -o "from ($module\.[a-zA-Z0-9_.]+) import" -r '$1' "$file" 2>/dev/null || true)
    
    for imp in $imports; do
        # Check if the imported module imports back
        imp_file="src/${imp//./\/}.py"
        if [[ -f "$imp_file" ]]; then
            if rg -q "from $module import|import $module" "$imp_file" 2>/dev/null; then
                echo -e "${YELLOW}⚠️  Potential circular import: $module ↔ $imp${NC}"
                CIRCULAR_FOUND=true
            fi
        fi
    done
done

if [[ "$CIRCULAR_FOUND" == "false" ]]; then
    echo -e "${GREEN}✅ No circular imports detected${NC}"
fi