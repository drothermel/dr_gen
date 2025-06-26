#\!/bin/bash

# Function to check types by directory
check_types_by_dir() {
    local total=0
    
    # Clean up any existing error files first
    echo "🧹 Cleaning up old error files..."
    rm -f .src_mypy_errors.jsonl .scripts_mypy_errors.jsonl .tests_mypy_errors.jsonl .mypy_errors.jsonl
    echo
    
    for dir in src scripts tests; do 
        if [ -d "$dir" ]; then
            echo "=== Checking $dir ==="
            dr-typecheck --output-format jsonl --output-file ".${dir}_mypy_errors.jsonl" "$dir"
            
            if [ -f ".${dir}_mypy_errors.jsonl" ]; then
                error_count=$(head -n 1 ".${dir}_mypy_errors.jsonl" | jq -r '.error_count // 0')
                echo "Errors in $dir: $error_count"
                total=$((total + error_count))
            else
                echo "Failed to generate error file for $dir"
            fi
            echo
        else
            echo "Directory $dir not found"
        fi
    done
    
    echo "=== TOTAL ERRORS (individual): $total ==="
    
    # Also run combined check
    echo
    echo "=== Checking all directories together ==="
    dr-typecheck --output-format jsonl --output-file ".mypy_errors.jsonl" src scripts tests
    combined_count=$(head -n 1 ".mypy_errors.jsonl" | jq -r '.error_count // 0')
    echo "Errors (combined): $combined_count"
    
    if [ $combined_count -ne $total ]; then
        echo "⚠️  Cross-directory errors detected: $((combined_count - total)) additional errors"
    fi
}

# Run it
check_types_by_dir
