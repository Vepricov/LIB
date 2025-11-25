#!/bin/bash

# Directory containing this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run all scripts matching pattern
for script in "$SCRIPT_DIR"/deberta_full_*.sh; do
    if [[ -x "$script" ]]; then
        echo "Running $script"
        "$script"
    else
        echo "Skipping $script (not executable)"
    fi
done

# Run all scripts matching pattern
for script in "$SCRIPT_DIR"/deberta_lora_*.sh; do
    if [[ -x "$script" ]]; then
        echo "Running $script"
        "$script"
    else
        echo "Skipping $script (not executable)"
    fi
done