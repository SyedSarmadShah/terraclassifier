#!/bin/bash
# Simple wrapper to run predict_image.py with virtual environment

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate virtual environment
source "$SCRIPT_DIR/lulc_env/bin/activate"

# Run the prediction script with the correct Python
python "$SCRIPT_DIR/predict_image.py" "$@"
