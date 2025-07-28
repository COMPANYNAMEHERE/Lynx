#!/usr/bin/env bash
# Simple setup script for Lynx
ENV_NAME="${1:-lynx}"

set -e

echo "Creating conda environment '$ENV_NAME'..."
conda create -n "$ENV_NAME" python=3.11 -y

echo "Installing requirements..."
conda run -n "$ENV_NAME" pip install -r requirements.txt

echo "All done! Activate the environment with:"
echo "  conda activate $ENV_NAME"
echo "Then launch the GUI with:"
echo "  python main.py"
