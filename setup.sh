#!/usr/bin/env bash
# Interactive setup script for Lynx with verbose logging

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR"
LOG_FILE="$SCRIPT_DIR/setup/setup.log"
ENV_NAME="${1:-lynx}"

# Log everything to stdout and the log file
mkdir -p "$(dirname "$LOG_FILE")"
echo "Logging to $LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1
set -x
echo "Running $0 $*"

cat <<EOF
This script will create or update the conda environment '$ENV_NAME'.
It installs Python 3.11, packages from requirements.txt and a matching
PyTorch build for your GPU.
EOF

read -rp "Continue? [y/N] " ans
case "$ans" in
    [yY]*) ;;
    *) echo "Aborted."; exit 1 ;;
esac

echo "Checking existing environment..."
conda env list
if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    echo "Environment '$ENV_NAME' already exists. Updating packages..."
    CREATE_ENV=0
else
    echo "Creating conda environment '$ENV_NAME'..."
    conda create -n "$ENV_NAME" python=3.11 -y
    CREATE_ENV=1
fi

echo "Installing/updating requirements from $REPO_DIR/requirements.txt..."
conda run -n "$ENV_NAME" pip install -r "$REPO_DIR/requirements.txt" -U --progress-bar on
conda run -n "$ENV_NAME" pip list

# Install PyTorch matching the detected CUDA version
echo "Detecting CUDA..."
CUDA_VERSION=""
if command -v nvidia-smi >/dev/null 2>&1; then
    CUDA_VERSION=$(nvidia-smi | grep -o 'CUDA Version: [0-9.]*' | head -n1 | awk '{print $3}')
fi
if [ -z "$CUDA_VERSION" ] && command -v nvcc >/dev/null 2>&1; then
    CUDA_VERSION=$(nvcc --version | grep -o -E 'release [0-9]+\.[0-9]+' | head -n1 | awk '{print $2}')
fi
if [ -n "$CUDA_VERSION" ]; then
    echo "Found CUDA version $CUDA_VERSION"
else
    echo "No CUDA detected"
fi
TORCH_TAG="cpu"
if [ -n "$CUDA_VERSION" ]; then
    case "$CUDA_VERSION" in
        12.*) TORCH_TAG="cu121" ;;
        11.*) TORCH_TAG="cu118" ;;
    esac
fi

if [ "$TORCH_TAG" = "cpu" ]; then
    echo "Installing CPU-only PyTorch..."
    conda run -n "$ENV_NAME" pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --progress-bar on
else
    echo "Installing PyTorch for $TORCH_TAG..."
    conda run -n "$ENV_NAME" pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/$TORCH_TAG --progress-bar on
fi

# Optional check for outdated packages if environment already existed
if [ "$CREATE_ENV" -eq 0 ]; then
    echo "Checking for outdated packages..."
    OUTDATED=$(conda run -n "$ENV_NAME" pip list --outdated --format=freeze)
    if [ -n "$OUTDATED" ]; then
        echo "$OUTDATED" | while IFS= read -r line; do echo " - $line"; done
        echo "Updating outdated packages..."
        conda run -n "$ENV_NAME" pip install -U $(echo "$OUTDATED" | cut -d= -f1 | tr '\n' ' ') --progress-bar on
        conda run -n "$ENV_NAME" pip list
    else
        echo "All packages up to date."
    fi
fi

echo "All done! Activate the environment with:\n  conda activate $ENV_NAME"
echo "Then launch the GUI with:\n  python main.py"
