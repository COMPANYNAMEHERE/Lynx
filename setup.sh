#!/usr/bin/env bash
# Simple setup script for Lynx
ENV_NAME="${1:-lynx}"

set -e

echo "Creating conda environment '$ENV_NAME'..."
conda create -n "$ENV_NAME" python=3.11 -y

echo "Installing requirements..."
conda run -n "$ENV_NAME" pip install -r requirements.txt

# Install PyTorch matching the detected CUDA version
CUDA_VERSION=""
if command -v nvidia-smi >/dev/null 2>&1; then
    CUDA_VERSION=$(nvidia-smi | grep -o 'CUDA Version: [0-9.]*' | head -n1 | awk '{print $3}')
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
    conda run -n "$ENV_NAME" pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
else
    echo "Installing PyTorch for $TORCH_TAG..."
    conda run -n "$ENV_NAME" pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/$TORCH_TAG
fi

echo "All done! Activate the environment with:"
echo "  conda activate $ENV_NAME"
echo "Then launch the GUI with:"
echo "  python main.py"
