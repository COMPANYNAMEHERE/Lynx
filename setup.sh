#!/usr/bin/env bash
# Interactive setup script for Lynx with verbose logging

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR"
LOG_FILE="$SCRIPT_DIR/setup/setup.log"
ENV_NAME="${1:-lynx}"

# Colours (fall back to empty strings if tput is unavailable)
bold=$(tput bold 2>/dev/null || true)
red=$(tput setaf 1 2>/dev/null || true)
green=$(tput setaf 2 2>/dev/null || true)
yellow=$(tput setaf 3 2>/dev/null || true)
blue=$(tput setaf 4 2>/dev/null || true)
reset=$(tput sgr0 2>/dev/null || true)

# Abort handler to print a helpful message
trap 'echo "${red}Setup failed - check $LOG_FILE for details${reset}"' ERR

# Ensure conda is available early
if ! command -v conda >/dev/null 2>&1; then
    echo "${red}conda not found. Please install Miniconda or Anaconda and ensure 'conda' is on your PATH.${reset}"
    read -rp "Press Enter to exit" _
    exit 1
fi

# Log everything to stdout and the log file
mkdir -p "$(dirname "$LOG_FILE")"
echo "${bold}Logging to $LOG_FILE${reset}"
exec > >(tee -a "$LOG_FILE") 2>&1

cat <<EOF
${bold}${blue}This script will create or update the conda environment '$ENV_NAME'.${reset}
It installs Python 3.11, packages from requirements.txt and a matching
PyTorch build for your GPU.
EOF

read -rp "Continue? [y/N] " ans
case "$ans" in
    [yY]*) ;;
    *)
        echo "Aborted."
        read -rp "Press Enter to exit" _
        exit 1
        ;;
esac

echo "Checking existing environment..."
conda env list
if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    echo "${yellow}Environment '$ENV_NAME' already exists.${reset}"
    read -rp "Reset it completely? [y/N] " resp
    if [[ $resp =~ ^[yY] ]]; then
        echo "Removing existing environment..."
        conda env remove -n "$ENV_NAME" -y
        echo "Creating conda environment '$ENV_NAME'..."
        conda create -n "$ENV_NAME" python=3.11 -y
        CREATE_ENV=1
    else
        echo "Updating packages in existing environment..."
        CREATE_ENV=0
    fi
else
    echo "Creating conda environment '$ENV_NAME'..."
    conda create -n "$ENV_NAME" python=3.11 -y
    CREATE_ENV=1
fi

echo "Installing/updating requirements from $REPO_DIR/requirements.txt..."
echo "Installing Python packages..."
conda run -n "$ENV_NAME" pip install -r "$REPO_DIR/requirements.txt" -U --quiet
conda run -n "$ENV_NAME" pip list

# Install PyTorch matching the detected CUDA version
echo "Detecting CUDA..."
CUDA_VERSION=""
GPU_DETECTED=0
if command -v nvidia-smi >/dev/null 2>&1; then
    CUDA_VERSION=$(nvidia-smi | grep -o 'CUDA Version: [0-9.]*' | head -n1 | awk '{print $3}')
    GPU_DETECTED=1
fi
if [ -z "$CUDA_VERSION" ] && command -v nvcc >/dev/null 2>&1; then
    CUDA_VERSION=$(nvcc --version | grep -o -E 'release [0-9]+\.[0-9]+' | head -n1 | awk '{print $2}')
    GPU_DETECTED=1
fi
if [ -n "$CUDA_VERSION" ]; then
    echo "Found CUDA version $CUDA_VERSION"
else
    echo "No CUDA version detected"
fi
TORCH_TAG="cpu"
if [ -n "$CUDA_VERSION" ]; then
    case "$CUDA_VERSION" in
        12.*) TORCH_TAG="cu121" ;;
        11.*) TORCH_TAG="cu118" ;;
        10.*) TORCH_TAG="cu102" ;;
    esac
fi

if [ "$TORCH_TAG" = "cpu" ]; then
    echo "Installing CPU-only PyTorch..."
    conda run -n "$ENV_NAME" pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet
else
    echo "Installing PyTorch for $TORCH_TAG..."
    conda run -n "$ENV_NAME" pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/$TORCH_TAG --quiet
fi

# Optional check for outdated packages if environment already existed
if [ "$CREATE_ENV" -eq 0 ]; then
    echo "Checking for outdated packages..."
    OUTDATED=$(conda run -n "$ENV_NAME" pip list --outdated --format=freeze)
    if [ -n "$OUTDATED" ]; then
        echo "$OUTDATED" | while IFS= read -r line; do echo " - $line"; done
        echo "Updating outdated packages..."
        conda run -n "$ENV_NAME" pip install -U $(echo "$OUTDATED" | cut -d= -f1 | tr '\n' ' ') --quiet
        conda run -n "$ENV_NAME" pip list
    else
        echo "All packages up to date."
    fi
fi

PYTORCH_VERSION=$(conda run -n "$ENV_NAME" python -c "import torch,sys;sys.stdout.write(torch.__version__)")
CUDA_AVAIL=$(conda run -n "$ENV_NAME" python -c "import torch,sys;sys.stdout.write('yes' if torch.cuda.is_available() else 'no')")

echo -e "${green}All done!${reset}"
echo "Next steps:"
echo "  1. conda activate $ENV_NAME"
echo "  2. python main.py"
echo
if [ "$CUDA_AVAIL" = "yes" ]; then
    echo -e "${green}PyTorch CUDA support detected.${reset}"
else
    echo -e "${yellow}PyTorch reports no CUDA support.${reset}"
    if [ $GPU_DETECTED -eq 1 ] && [ "$TORCH_TAG" != "cpu" ]; then
        echo "Reinstall with CUDA via:"
        echo "  conda run -n $ENV_NAME pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/$TORCH_TAG"
    fi
fi
if command -v nvidia-smi >/dev/null 2>&1 || command -v nvcc >/dev/null 2>&1; then
    echo -e "${green}GPU hardware detected on your system.${reset}"
else
    echo -e "${red}No NVIDIA GPU detected.${reset}"
fi

read -rp "Press Enter to exit" _
