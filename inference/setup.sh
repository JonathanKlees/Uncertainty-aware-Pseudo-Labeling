#!/bin/bash

set -e # Exit on Error

# Directory of this script (inference/), so the repo can be cloned anywhere.
INFERENCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Searching for conda base path"
CONDA_BASE=$(conda info --base)

ENVIRONMENT_NAME="open_vocabs"

if [ -z "$CONDA_BASE" ]; then
  echo "Error: 'conda' has not been found. Is it in your PATH?"
  exit 1
fi

echo "Conda found in: $CONDA_BASE"

echo "Creating a conda environment: $ENVIRONMENT_NAME"
if conda info --envs | grep -q "$ENVIRONMENT_NAME"; then
    conda env remove -n $ENVIRONMENT_NAME -y
fi

conda create -n $ENVIRONMENT_NAME python=3.10 -y
echo "Environment created"

sleep 1

CONDA_ACTIVATE_SCRIPT="$CONDA_BASE/etc/profile.d/conda.sh"
source "$CONDA_ACTIVATE_SCRIPT"
conda activate $ENVIRONMENT_NAME
echo "Environment activated"

sleep 1

conda install -c "nvidia/label/cuda-11.8.0" \
        cuda-nvcc \
        cuda-cudart-dev \
        cuda-libraries-dev \
        "gxx_linux-64=11.*" \
        "gcc_linux-64=11.*" \
        sysroot_linux-64 \
        -y
echo "CUDA Dependencies installed successfully"
sleep 1

echo "Setting Environment Variables"
export CUDA_HOME=$CONDA_PREFIX
export CPATH=$CONDA_PREFIX/include:$CONDA_PREFIX/targets/x86_64-linux/include:$CPATH
export LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
echo "Environment Variables set successfully"

sleep 2

cd ~

GD_PATH="$HOME/GroundingDINO"

# The CUDA sources are patched below, so start from an unmodified checkout.
if [ -d "$GD_PATH" ]; then
  echo "Resetting existing GroundingDINO checkout at $GD_PATH..."
  git -C "$GD_PATH" checkout -- .
else
  echo "Cloning official GroundingDINO repo..."
  git clone https://github.com/IDEA-Research/GroundingDINO.git "$GD_PATH"
  echo "GroundingDINO repo cloned successfully"
fi

cd ~/GroundingDINO

OLD_REQUIREMENTS="$HOME/GroundingDINO/requirements.txt"
if [ -f "$OLD_REQUIREMENTS" ]; then
  rm requirements.txt
fi

# PyTorch Installation
echo "Installing PyTorch..."
pip install numpy==1.26.4 torch torchvision --index-url https://download.pytorch.org/whl/cu118

NEW_REQUIREMENT="$INFERENCE_DIR/requirements_inference.txt"
echo "Copying new requirements from $NEW_REQUIREMENT"
cp "$NEW_REQUIREMENT" ./requirements.txt

pip install -r requirements.txt

# --- Patch for PyTorch >= 2.1 compatibility ---
echo "Applying Patch for Modern PyTorch Compatibility..."
CUDA_SOURCE="groundingdino/models/GroundingDINO/csrc/MsDeformAttn/ms_deform_attn_cuda.cu"
CUDA_HEADER="groundingdino/models/GroundingDINO/csrc/MsDeformAttn/ms_deform_attn.h"

# Step 1: replace ".type().is_cuda()" with ".is_cuda()" (must run before step 2)
if [ -f "$CUDA_SOURCE" ]; then
    sed -i 's/\.type()\.is_cuda()/.is_cuda()/g' "$CUDA_SOURCE"
    echo "Patched .is_cuda() checks in Source."
fi

if [ -f "$CUDA_HEADER" ]; then
    sed -i 's/\.type()\.is_cuda()/.is_cuda()/g' "$CUDA_HEADER"
    echo "Patched .is_cuda() checks in Header."
fi

# Step 2: replace the remaining ".type()" calls with ".scalar_type()" (needed by the
# AT_DISPATCH macros)
if [ -f "$CUDA_SOURCE" ]; then
    sed -i 's/\.type()/.scalar_type()/g' "$CUDA_SOURCE"
    echo "Patched remaining .type() calls to .scalar_type() in Source."
fi
# --- end of patch ---

echo "Building GroundingDINO..."
export MAX_JOBS=4
pip install -e . --no-build-isolation

cd ~

echo "GroundingDINO has been built successfully!"
echo "Installing Detectron2..."

sleep 3

DETECTRON_PATH="$HOME/detectron2"

if [ -d "$DETECTRON_PATH" ]; then
  echo "Detectron2 repo has been cloned before, skipping this step..."
else
  echo "Cloning official Detectron2 repo..."
  git clone https://github.com/facebookresearch/detectron2.git
  echo "Detectron2 repo cloned successfully"
fi

cd ~/detectron2

pip install -e . --no-build-isolation

# Config and weights go where inference.sh expects them (configs/, checkpoints/).
mkdir -p "$INFERENCE_DIR/configs" "$INFERENCE_DIR/checkpoints"

echo "Downloading GroundingDINO config and weights"
# Use curl to avoid LD_LIBRARY_PATH libffi conflict with system wget on some distros
curl -L -o "$INFERENCE_DIR/configs/GroundingDINO_SwinB_cfg.py" \
    https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinB.cfg.py
if [ ! -f "$INFERENCE_DIR/checkpoints/groundingdino_swinb_cogcoor.pth" ]; then
  curl -L -o "$INFERENCE_DIR/checkpoints/groundingdino_swinb_cogcoor.pth" \
      https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth
fi
echo "Models downloaded successfully"

sleep 4

echo "Installing CLIP (necessary for YOLO-World)"
pip install git+https://github.com/openai/CLIP.git
echo "CLIP installed successfully"

MODEL_PATH_FOR_YW_AND_OWL="$INFERENCE_DIR/YW_OWL_Model_Cache_dir"
mkdir -p "$MODEL_PATH_FOR_YW_AND_OWL"

conda env config vars set MODEL_CACHE_DIR="$MODEL_PATH_FOR_YW_AND_OWL" -n $ENVIRONMENT_NAME

pip install ultralytics
echo "Ultralytics installed successfully"

echo "--------------------------------------------------------"
echo "OPTIMIZATION: Fixing ONNX Runtime for GPU Support..."
pip uninstall -y onnxruntime onnxruntime-gpu
pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/pypi/simple/
echo "ONNX Runtime GPU (CUDA 11 compatible) installed successfully."
echo "--------------------------------------------------------"

echo "--------------------------------------------------------"
echo "Installation completed!"
echo "IMPORTANT: Please reactivate your environment."
echo "Run: conda activate $ENVIRONMENT_NAME"
echo "--------------------------------------------------------"