#!/bin/bash
set -Eeuo pipefail

PROJECT_ROOT="${UNICAL_PROJECT_ROOT:-$HOME/UNICAL_Implementation}"
CONDA_ENV="${UNICAL_JOB_CONDA_ENV:-${UNICAL_CONDA_ENV:-cocolit310}}"
CONDA_SH="${UNICAL_CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
TORCH_VERSION="${UNICAL_TORCH_VERSION:-2.7.1}"
TORCH_INDEX="${UNICAL_TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"

if [[ ! -f "$CONDA_SH" ]]; then
  echo "Conda initialization script not found: $CONDA_SH"
  echo "Set UNICAL_CONDA_SH to the correct conda.sh path and rerun."
  exit 1
fi

source "$CONDA_SH"
if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  conda create -y -n "$CONDA_ENV" python=3.11
fi
conda activate "$CONDA_ENV"

cd "$PROJECT_ROOT"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-hpc.txt
python -m pip install cocolit
# Install the driver-compatible PyTorch build last. This prevents a dependency
# installation from silently replacing it with a newer CUDA runtime.
python -m pip uninstall -y torch torchvision torchaudio || true
python -m pip install --index-url "$TORCH_INDEX" "torch==$TORCH_VERSION"
python -m pip check

mkdir -p "$HOME/logs/unical" "$HOME/.cache/huggingface" \
  "$PROJECT_ROOT/outputs/logs" "$PROJECT_ROOT/data/source" \
  "$HOME/oasis3/MR_T1w" "$HOME/oasis3/AV45_PET"

python - <<'PY'
import shutil
import sys
import torch

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA runtime:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("mri2pet:", shutil.which("mri2pet"))
if torch.version.cuda is None:
    raise SystemExit("This is a CPU-only PyTorch build. Install a CUDA-enabled PyTorch build before submitting GPU jobs.")
if not torch.cuda.is_available():
    print("GPU is not visible on this node. This is expected on many login nodes; the GPU preflight job performs the final check.")
if not shutil.which("mri2pet"):
    raise SystemExit("CoCoLIT installation did not provide the mri2pet command.")
PY

echo "HPC environment setup completed."
