#!/bin/bash
set -Eeuo pipefail

PROJECT_ROOT="${UNICAL_PROJECT_ROOT:-$HOME/UNICAL_Implementation}"
CONFIG="${UNICAL_CONFIG:-$PROJECT_ROOT/configs/oasis3_unical_public.example.yaml}"
LOG_ROOT="${UNICAL_LOG_ROOT:-$HOME/logs/unical}"
CONDA_ENV="${UNICAL_JOB_CONDA_ENV:-cocolit310}"
CONDA_SH="${UNICAL_CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"

if [[ "$CONFIG" != /* ]]; then
  CONFIG="$PROJECT_ROOT/$CONFIG"
fi
export UNICAL_CONFIG="$CONFIG"

if [[ ! -d "$PROJECT_ROOT" ]]; then
  echo "Project root not found: $PROJECT_ROOT"
  exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "Config file not found or is not a regular file: $CONFIG"
  exit 1
fi
mkdir -p "$LOG_ROOT" "$PROJECT_ROOT/outputs/logs"

if [[ ! -f "$CONDA_SH" ]]; then
  echo "Conda initialization script not found: $CONDA_SH"
  exit 1
fi
source "$CONDA_SH"
conda activate "$CONDA_ENV"

if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
  echo "Requested conda environment '$CONDA_ENV' was not activated; active environment is '${CONDA_DEFAULT_ENV:-unset}'."
  exit 1
fi

EXPECTED_PREFIX="$(conda env list | awk -v env="$CONDA_ENV" '$1 == env {print $NF; exit}')"
ACTIVE_PYTHON="$(python -c 'import sys; print(sys.executable)')"
if [[ -z "$EXPECTED_PREFIX" || "$ACTIVE_PYTHON" != "$EXPECTED_PREFIX/bin/python" ]]; then
  echo "Conda environment/Python mismatch."
  echo "Requested environment: $CONDA_ENV"
  echo "Expected prefix: ${EXPECTED_PREFIX:-not found}"
  echo "Active Python: $ACTIVE_PYTHON"
  exit 1
fi

export UNICAL_CONDA_ENV="$CONDA_ENV"
cd "$PROJECT_ROOT"

export PYTHONUNBUFFERED=1
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export WANDB_MODE="${WANDB_MODE:-offline}"

if ulimit -n 8192 2>/dev/null; then
  :
else
  echo "Warning: could not raise open-file limit; current limit is $(ulimit -n)."
fi

echo "===== JOB CONTEXT ====="
echo "Date: $(date -Is)"
echo "Host: $(hostname)"
echo "Job: ${SLURM_JOB_ID:-interactive}"
echo "Array task: ${SLURM_ARRAY_TASK_ID:-none}"
echo "Project: $PROJECT_ROOT"
echo "Config: $CONFIG"
echo "Requested conda environment: $CONDA_ENV"
echo "Active conda environment: ${CONDA_DEFAULT_ENV:-unset}"
echo "Conda prefix: ${CONDA_PREFIX:-unset}"
echo "Python: $(which python)"
echo "Open-file limit: $(ulimit -n)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "SLURM_JOB_GPUS: ${SLURM_JOB_GPUS:-unset}"
echo "SLURM_GPUS_ON_NODE: ${SLURM_GPUS_ON_NODE:-unset}"
python -V
df -h "$PROJECT_ROOT" || true
nvidia-smi || true
nvidia-smi pmon -c 1 || true
conda list > "$PROJECT_ROOT/outputs/logs/conda_environment.${SLURM_JOB_ID:-interactive}.txt"

MONITOR_PID=""
if [[ "${UNICAL_DISABLE_GPU_MONITOR:-0}" != "1" ]]; then
  GPU_MONITOR_SECONDS="${UNICAL_GPU_MONITOR_SECONDS:-5}"
  (
    echo "timestamp,name,utilization.gpu [%],memory.used [MiB],memory.total [MiB],power.draw [W]"
    while true; do
      nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,power.draw \
        --format=csv,noheader,nounits
      sleep "$GPU_MONITOR_SECONDS"
    done
  ) > "$LOG_ROOT/gpu_watch.${SLURM_JOB_ID:-interactive}.${SLURM_ARRAY_TASK_ID:-0}.csv" 2>&1 &
  MONITOR_PID=$!
fi

cleanup() {
  code=$?
  if [[ -n "$MONITOR_PID" ]]; then
    kill "$MONITOR_PID" 2>/dev/null || true
  fi
  echo "Finished $(date -Is), exit code $code"
  exit "$code"
}
trap cleanup EXIT
