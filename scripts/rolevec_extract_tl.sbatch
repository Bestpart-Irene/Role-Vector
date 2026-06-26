#!/bin/bash
# FULL: 30-run TransformerLens extraction + the run_all validation gate. Run only after
# rolevec_smoke_1gpu.sbatch passes. Writes runs/report.md and exits non-zero if any required
# check fails (Q1 domain-sensitivity becomes required once a real Claude judge is configured).
#SBATCH --job-name=rolevec_extract
#SBATCH --output=logs/rolevec_extract_%j.out
#SBATCH --error=logs/rolevec_extract_%j.err
#SBATCH --partition=multigpu
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=12
#SBATCH --time=08:00:00
#SBATCH --requeue

set -euo pipefail

CONDA_SH="${CONDA_SH:-/path/to/miniconda3/etc/profile.d/conda.sh}"  # point at your cluster's conda.sh
ENV_PATH="${ROLEVEC_ENV:-rolevec}"
if [ -f "$CONDA_SH" ]; then source "$CONDA_SH"; conda activate "$ENV_PATH"; fi

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

# Pull NDIF_API_KEY / ANTHROPIC_API_KEY / ROLEVEC_MODEL from .env if present (gitignored).
if [ -f .env ]; then set -a; . ./.env; set +a; fi

export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HOME="/scratch/${USER}/hf_cache"
export HF_DATASETS_CACHE="/scratch/${USER}/hf_cache/datasets"
mkdir -p "$HF_HOME"

MODEL="${MODEL:-${ROLEVEC_MODEL:-Qwen/Qwen2.5-1.5B}}"
RUNS="${RUNS:-30}"
CONSTRUCTION="${CONSTRUCTION:-balanced}"

echo "=========================================="
echo "ROLEVEC EXTRACT  |  backend=transformer_lens  |  model=$MODEL"
echo "runs=$RUNS  construction=$CONSTRUCTION  |  Job ${SLURM_JOB_ID:-local}  |  $(date)"
echo "ANTHROPIC judge: $([ -n "${ANTHROPIC_API_KEY:-}" ] && echo enabled || echo 'dummy heuristic (Q1 reported-only)')"
echo "=========================================="

python -m rolevec.run_all \
    --backend transformer_lens \
    --model "$MODEL" \
    --runs "$RUNS" \
    --construction "$CONSTRUCTION"

echo ""
echo "Finished: $(date) — see runs/report.md (gate exit code: $?)."
