#!/bin/bash
# SMOKE: confirm the transformer_lens backend returns finite role vectors of the right
# hidden_dim BEFORE committing to a 30-run job. Runs a single extraction (--runs 1) on one
# GPU. Success = job exits 0 and runs/report.md / meta.json show a finite hidden_dim matching
# the model. (Sanity checklist step 1 from docs/ndif-setup.md.)
#SBATCH --job-name=rolevec_smoke
#SBATCH --output=logs/rolevec_smoke_%j.out
#SBATCH --error=logs/rolevec_smoke_%j.err
#SBATCH --partition=multigpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --requeue

set -euo pipefail

CONDA_SH="/path/to/miniconda3/etc/profile.d/conda.sh"
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

echo "=========================================="
echo "ROLEVEC SMOKE  |  backend=transformer_lens  |  model=$MODEL  |  runs=1"
echo "Job ID : ${SLURM_JOB_ID:-local}   GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)   Started: $(date)"
echo "=========================================="

python -m rolevec.pipeline \
    --backend transformer_lens \
    --model "$MODEL" \
    --runs 1

echo ""
echo "Finished: $(date) — check runs/ for meta.json hidden_dim before scaling to 30 runs."
