#!/bin/bash
# Role Vector Validation — run the full roadmap on [redacted]'s SLURM cluster (local
# TransformerLens path, GPU). Conventions copied from the Reinforcement-Casino project.
#
# Usage:
#   ssh cluster                 # SSH alias is already configured (no key to "sync")
#   # one-time: git clone the repo to /home/user/Role-Vector and create the env (see below)
#   cd /home/user/Role-Vector
#   sbatch scripts/run_on_cluster.sbatch
#
#SBATCH --job-name=rolevec
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/rolevec_%j.out
#SBATCH --error=logs/rolevec_%j.err
#SBATCH --requeue

set -euo pipefail

# ---- config (edit these) ----
REPO="/home/user/Role-Vector"
ENV_PREFIX="/scratch/user/conda_envs/rolevec"
CONDA_SH="/path/to/miniconda3/etc/profile.d/conda.sh"
MODEL="${ROLEVEC_MODEL:-Qwen/Qwen2.5-1.5B}"   # start small; scale up after a --runs 1 sanity check
RUNS="${RUNS:-30}"

echo "==== rolevec job $SLURM_JOB_ID on $SLURMD_NODENAME @ $(date) ===="

# ---- env ----
source "$CONDA_SH"
if [ ! -d "$ENV_PREFIX" ]; then
  echo "creating conda env at $ENV_PREFIX"
  conda create --prefix "$ENV_PREFIX" python=3.11 -y
  conda activate "$ENV_PREFIX"
  pip install -e "$REPO"                  # rolevec + numpy/pyyaml
  pip install "transformer-lens>=2.0" "torch>=2.2" transformers anthropic
else
  conda activate "$ENV_PREFIX"
fi

# ---- caches + secrets (set HF_TOKEN / ANTHROPIC_API_KEY in your ~/.bashrc, NOT in git) ----
export HF_HOME="/scratch/user/hf_cache"
export HF_DATASETS_CACHE="/scratch/user/hf_cache/datasets"
export PYTHONUNBUFFERED=1
: "${ANTHROPIC_API_KEY:?set ANTHROPIC_API_KEY for the judge (export in ~/.bashrc)}"

cd "$REPO"
mkdir -p logs

# ---- run: full roadmap + validation gate on a real model ----
PYTHONPATH=src python -m rolevec.run_all \
    --backend transformer_lens --model "$MODEL" --runs "$RUNS"

echo "==== done @ $(date) — see runs/report.md ===="
