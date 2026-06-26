#!/bin/bash
# Ship this project to the cluster and submit a job — run from your laptop, repo root or scripts/.
#   MODEL=Qwen/Qwen2.5-1.5B bash scripts/transfer_launch.sh smoke    # rolevec_smoke_1gpu.sbatch
#   MODEL=Qwen/Qwen2.5-1.5B bash scripts/transfer_launch.sh extract  # rolevec_extract_tl.sbatch
#
# Connection comes from an ~/.ssh/config host alias you define (override with SSH_HOST=...).
# The private key never leaves ~/.ssh and is never rsync'd.
set -euo pipefail

SSH_HOST="${SSH_HOST:-cluster}"
REMOTE_DIR="${REMOTE_DIR:-role-vector}"
JOB="${1:-smoke}"

case "$JOB" in
    smoke)   SBATCH="scripts/rolevec_smoke_1gpu.sbatch" ;;
    extract) SBATCH="scripts/rolevec_extract_tl.sbatch" ;;
    *) echo "usage: $0 {smoke|extract}"; exit 2 ;;
esac

# Resolve repo root (this script lives in scripts/).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

echo ">> rsync $HERE  ->  $SSH_HOST:$REMOTE_DIR"
rsync -az --delete \
    --exclude '.git/' \
    --exclude '.venv/' --exclude 'venv/' \
    --exclude '__pycache__/' \
    --exclude 'runs/' \
    --exclude '*.pt' --exclude '*.npz' --exclude '*.npy' \
    --exclude '*.pdf' \
    --exclude '.DS_Store' \
    ./ "$SSH_HOST:$REMOTE_DIR/"

echo ">> submitting $SBATCH on $SSH_HOST (MODEL=${MODEL:-<env default>})"
ssh "$SSH_HOST" \
    "cd $REMOTE_DIR && mkdir -p logs && MODEL='${MODEL:-}' ROLEVEC_ENV='${ROLEVEC_ENV:-rolevec}' sbatch $SBATCH"

echo ">> queued. Tail logs with:  ssh $SSH_HOST 'tail -f $REMOTE_DIR/logs/${JOB}*'"
