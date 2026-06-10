#!/bin/bash
# One-time: create the `rolevec` conda env on the cluster with the TransformerLens backend deps.
# Idempotent — re-running just updates pip deps. Run via:
#   ssh cluster 'bash -s' < scripts/setup_cluster_env.sh
set -euo pipefail

CONDA_SH="/path/to/miniconda3/etc/profile.d/conda.sh"
ENV_NAME="${ROLEVEC_ENV_NAME:-rolevec}"

source "$CONDA_SH"
if ! conda env list | grep -qE "/${ENV_NAME}\$|^${ENV_NAME}\s"; then
    echo "Creating conda env: $ENV_NAME (python 3.11)"
    conda create -y -n "$ENV_NAME" python=3.11
fi
conda activate "$ENV_NAME"

echo "Installing rolevec + transformer_lens backend deps into $ENV_NAME ..."
pip install --upgrade pip
pip install "transformer-lens>=2.0" "torch>=2.2" "transformers>=4.44" "anthropic>=0.40" numpy pyyaml

echo "Done. Env ready: $(conda info --envs | grep "$ENV_NAME")"
echo "Use with: conda activate $ENV_NAME"
