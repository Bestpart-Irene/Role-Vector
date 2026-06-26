# Cluster scripts (SLURM GPU cluster)

Self-contained SLURM scripts for running the **`transformer_lens`** extraction backend
on a GPU node. The `dummy` and `nnsight` (NDIF remote) backends do **not** need a cluster job —
run those locally; see `../docs/ndif-setup.md`.

## What's here
| File | Purpose |
|---|---|
| `setup_cluster_env.sh` | One-time: create the `rolevec` conda env with TransformerLens + torch deps. |
| `transfer_launch.sh`   | rsync this project to the cluster and submit a job in one step (run from your laptop). |
| `rolevec_smoke_1gpu.sbatch` | Sanity job: `--runs 1` on one GPU; confirms the backend returns finite vectors. |
| `rolevec_extract_tl.sbatch` | Full 30-run TransformerLens extraction + validation gate. |

## Conventions (inherited)
- **8h walltime cap** on every job → `--time=08:00:00 --requeue`. Extraction is short, but keep it.
- **All caches/outputs go to `/scratch/$USER`**, never `$HOME`. HF cache → `/scratch/$USER/hf_cache`.
- **Model is deferred** — scripts read `$ROLEVEC_MODEL` / pass `--model`; no model id is hard-coded
  in the library. Override `MODEL=...` on submit.
- **Env is overridable** — `ROLEVEC_ENV=/path/to/env sbatch ...` to point at a different conda env.
- Extraction = forward passes (not training), so a bare `--gres=gpu:1` is fine; no H200 requirement.

## Quick start (from your laptop)
```bash
# 1) one-time: build the env on the cluster
ssh <cluster> 'bash -s' < scripts/setup_cluster_env.sh

# 2) ship code + submit a smoke job
MODEL=Qwen/Qwen2.5-1.5B bash scripts/transfer_launch.sh smoke

# 3) once the smoke passes, the full 30-run extraction
MODEL=Qwen/Qwen2.5-1.5B bash scripts/transfer_launch.sh extract
```

> Cluster connection lives in your local `~/.ssh/config` (define a host alias with your own
> HostName / User). The private key stays in `~/.ssh` and is never copied here. Override the
> alias at submit time with `SSH_HOST=<your-host> bash scripts/transfer_launch.sh ...`.
