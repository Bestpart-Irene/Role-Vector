# Going live: NDIF + nnsight (and TransformerLens / Claude judge)

How to turn the synthetic dummy run into real model results. The pipeline code does not change —
you pick a backend, install its deps, set a couple of env vars, and run the same commands.

## Recommended path — nnsight on NDIF (free, no local GPU)
NDIF (National Deep Inference Fabric, NSF-funded) runs nnsight requests remotely on shared GPUs,
up to Llama-3.1-405B. Eligible academic researchers can apply for free access.

1. **Get access & key:** register at https://ndif.us/ → get an NDIF API key.
2. **Install:** `pip install nnsight transformers` (no local `torch` GPU build needed for remote).
3. **Env:**
   ```bash
   export ROLEVEC_NDIF_REMOTE=1
   export NDIF_API_KEY=<your key>
   export ANTHROPIC_API_KEY=<key>          # for the real judge
   ```
4. **Run:**
   ```bash
   PYTHONPATH=src python -m rolevec.run_all \
       --backend nnsight --model meta-llama/Meta-Llama-3.1-405B --runs 30
   ```
The `NNSightBackend` (`src/rolevec/backends.py`) implements:
- `hidden_states` — traces `prefix + answer`, mean-pools each layer's residual over the answer tokens.
- `generate` / `generate_steered` — the latter adds `coeff·v_r` to a layer's output across all
  generated forwards (steering injection, Future Work #5).

## Local alternative — TransformerLens (your own GPU)
```bash
pip install transformer-lens torch transformers
PYTHONPATH=src python -m rolevec.pipeline --backend transformer_lens --model Qwen/Qwen2.5-1.5B --runs 30
```
`TransformerLensBackend` uses `run_with_cache` and pools `resid_post` over the answer tokens.

## Cluster route — SLURM GPU cluster (TransformerLens)
Configure an SSH host alias for your cluster in `~/.ssh/config` (no key is copied into this repo).
Set your own scratch/home paths, conda base, and partition in `scripts/run_on_cluster.sbatch`.
```bash
ssh <cluster>
git clone <this repo> ~/Role-Vector
cd ~/Role-Vector
# put HF_TOKEN + ANTHROPIC_API_KEY in ~/.bashrc (never in git)
sbatch scripts/run_on_cluster.sbatch          # creates the env on first run, then runs the roadmap
```
NDIF (remote) and the cluster (local GPU) are **independent routes** — NDIF needs no cluster; the
cluster needs no NDIF key. Pick one.

## The judge (real role-adherence scoring) — **FREE by default**
The judge is **separate from the extraction model** (no self-grading) and is pluggable:

- **`local` (default, FREE):** `LocalJudge` runs an open-weight instruct model via HF transformers —
  no API key, no per-token cost. Runs on your cluster GPU.
  ```bash
  pip install transformers torch
  export ROLEVEC_JUDGE_BACKEND=local
  export ROLEVEC_JUDGE_MODEL=Qwen/Qwen2.5-7B-Instruct   # any HF instruct model
  ```
- **`anthropic` (paid):** `LLMJudge` calls the Claude API.
  ```bash
  pip install anthropic
  export ROLEVEC_JUDGE_BACKEND=anthropic ANTHROPIC_API_KEY=<key> ROLEVEC_JUDGE_MODEL=claude-haiku-4-5
  ```

With any real judge, the Q1 domain-sensitivity checks become *required* in the validation gate
(under the dummy heuristic judge they are reported-only).

## Sanity checklist before a full 30-run job
1. `--runs 1` on one role to confirm the backend returns finite vectors of the right `hidden_dim`.
2. Eyeball a few `generate` outputs — are answers on-topic and in-role?
3. Spot-check judge scores against the rubric on ~10 answers.
4. Confirm `meta.json` hidden_dim matches the model, then scale to 30 runs.

## Cost / throughput notes
- Extraction is the cheap part (forward passes); the **judge** is the volume cost
  (≈ roles × 90 questions × runs calls). Use Haiku and cache where possible.
- For 30 repeated extractions, NDIF remote batching or vllm-lens is the throughput path.
