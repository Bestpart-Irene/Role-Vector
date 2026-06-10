"""Model-agnostic activation backends.

A backend's only job: given (role prompt, question, answer), return per-layer hidden-state vectors
(one vector per layer, mean-pooled over the ANSWER tokens). Everything downstream (role-minus-default,
score weighting, metrics, steering) is identical regardless of backend.

- `DummyBackend` runs with no model (synthetic activations) for wiring/tests.
- `TransformerLensBackend` / `NNSightBackend` are REAL implementations behind lazy imports, so importing
  this module never requires torch. nnsight can run remotely on NDIF (free [redacted] access).
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod

import numpy as np

from .config import Config


def _chat_prefix(tokenizer, role_prompt: str, question: str) -> str:
    """Render (system=role, user=question) with the model's chat template, ready for the answer."""
    msgs = [{"role": "system", "content": role_prompt}, {"role": "user", "content": question}]
    try:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"{role_prompt}\n\nUser: {question}\nAssistant: "


def _to_numpy(x) -> np.ndarray:
    """Resolve an nnsight proxy / torch tensor to a 1-D float32 numpy array."""
    x = getattr(x, "value", x)              # nnsight <0.3 used .value
    if hasattr(x, "detach"):
        x = x.detach().to("cpu").float().numpy()
    return np.asarray(x, dtype=np.float32).reshape(-1)


class ActivationBackend(ABC):
    """Interface every extraction backend implements."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    @abstractmethod
    def hidden_states(self, role_prompt: str, question: str, answer: str) -> dict[int, np.ndarray]:
        """Return {layer: vector} for one (role prompt, question, answer) triple.
        Each value is a 1-D float array of size hidden_dim, mean-pooled over the answer tokens."""

    @property
    @abstractmethod
    def hidden_dim(self) -> int: ...

    def generate(self, role_prompt: str, question: str) -> str:
        """Produce the role's answer to a question. Real backends override; the default returns a
        deterministic placeholder so the dummy pipeline is self-contained."""
        return f"[answer:{abs(hash((role_prompt, question))) % 10_000}]"

    def generate_batch(self, role_prompt: str, questions: list[str]) -> list[str]:
        """Answer many questions at once. Default loops `generate`; real backends override with a
        true batched generation (the big speedup — autoregressive generation is the bottleneck)."""
        return [self.generate(role_prompt, q) for q in questions]

    def hidden_states_batch(self, role_prompt: str, qa_pairs: list[tuple[str, str]]
                            ) -> list[dict[int, np.ndarray]]:
        """Per-(question,answer) layer vectors for many pairs. Default loops; real backends may batch."""
        return [self.hidden_states(role_prompt, q, a) for q, a in qa_pairs]

    def generate_steered(self, role_prompt: str, question: str, vector, layer: int, coeff: float) -> str:
        """Future Work #5: generate while adding `coeff * vector` to `layer`'s residual stream.
        Default falls back to unsteered generation (dummy)."""
        return self.generate(role_prompt, question)


class DummyBackend(ActivationBackend):
    """Deterministic synthetic activations — for wiring/tests, NOT for real results."""

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self._dim = cfg.hidden_dim
        self._rng = np.random.default_rng(cfg.seed)
        self._role_centers: dict[str, np.ndarray] = {}

    @property
    def hidden_dim(self) -> int:
        return self._dim

    def _center(self, key: str) -> np.ndarray:
        if key not in self._role_centers:
            h = abs(hash(key)) % (2**32)
            self._role_centers[key] = np.random.default_rng(h).standard_normal(self._dim)
        return self._role_centers[key]

    def hidden_states(self, role_prompt: str, question: str, answer: str) -> dict[int, np.ndarray]:
        # Stable per-(role,layer) signal + noise. The baseline center is de-emphasized so that, after
        # role-minus-default subtraction, the role-specific component dominates and roles come out
        # separable (matching the deck). Purely synthetic — question/answer ignored on purpose.
        scale = 0.1 if role_prompt == self.cfg.baseline_prompt else 1.0
        out = {}
        for layer in self.cfg.layers:
            center = self._center(f"{role_prompt}|{layer}") * scale
            out[layer] = center + self._rng.standard_normal(self._dim) * 0.35
        return out


class TransformerLensBackend(ActivationBackend):
    """Local white-box HF models (Llama/Qwen/Gemma) via TransformerLens `run_with_cache`."""

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        from transformer_lens import HookedTransformer  # lazy
        self.model = HookedTransformer.from_pretrained(cfg.require_model())
        self.tokenizer = self.model.tokenizer
        self._dim = self.model.cfg.d_model

    @property
    def hidden_dim(self) -> int:
        return self._dim

    def hidden_states(self, role_prompt: str, question: str, answer: str) -> dict[int, np.ndarray]:
        prefix = _chat_prefix(self.tokenizer, role_prompt, question)
        prefix_len = self.model.to_tokens(prefix).shape[1]
        tokens = self.model.to_tokens(prefix + answer)
        _, cache = self.model.run_with_cache(tokens, return_type=None)
        out = {}
        for l in self.cfg.layers:
            resid = cache["resid_post", l][0]            # (seq, dim)
            out[l] = _to_numpy(resid[prefix_len:].mean(0))
        return out

    def generate(self, role_prompt: str, question: str, max_new_tokens: int | None = None) -> str:
        # Sample (not greedy) so repeated extractions vary -> Q3 stability is a real test, not 1.0.
        n = max_new_tokens or self.cfg.max_new_tokens
        prefix = _chat_prefix(self.tokenizer, role_prompt, question)
        text = self.model.generate(prefix, max_new_tokens=n, verbose=False,
                                   do_sample=True, temperature=1.0, top_p=0.95)
        return text[len(prefix):].strip()


class NNSightBackend(ActivationBackend):
    """nnsight backend — extraction + steering injection, LOCAL or REMOTE on NDIF.

    Env:
      ROLEVEC_NDIF_REMOTE=1   run on NDIF (no local GPU; free [redacted] access)
      NDIF_API_KEY=...        your NDIF key (also settable via nnsight.CONFIG.set_default_api_key)
    """

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        from nnsight import LanguageModel  # lazy
        self.remote = os.environ.get("ROLEVEC_NDIF_REMOTE", "0") == "1"
        key = os.environ.get("NDIF_API_KEY")
        if key:
            try:
                import nnsight
                nnsight.CONFIG.set_default_api_key(key)
            except Exception:
                pass
        # device_map only matters for local execution
        self.model = LanguageModel(cfg.require_model(),
                                   device_map=None if self.remote else "auto")
        self.tokenizer = self.model.tokenizer
        self._dim = self.model.config.hidden_size

    @property
    def hidden_dim(self) -> int:
        return self._dim

    def _answer_slice(self, role_prompt: str, question: str):
        prefix = _chat_prefix(self.tokenizer, role_prompt, question)
        prefix_len = len(self.tokenizer(prefix)["input_ids"])
        return prefix, prefix_len

    def hidden_states(self, role_prompt: str, question: str, answer: str) -> dict[int, np.ndarray]:
        prefix, prefix_len = self._answer_slice(role_prompt, question)
        saved = {}
        with self.model.trace(prefix + answer, remote=self.remote):
            for l in self.cfg.layers:
                hs = self.model.model.layers[l].output[0]      # (1, seq, dim)
                saved[l] = hs[0, prefix_len:, :].mean(dim=0).save()
        return {l: _to_numpy(saved[l]) for l in self.cfg.layers}

    def generate(self, role_prompt: str, question: str, max_new_tokens: int | None = None) -> str:
        n = max_new_tokens or self.cfg.max_new_tokens
        prefix = _chat_prefix(self.tokenizer, role_prompt, question)
        in_len = len(self.tokenizer(prefix)["input_ids"])
        with self.model.generate(prefix, max_new_tokens=n, remote=self.remote,
                                 do_sample=True, temperature=1.0, top_p=0.95):
            out = self.model.generator.output.save()
        ids = _to_numpy(out[0]).astype(int).tolist()
        return self.tokenizer.decode(ids[in_len:], skip_special_tokens=True).strip()

    def generate_steered(self, role_prompt: str, question: str, vector, layer: int,
                         coeff: float, max_new_tokens: int | None = None) -> str:
        import torch
        n = max_new_tokens or self.cfg.max_new_tokens
        prefix = _chat_prefix(self.tokenizer, role_prompt, question)
        in_len = len(self.tokenizer(prefix)["input_ids"])
        vec = torch.as_tensor(np.asarray(vector, dtype=np.float32))
        with self.model.generate(prefix, max_new_tokens=n, remote=self.remote):
            # apply the steering vector to this layer's output on every generated forward pass
            with self.model.model.layers[layer].all():
                self.model.model.layers[layer].output[0][:] += coeff * vec.to(
                    self.model.model.layers[layer].output[0].device)
            out = self.model.generator.output.save()
        ids = _to_numpy(out[0]).astype(int).tolist()
        return self.tokenizer.decode(ids[in_len:], skip_special_tokens=True).strip()


class HFBackend(ActivationBackend):
    """transformers backend — BATCHED generation + `output_hidden_states` extraction. Model-agnostic
    (any HF causal LM, incl. Instruct models), runs on local/cluster GPU. This is the scalable backend:
    it batches the autoregressive generation (the bottleneck) and reads layer activations in one forward
    pass — no TransformerLens model support needed, no manual hooks.
    """

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_id = cfg.require_model()
        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype="auto", device_map="auto")
        self.model.eval()
        self._dim = self.model.config.hidden_size
        self.batch_size = 16

    @property
    def hidden_dim(self) -> int:
        return self._dim

    def _device(self):
        return next(self.model.parameters()).device

    def generate_batch(self, role_prompt: str, questions: list[str]) -> list[str]:
        torch = self._torch
        prefixes = [_chat_prefix(self.tok, role_prompt, q) for q in questions]
        out: list[str] = []
        for i in range(0, len(prefixes), self.batch_size):
            chunk = prefixes[i:i + self.batch_size]
            enc = self.tok(chunk, return_tensors="pt", padding=True).to(self._device())
            with torch.no_grad():
                gen = self.model.generate(
                    **enc, max_new_tokens=self.cfg.max_new_tokens, do_sample=True,
                    temperature=1.0, top_p=0.95, pad_token_id=self.tok.pad_token_id)
            new = gen[:, enc["input_ids"].shape[1]:]            # only the freshly generated tokens
            out.extend(s.strip() for s in self.tok.batch_decode(new, skip_special_tokens=True))
        return out

    def generate(self, role_prompt: str, question: str) -> str:
        return self.generate_batch(role_prompt, [question])[0]

    def hidden_states(self, role_prompt: str, question: str, answer: str) -> dict[int, np.ndarray]:
        return self.hidden_states_batch(role_prompt, [(question, answer)])[0]

    def hidden_states_batch(self, role_prompt, qa_pairs):
        torch = self._torch
        results: list = [None] * len(qa_pairs)
        for i in range(0, len(qa_pairs), self.batch_size):
            chunk = qa_pairs[i:i + self.batch_size]
            prefixes = [_chat_prefix(self.tok, role_prompt, q) for q, _ in chunk]
            fulls = [p + a for p, (_, a) in zip(prefixes, chunk)]
            ans_lens = [max(1, len(self.tok(f)["input_ids"]) - len(self.tok(p)["input_ids"]))
                        for p, f in zip(prefixes, fulls)]
            enc = self.tok(fulls, return_tensors="pt", padding=True).to(self._device())
            with torch.no_grad():
                hs = self.model(**enc, output_hidden_states=True).hidden_states  # (L+1) x (B,S,dim)
            for j, alen in enumerate(ans_lens):
                # left-padded -> the real answer tokens are the LAST `alen` positions
                results[i + j] = {l: _to_numpy(hs[l][j, -alen:, :].mean(dim=0)) for l in self.cfg.layers}
        return results


_BACKENDS = {
    "dummy": DummyBackend,
    "hf": HFBackend,
    "transformer_lens": TransformerLensBackend,
    "nnsight": NNSightBackend,
}


def get_backend(cfg: Config) -> ActivationBackend:
    if cfg.backend not in _BACKENDS:
        raise ValueError(f"unknown backend {cfg.backend!r}; choose from {sorted(_BACKENDS)}")
    return _BACKENDS[cfg.backend](cfg)
