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

    def generate(self, role_prompt: str, question: str, max_new_tokens: int = 200) -> str:
        # Sample (not greedy) so repeated extractions vary -> Q3 stability is a real test, not 1.0.
        prefix = _chat_prefix(self.tokenizer, role_prompt, question)
        text = self.model.generate(prefix, max_new_tokens=max_new_tokens, verbose=False,
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

    def generate(self, role_prompt: str, question: str, max_new_tokens: int = 200) -> str:
        prefix = _chat_prefix(self.tokenizer, role_prompt, question)
        in_len = len(self.tokenizer(prefix)["input_ids"])
        with self.model.generate(prefix, max_new_tokens=max_new_tokens, remote=self.remote,
                                 do_sample=True, temperature=1.0, top_p=0.95):
            out = self.model.generator.output.save()
        ids = _to_numpy(out[0]).astype(int).tolist()
        return self.tokenizer.decode(ids[in_len:], skip_special_tokens=True).strip()

    def generate_steered(self, role_prompt: str, question: str, vector, layer: int,
                         coeff: float, max_new_tokens: int = 200) -> str:
        import torch
        prefix = _chat_prefix(self.tokenizer, role_prompt, question)
        in_len = len(self.tokenizer(prefix)["input_ids"])
        vec = torch.as_tensor(np.asarray(vector, dtype=np.float32))
        with self.model.generate(prefix, max_new_tokens=max_new_tokens, remote=self.remote):
            # apply the steering vector to this layer's output on every generated forward pass
            with self.model.model.layers[layer].all():
                self.model.model.layers[layer].output[0][:] += coeff * vec.to(
                    self.model.model.layers[layer].output[0].device)
            out = self.model.generator.output.save()
        ids = _to_numpy(out[0]).astype(int).tolist()
        return self.tokenizer.decode(ids[in_len:], skip_special_tokens=True).strip()


_BACKENDS = {
    "dummy": DummyBackend,
    "transformer_lens": TransformerLensBackend,
    "nnsight": NNSightBackend,
}


def get_backend(cfg: Config) -> ActivationBackend:
    if cfg.backend not in _BACKENDS:
        raise ValueError(f"unknown backend {cfg.backend!r}; choose from {sorted(_BACKENDS)}")
    return _BACKENDS[cfg.backend](cfg)
