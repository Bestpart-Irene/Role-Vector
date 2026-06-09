"""Model-agnostic activation backends.

A backend's only job: given prompts, return per-layer hidden-state vectors (one vector per prompt
per layer, mean-pooled over tokens). Everything downstream (role-minus-default, score weighting,
metrics) is identical regardless of backend.

`DummyBackend` runs today with no model so the pipeline is end-to-end testable. The real backends
are thin stubs with the exact integration point marked — fill them in once a model is chosen.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .config import Config


class ActivationBackend(ABC):
    """Interface every extraction backend implements."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    @abstractmethod
    def hidden_states(self, prompt: str, text: str) -> dict[int, np.ndarray]:
        """Return {layer: vector} for one (role-prompt, generated/forced text) pair.

        Keys are layer indices in cfg.layers; each value is a 1-D float array of size hidden_dim,
        mean-pooled over the answer tokens.
        """

    @property
    @abstractmethod
    def hidden_dim(self) -> int: ...

    def generate(self, role_prompt: str, question: str) -> str:
        """Produce the role's answer to a question. Real backends override with model generation;
        the default returns a deterministic placeholder so the dummy pipeline is self-contained."""
        return f"[answer:{abs(hash((role_prompt, question))) % 10_000}]"

    def generate_steered(self, prompt: str, question: str, vector, layer: int, coeff: float) -> str:
        """Future Work #5: generate while adding `coeff * vector` to `layer`'s residual stream.
        Real backends override (nnsight intervention). Default falls back to unsteered generation."""
        return self.generate(prompt, question)


class DummyBackend(ActivationBackend):
    """Deterministic random activations — for wiring/tests, NOT for real results.

    Encodes a faint, reproducible per-(role,layer) signal plus noise so that downstream metrics
    produce sane, non-degenerate numbers (same-role > cross-role, separable roles).
    """

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

    def hidden_states(self, prompt: str, text: str) -> dict[int, np.ndarray]:
        # Stable role/layer signal + small noise -> realistic stability & separability.
        # The baseline (default-assistant) center is de-emphasized so that, after role-minus-default
        # subtraction, the role-specific component dominates and roles come out genuinely separable
        # (matching the deck's finding that all pairs separate). Purely synthetic — not real signal.
        is_baseline = (prompt == self.cfg.baseline_prompt)
        scale = 0.1 if is_baseline else 1.0
        out = {}
        for layer in self.cfg.layers:
            center = self._center(f"{prompt}|{layer}") * scale
            noise = self._rng.standard_normal(self._dim) * 0.35
            out[layer] = center + noise
        return out


class TransformerLensBackend(ActivationBackend):
    """White-box HF models (Llama/Qwen/Gemma) via TransformerLens `run_with_cache`.

    INTEGRATION POINT (fill once model chosen):
        from transformer_lens import HookedTransformer
        self.model = HookedTransformer.from_pretrained(cfg.require_model())
        _, cache = self.model.run_with_cache(prompt + text)
        # mean-pool resid_post over answer tokens for each layer in cfg.layers
    """

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        cfg.require_model()
        raise NotImplementedError(
            "TransformerLensBackend is a stub. Install transformer-lens, set --model, "
            "and implement hidden_states() at the marked integration point."
        )

    @property
    def hidden_dim(self) -> int:  # pragma: no cover - stub
        raise NotImplementedError

    def hidden_states(self, prompt: str, text: str):  # pragma: no cover - stub
        raise NotImplementedError


class NNSightBackend(ActivationBackend):
    """nnsight backend — supports both extraction and (later) steering injection.

    Can run LOCALLY or REMOTELY on NDIF (National Deep Inference Fabric, [redacted]/[redacted]):
    free remote access to large open-weight models incl. Llama-3.1-405B, no local GPU. Pass
    remote=True to LanguageModel(...).trace(...) and set your NDIF API key.

    INTEGRATION POINT:
        from nnsight import LanguageModel
        self.model = LanguageModel(cfg.require_model())
        with self.model.trace(prompt + text, remote=True):   # remote=True -> runs on NDIF
            acts = {l: self.model.model.layers[l].output[0].mean(dim=1).save() for l in cfg.layers}
    """

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        cfg.require_model()
        raise NotImplementedError(
            "NNSightBackend is a stub. Install nnsight, set --model, "
            "and implement hidden_states() at the marked integration point."
        )

    @property
    def hidden_dim(self) -> int:  # pragma: no cover - stub
        raise NotImplementedError

    def hidden_states(self, prompt: str, text: str):  # pragma: no cover - stub
        raise NotImplementedError


_BACKENDS = {
    "dummy": DummyBackend,
    "transformer_lens": TransformerLensBackend,
    "nnsight": NNSightBackend,
}


def get_backend(cfg: Config) -> ActivationBackend:
    try:
        return _BACKENDS[cfg.backend](cfg)
    except KeyError:
        raise ValueError(f"unknown backend {cfg.backend!r}; choose from {sorted(_BACKENDS)}")
