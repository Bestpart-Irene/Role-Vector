"""Role Vector Validation — construct-validation pipeline for LLM agent personas.

Model choice is deferred: the default backend is `dummy` (random activations) so the full
extract -> metrics pipeline runs without a GPU. Swap in a real backend via config / --backend.
"""
__version__ = "0.1.0"
