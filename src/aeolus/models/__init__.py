"""Model architectures (Scope v2.1 §3).

All builders return ``(module, spec)`` and require the optional torch extra.
"""

from .base import DEFAULT_LEADS, ModelSpec, require_torch, torch_available

__all__ = [
    "DEFAULT_LEADS",
    "ModelSpec",
    "require_torch",
    "torch_available",
    "build_lstm",
    "build_cnn",
    "build_transformer",
    "build_gnn",
    "build_pinn",
    "build_diffusion",
    "build_fusion",
]


def __getattr__(name: str):
    """Lazy builder import so importing the package never requires torch."""
    modules = {
        "build_lstm": "lstm",
        "build_cnn": "cnn",
        "build_transformer": "transformer",
        "build_gnn": "gnn",
        "build_pinn": "pinn",
        "build_diffusion": "diffusion",
        "build_fusion": "fusion",
    }
    if name in modules:
        import importlib

        return getattr(importlib.import_module(f".{modules[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
