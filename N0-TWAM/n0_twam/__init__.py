# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""N0-TWAM package with lazy top-level imports.

Keeping the package root lightweight lets contracts and codecs run in data
conversion and validation processes without importing Torch, Diffusers, or the
training configuration stack. Existing ``n0_twam.models``-style attribute access
continues to work through :func:`__getattr__`.
"""

from importlib import import_module
from types import ModuleType

__version__ = "0.1.0"

__all__ = ("actions", "configs", "data", "distributed", "models")


def __getattr__(name: str) -> ModuleType:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module
