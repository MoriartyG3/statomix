"""Human-in-the-loop datatype curation services and contracts."""

from __future__ import annotations

from importlib import import_module
from typing import Any


def __getattr__(name: str) -> Any:
    if name == "apply_curation_schemas":
        value = getattr(import_module("statomix.curation.service"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ("apply_curation_schemas",)
