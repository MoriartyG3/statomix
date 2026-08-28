"""Human-in-the-loop datatype curation services and contracts."""

from __future__ import annotations

from importlib import import_module
from typing import Any


def __getattr__(name: str) -> Any:
    exports = {
        "apply_curation_schemas": "statomix.curation.service",
        "apply_inherited_category_edits": "statomix.curation.inheritance",
        "build_inherited_curated_state": "statomix.curation.inheritance",
        "InheritedCuratedState": "statomix.curation.inheritance",
    }
    if name in exports:
        value = getattr(import_module(exports[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = (
    "InheritedCuratedState",
    "apply_curation_schemas",
    "apply_inherited_category_edits",
    "build_inherited_curated_state",
)
