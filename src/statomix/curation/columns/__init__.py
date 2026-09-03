"""Column profiling, semantic inference, and edit-schema contracts."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "ColumnAudit": (
        "statomix.curation.columns.audit",
        "ColumnAudit",
    ),
    "ColumnAuditProfile": (
        "statomix.curation.columns.audit",
        "ColumnAuditProfile",
    ),
    "ColumnValueFrequency": (
        "statomix.curation.columns.audit",
        "ColumnValueFrequency",
    ),
    "ColEdit": ("statomix.curation.columns.report", "ColEdit"),
    "ColEditSchema": ("statomix.curation.columns.report", "ColEditSchema"),
    "ColProfile": ("statomix.curation.columns.profiler", "ColProfile"),
    "ColProfiler": ("statomix.curation.columns.profiler", "ColProfiler"),
    "ColReport": ("statomix.curation.columns.report", "ColReport"),
    "DataTypes": ("statomix.curation.columns.semantic_rules", "DataTypes"),
    "DatatypeInventory": (
        "statomix.curation.columns.inventory",
        "DatatypeInventory",
    ),
    "RawColProfile": ("statomix.curation.columns.profiler", "RawColProfile"),
    "SemanticProfile": ("statomix.curation.columns.profiler", "SemanticProfile"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from error
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


__all__ = (
    "ColumnAudit",
    "ColumnAuditProfile",
    "ColumnValueFrequency",
    "ColEdit",
    "ColEditSchema",
    "ColProfile",
    "ColProfiler",
    "ColReport",
    "DataTypes",
    "DatatypeInventory",
    "RawColProfile",
    "SemanticProfile",
)
