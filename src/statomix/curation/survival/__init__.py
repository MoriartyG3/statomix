"""Survival-column inference, pairing, and event-label curation."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "SurvCatMetaEditSchema": (
        "statomix.curation.survival.report",
        "SurvCatMetaEditSchema",
    ),
    "SurvEditSchema": ("statomix.curation.survival.report", "SurvEditSchema"),
    "SurvMetaReport": ("statomix.curation.survival.report", "SurvMetaReport"),
    "SurvPair": ("statomix.curation.survival.report", "SurvPair"),
    "SurvPairs": ("statomix.curation.survival.report", "SurvPairs"),
    "SurvivalDataTypes": (
        "statomix.curation.survival.profiler",
        "SurvivalDataTypes",
    ),
    "SurvivalSemanticProfile": (
        "statomix.curation.survival.profiler",
        "SurvivalSemanticProfile",
    ),
    "get_survival_semantic_col_profile": (
        "statomix.curation.survival.profiler",
        "get_survival_semantic_col_profile",
    ),
    "get_survival_sematic_col_profile": (
        "statomix.curation.survival.profiler",
        "get_survival_sematic_col_profile",
    ),
    "normalize_survival_event_columns": (
        "statomix.curation.survival.events",
        "normalize_survival_event_columns",
    ),
    "parse_optional_event_observed": (
        "statomix.curation.survival.events",
        "parse_optional_event_observed",
    ),
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
    "SurvCatMetaEditSchema",
    "SurvEditSchema",
    "SurvMetaReport",
    "SurvPair",
    "SurvPairs",
    "SurvivalDataTypes",
    "SurvivalSemanticProfile",
    "get_survival_semantic_col_profile",
    "get_survival_sematic_col_profile",
    "normalize_survival_event_columns",
    "parse_optional_event_observed",
)
