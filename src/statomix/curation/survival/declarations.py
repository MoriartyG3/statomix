"""Survival endpoint declarations and current processing capabilities."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any

EVENT_STRUCTURES = (
    "single_event",
    "competing_risks",
    "recurrent_events",
    "multi_state",
    "semi_competing_risks",
)

OBSERVATION_SCHEMES = (
    "exact",
    "right_censored",
    "left_censored",
    "interval_censored",
    "mixed",
)

DEFAULT_EVENT_STRUCTURE = "single_event"
DEFAULT_OBSERVATION_SCHEME = "right_censored"

DECLARATION_FIELDS = (
    "event_structure",
    "observation_scheme",
)


def has_declaration_fields(columns: Collection[str]) -> bool:
    """Allow legacy artifacts, but reject partially specified schemas."""

    present = set(DECLARATION_FIELDS).intersection(columns)

    if present and present != set(DECLARATION_FIELDS):
        missing = sorted(set(DECLARATION_FIELDS) - present)
        raise ValueError(f"Survival declaration is missing fields: {missing!r}.")

    return bool(present)


def _validated_choice(
    value: Any,
    *,
    field_name: str,
    choices: tuple[str, ...],
) -> str:
    if not isinstance(value, str):
        raise ValueError(
            f"{field_name} must contain one of {choices!r}; " f"received {value!r}."
        )

    value = value.strip()

    if value not in choices:
        raise ValueError(
            f"Invalid {field_name}: {value!r}. " f"Choose one of {choices!r}."
        )

    return value


def read_survival_declaration(
    data: Mapping[str, Any],
) -> tuple[str, str]:
    """Read explicit declarations or default a legacy artifact."""

    if not has_declaration_fields(data.keys()):
        return DEFAULT_EVENT_STRUCTURE, DEFAULT_OBSERVATION_SCHEME

    event_structure = _validated_choice(
        data["event_structure"],
        field_name="event_structure",
        choices=EVENT_STRUCTURES,
    )
    observation_scheme = _validated_choice(
        data["observation_scheme"],
        field_name="observation_scheme",
        choices=OBSERVATION_SCHEMES,
    )

    return event_structure, observation_scheme


def require_supported_survival(
    *,
    surv_label: str,
    event_structure: str,
    observation_scheme: str,
    operation: str,
) -> None:
    """Reject declared types without an implemented processing path."""

    supported = (
        event_structure == DEFAULT_EVENT_STRUCTURE
        and observation_scheme == DEFAULT_OBSERVATION_SCHEME
    )

    if not supported:
        raise NotImplementedError(
            f"{operation}: survival endpoint {surv_label!r} declares "
            f"event_structure={event_structure!r} and "
            f"observation_scheme={observation_scheme!r}. "
            "This combination is declared but not implemented. "
            "Currently only single_event + right_censored can be "
            "processed. Keep the declaration; do not recode this "
            "endpoint as Boolean merely to bypass this restriction."
        )
