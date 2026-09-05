"""Persistent dataset-purpose declarations."""

from __future__ import annotations

DATASET_ROLES = ("analysis", "reference")
DEFAULT_DATASET_ROLE = "analysis"


def normalize_dataset_role(dataset_role: str) -> str:
    """Return a supported, normalized dataset role."""

    if not isinstance(dataset_role, str):
        raise TypeError("dataset_role must be a string")

    normalized = dataset_role.strip().casefold()

    if normalized not in DATASET_ROLES:
        raise ValueError(
            f"dataset_role must be one of {DATASET_ROLES!r}; "
            f"received {dataset_role!r}."
        )

    return normalized
