from __future__ import annotations

import pytest

from statomix.core.errors import VersionSelectionError
from statomix.core.version_selection import resolve_artifact_version


def test_requested_existing_version_is_honored() -> None:
    selection = resolve_artifact_version(
        tracking={
            "latest_version": 3,
            "version_history": [1, 2, 3],
        },
        requested_version=2,
        create_new=False,
    )

    assert selection.version == 2
    assert selection.latest_version == 3


def test_missing_version_is_not_silently_replaced_by_latest() -> None:
    with pytest.raises(VersionSelectionError, match="does not exist"):
        resolve_artifact_version(
            tracking={
                "latest_version": 2,
                "version_history": [1, 2],
            },
            requested_version=4,
            create_new=False,
        )


def test_create_new_requires_exact_next_version_when_requested() -> None:
    selection = resolve_artifact_version(
        tracking={
            "latest_version": 2,
            "version_history": [1, 2],
        },
        requested_version=3,
        create_new=True,
    )
    assert selection.version == 3
    assert selection.version_history == (1, 2, 3)

    with pytest.raises(VersionSelectionError, match="next version"):
        resolve_artifact_version(
            tracking={
                "latest_version": 2,
                "version_history": [1, 2],
            },
            requested_version=7,
            create_new=True,
        )
