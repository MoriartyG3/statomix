"""Pure helpers for versioned non-Zarr artifacts such as config workbooks."""

from __future__ import annotations

from dataclasses import dataclass

from statomix.core.errors import VersionSelectionError


@dataclass(frozen=True, slots=True, kw_only=True)
class ArtifactVersionSelection:
    """Resolved artifact version and updated tracking metadata."""

    version: int
    latest_version: int
    version_history: tuple[int, ...]

    def as_tracking_dict(self, *, name: str | None) -> dict[str, object]:
        return {
            "latest_version": self.latest_version,
            "version_history": list(self.version_history),
            "name": name,
        }


def resolve_artifact_version(
    *,
    tracking: dict[str, object] | None,
    requested_version: int | None,
    create_new: bool,
) -> ArtifactVersionSelection:
    """Resolve a workbook-like version without silently ignoring requests."""

    if tracking is None:
        latest = 1
        history = [1]
    else:
        latest = int(tracking.get("latest_version", 1))
        history = sorted({int(value) for value in tracking.get("version_history", [1])})
        if latest not in history:
            history.append(latest)
            history.sort()

    if create_new:
        next_version = latest + 1
        if requested_version is not None and int(requested_version) != next_version:
            raise VersionSelectionError(
                "When create_new=True, analysis_config_version must be the next "
                f"version ({next_version}), got {requested_version}."
            )
        history.append(next_version)
        return ArtifactVersionSelection(
            version=next_version,
            latest_version=next_version,
            version_history=tuple(sorted(set(history))),
        )

    resolved = latest if requested_version is None else int(requested_version)
    if resolved not in history:
        raise VersionSelectionError(
            f"Analysis configuration version {resolved} does not exist; "
            f"available versions are {history}. Use create_new=True to create "
            "the next version."
        )
    return ArtifactVersionSelection(
        version=resolved,
        latest_version=latest,
        version_history=tuple(history),
    )
