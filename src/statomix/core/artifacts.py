"""Portable, content-pinned references to data produced by a pipeline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def safe_relative(root: Path, relative: str) -> Path:
    value = PurePosixPath(relative)
    if value.is_absolute() or ".." in value.parts or not value.parts:
        raise ValueError(f"Unsafe artifact path: {relative!r}")
    candidate = (root / value).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError(f"Artifact escapes project root: {relative!r}")
    return candidate


@dataclass(frozen=True, slots=True, kw_only=True)
class DatasetArtifactRef:
    """Immutable manifest snapshot; changing project_root only relocates it.

    Manifests embed parent snapshots so legacy Cleaner files need not be edited.
    Hashes identify file bytes, not mathematical equivalence across serializers.
    """

    project_root: Path
    manifest_json: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "project_root", Path(self.project_root).resolve())
        data = json.loads(self.manifest_json)
        if data.get("schema_version") != 1 or data.get("status") != "completed":
            raise ValueError("Unsupported or incomplete dataset artifact.")
        required = {
            "project",
            "dataset",
            "pipeline",
            "version",
            "config_version",
            "files",
        }
        if not required.issubset(data):
            raise ValueError("Incomplete artifact identity.")
        for field in ("version", "config_version"):
            if type(data[field]) is not int or data[field] < 1:
                raise ValueError(f"Invalid {field}.")
        if data["pipeline"] not in {"cleaner", "reference", "transformer"}:
            raise ValueError("Unknown artifact producer.")
        if not {"df", "surv_pairs", "col_profiles"}.issubset(data["files"]):
            raise ValueError("Artifact is missing required files.")
        for record in data["files"].values():
            safe_relative(self.project_root, record["path"])
            checksum = record["sha256"]
            if len(checksum) != 64 or any(
                c not in "0123456789abcdef" for c in checksum
            ):
                raise ValueError("Invalid artifact checksum.")
        object.__setattr__(self, "manifest_json", canonical_json(data))

    @property
    def manifest(self) -> dict:
        return json.loads(self.manifest_json)

    @property
    def artifact_id(self) -> str:
        return digest_json(self.manifest)

    def path(self, name: str) -> Path:
        return safe_relative(self.project_root, self.manifest["files"][name]["path"])

    def to_dict(self) -> dict:
        return {"artifact_id": self.artifact_id, "manifest": self.manifest}

    @classmethod
    def from_dict(cls, *, project_root: Path, data: dict) -> DatasetArtifactRef:
        reference = cls(
            project_root=project_root, manifest_json=canonical_json(data["manifest"])
        )
        if reference.artifact_id != data["artifact_id"]:
            raise ValueError("Artifact manifest checksum mismatch.")
        return reference
