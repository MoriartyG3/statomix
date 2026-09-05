"""Portable artifact loading and staged, immutable bundle publication."""

from __future__ import annotations

import json
import os
import shutil
import sys
from contextlib import contextmanager
from importlib.metadata import version as package_version
from pathlib import Path
from tempfile import mkdtemp

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from statomix.core.artifacts import DatasetArtifactRef, canonical_json
from statomix.curation.columns import ColReport
from statomix.curation.survival.report import SurvPairs
from statomix.storage.hashing import sha256_file
from statomix.storage.parquet_metadata import (
    RANK_METADATA_KEY,
    load_category_rank_metadata,
)
from statomix.transformation.metadata import (
    ArtifactData,
    initial_metadata,
    validate_state,
)

SEMANTIC_METADATA_KEY = b"statomix.semantics"


def execution_fingerprint():
    """Invalidate execution reuse when implementation/runtime dependencies change."""
    root = Path(__file__).resolve().parents[1]
    paths = []
    for folder in (
        "transformation",
        "pipelines/transformer",
        "pipelines/reference",
        "curation/columns",
        "curation/survival",
    ):
        paths.extend((root / folder).glob("*.py"))
    paths.extend(
        [
            Path(__file__),
            root / "core/artifacts.py",
            root / "storage/parquet_metadata.py",
        ]
    )
    return {
        "python": sys.version.split()[0],
        "packages": {
            name: package_version(name)
            for name in ("statomix", "pandas", "numpy", "pyarrow", "fileverse17")
        },
        "source_sha256": {
            p.relative_to(root).as_posix(): sha256_file(path=p)
            for p in sorted(set(paths))
        },
    }


def project_root_for_dataset(dataset) -> Path:
    # Canonical layout: project/datasets/<dataset>/df/source_df.parquet.
    return Path(dataset.paths["df"]["source"]).resolve().parents[3]


def verify_artifact(reference: DatasetArtifactRef) -> None:
    for name, record in reference.manifest["files"].items():
        if sha256_file(path=reference.path(name)) != record["sha256"]:
            raise ValueError(
                f"Artifact integrity failure: {name!r} in {reference.artifact_id}."
            )
    for parent in reference.manifest.get("parents", []):
        verify_artifact(
            DatasetArtifactRef.from_dict(
                project_root=reference.project_root, data=parent
            )
        )


def cleaner_artifact(
    dataset,
    *,
    version: int,
    config_version: int,
    units=None,
    endpoint_definitions=None,
    reason: str = "",
):
    if (units or endpoint_definitions) and not reason.strip():
        raise ValueError("A reason is required for new semantic declarations.")
    bundle = dataset.cleaner._find_group_bundle(
        version=version, config_version=config_version
    )
    directory = bundle["config"]["path"] / "curated_data"
    root = project_root_for_dataset(dataset)
    paths = {
        "df": directory / "df.parquet",
        "col_profiles": directory / "col_profiles.parquet",
        "surv_pairs": directory / "surv_pairs.parquet",
    }
    files = {
        name: {
            "path": path.resolve().relative_to(root).as_posix(),
            "sha256": sha256_file(path=path),
        }
        for name, path in paths.items()
    }
    df = pd.read_parquet(paths["df"])
    profiles = ColReport.load_col_profiles(path=paths["col_profiles"])
    pairs = SurvPairs.load(paths["surv_pairs"])
    metadata = initial_metadata(
        df, profiles, pairs, units=units, endpoint_definitions=endpoint_definitions
    )
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "project": dataset.cleaner.project_name,
        "dataset": dataset.dataset_name,
        "pipeline": "cleaner",
        "version": int(version),
        "config_version": int(config_version),
        "files": files,
        "metadata": metadata,
        "parents": [],
        "declaration_reason": reason,
    }
    reference = DatasetArtifactRef(
        project_root=root, manifest_json=canonical_json(manifest)
    )
    load_artifact(reference)
    return reference


def load_artifact(reference: DatasetArtifactRef) -> ArtifactData:
    verify_artifact(reference)
    manifest = reference.manifest
    df = pd.read_parquet(reference.path("df"))
    profiles = ColReport.load_col_profiles(path=reference.path("col_profiles"))
    pairs = SurvPairs.load(reference.path("surv_pairs"))
    ranks = load_category_rank_metadata(reference.path("df"))
    if "metadata" in manifest["files"]:
        metadata = json.loads(reference.path("metadata").read_text())
    else:
        metadata = manifest["metadata"]
    # A child records its own immediate parents. Source indexes may be nonunique;
    # row ordinal is the unambiguous within-artifact identity.
    lineage = pd.DataFrame(
        {
            "output_row": range(len(df)),
            "parent_artifact": [reference.artifact_id] * len(df),
            "parent_row": range(len(df)),
            "source_dataset": [manifest["dataset"]] * len(df),
        }
    )
    state = ArtifactData(df, profiles, pairs, metadata, ranks, lineage)
    validate_state(state)
    return state


def write_state(state: ArtifactData, directory: Path) -> None:
    validate_state(state)
    table = pa.Table.from_pandas(state.df, preserve_index=True)
    footer = dict(table.schema.metadata or {})
    footer[RANK_METADATA_KEY] = canonical_json(state.ranks).encode()
    footer[SEMANTIC_METADATA_KEY] = canonical_json(state.metadata).encode()
    pq.write_table(table.replace_schema_metadata(footer), directory / "df.parquet")
    state.pairs.save(directory / "surv_pairs.parquet")
    ColReport().save_col_profiles(
        col_profiles=state.profiles, path=directory / "col_profiles.parquet"
    )
    (directory / "metadata.json").write_text(
        canonical_json(state.metadata), encoding="utf-8"
    )
    state.lineage.to_parquet(directory / "row_lineage.parquet", index=False)


@contextmanager
def artifact_lock(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ".transformer.lock"
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(
            f"A writer lock exists at {path}. Do not remove it until the writer is confirmed stopped."
        ) from exc
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(str(os.getpid()))
        yield
    finally:
        path.unlink(missing_ok=True)


def publish_artifact(
    *,
    project_root,
    destination,
    state,
    identity,
    parents,
    specification,
    audit,
    exclusions=(),
    column_updates=(),
    unused_updates=(),
    linked_files=None,
):
    """Called under the producer lock. Rename a complete staged directory once."""
    if destination.exists():
        raise FileExistsError(f"Immutable output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(mkdtemp(prefix=".transformer-stage-", dir=destination.parent))
    try:
        write_state(state, staging)
        (staging / "specification.json").write_text(
            canonical_json(specification), encoding="utf-8"
        )
        from statomix.reporting.excel.transformation import write_transformation_report

        write_transformation_report(
            path=staging / "audit.xlsx",
            audit=audit,
            exclusions=exclusions,
            column_updates=column_updates,
            unused_updates=unused_updates,
            parents=parents,
            specification=specification,
        )
        filenames = {
            "df": "df.parquet",
            "col_profiles": "col_profiles.parquet",
            "surv_pairs": "surv_pairs.parquet",
            "metadata": "metadata.json",
            "lineage": "row_lineage.parquet",
            "specification": "specification.json",
            "audit": "audit.xlsx",
        }
        if exclusions:
            pd.DataFrame(exclusions).to_parquet(
                staging / "excluded_rows.parquet",
                index=False,
            )
            filenames["exclusions"] = "excluded_rows.parquet"

        if column_updates:
            pd.DataFrame(column_updates).to_parquet(
                staging / "column_updates.parquet",
                index=False,
            )

            filenames["column_updates"] = "column_updates.parquet"

        if unused_updates:
            pd.DataFrame(unused_updates).to_parquet(
                staging / "unused_update_rows.parquet",
                index=False,
            )

            filenames["unused_updates"] = "unused_update_rows.parquet"
        files = {
            key: {
                "path": (destination / filename).relative_to(project_root).as_posix(),
                "sha256": sha256_file(path=staging / filename),
            }
            for key, filename in filenames.items()
        }
        linked_files = dict(linked_files or {})
        duplicate_file_keys = set(files).intersection(linked_files)
        if duplicate_file_keys:
            raise ValueError(
                "Linked-file names collide with generated artifact files: "
                f"{sorted(duplicate_file_keys)!r}."
            )
        for key, path in linked_files.items():
            linked_path = Path(path).resolve()
            relative_path = linked_path.relative_to(Path(project_root).resolve())
            files[key] = {
                "path": relative_path.as_posix(),
                "sha256": sha256_file(path=linked_path),
            }
        manifest = {
            "schema_version": 1,
            "status": "completed",
            **identity,
            "files": files,
            "parents": [p.to_dict() for p in parents],
            "specification": specification,
            "rows": len(state.df),
            "columns": len(state.df.columns),
            "software": {
                name: package_version(name)
                for name in ("statomix", "pandas", "numpy", "pyarrow")
            },
        }
        (staging / "manifest.json").write_text(
            canonical_json(manifest), encoding="utf-8"
        )
        # Recheck parents after computation and before publication.
        for parent in parents:
            verify_artifact(parent)
        os.rename(staging, destination)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return DatasetArtifactRef(
        project_root=project_root, manifest_json=canonical_json(manifest)
    )


def read_published(*, project_root: Path, directory: Path):
    path = directory / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"No completed artifact manifest at {path}")
    reference = DatasetArtifactRef(
        project_root=project_root, manifest_json=path.read_text()
    )
    verify_artifact(reference)
    return reference
