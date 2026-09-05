"""Immutable Transformer configurations independent of Analyzer numbering."""

from __future__ import annotations

import json
from pathlib import Path

from fileverse.formats.zarr import BaseZARR

from statomix.core.artifacts import canonical_json
from statomix.storage.artifacts import (
    artifact_lock,
    execution_fingerprint,
    load_artifact,
    publish_artifact,
    read_published,
)
from statomix.transformation.columns import apply_operations
from statomix.transformation.concatenation import concatenate_states
from statomix.transformation.operations import (
    apply_operations as apply_mixed_operations,
)
from statomix.transformation.specifications import ExcludeRows, operation_from_dict


class Transformer:
    """Versions pin parents; configurations pin the ordered operation plan.

    Reads do not allocate Zarr groups. Public writes require explicit positive
    version/config numbers and never overwrite completed output.
    """

    def __init__(self, *, dataset_group, dataset_name, project_root, project_name):
        self.dataset_group = dataset_group
        self.dataset_name = dataset_name
        self.project_root = Path(project_root).resolve()
        self.project_name = project_name
        self.root = BaseZARR.get_abs_path(dataset_group) / "transformer"

    def artifact(self, *, version: int, config_version: int):
        self._numbers(version, config_version)
        reference = read_published(
            project_root=self.project_root,
            directory=self.root
            / f"version{version}"
            / f"config{config_version}"
            / "data",
        )
        group_path = f"transformer/version{version}/config{config_version}"
        if group_path in self.dataset_group:
            recorded = (
                self.dataset_group[group_path].attrs.get("meta", {}).get("artifact_id")
            )
            if recorded is not None and recorded != reference.artifact_id:
                raise ValueError(
                    "Published manifest differs from registered artifact identity."
                )
        return reference

    @staticmethod
    def _numbers(version, config_version):
        if any(type(x) is not int or x < 1 for x in (version, config_version)):
            raise ValueError(
                "Explicit positive integer version/config_version required."
            )

    def create_data(
        self, *, source, operations, version: int, config_version: int, name: str = ""
    ):
        operations = tuple(operation_from_dict(op.to_dict()) for op in operations)
        specification = {
            "schema_version": 1,
            # Preserve the v1 identity of plans containing only the original
            # row-preserving column operations.
            "kind": (
                "operations"
                if any(isinstance(operation, ExcludeRows) for operation in operations)
                else "columns"
            ),
            "name": name,
            "operations": [op.to_dict() for op in operations],
        }
        return self._create(
            parents=(source,),
            specification=specification,
            version=version,
            config_version=config_version,
        )

    def create_concatenated_data(
        self,
        *,
        sources,
        mappings,
        identity_columns,
        version: int,
        config_version: int,
        reason: str,
        cohort_column: str = "source_cohort",
    ):
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("A concatenation reason is required.")
        if isinstance(identity_columns, str):
            raise TypeError("identity_columns must be a sequence, not a string.")
        specification = {
            "schema_version": 1,
            "kind": "concatenate",
            "mappings": [dict(m) for m in mappings],
            "identity_columns": list(identity_columns),
            "cohort_column": cohort_column,
            "reason": reason,
        }
        return self._create(
            parents=tuple(sources),
            specification=specification,
            version=version,
            config_version=config_version,
        )

    def _create(self, *, parents, specification, version, config_version):
        self._numbers(version, config_version)
        specification = {**specification, "execution": execution_fingerprint()}
        if not parents or any(
            p.project_root != self.project_root
            or p.manifest["project"] != self.project_name
            for p in parents
        ):
            raise ValueError(
                "All parents must belong to this project; cross-project imports are not implemented."
            )
        if len({p.artifact_id for p in parents}) != len(parents):
            raise ValueError("The same parent artifact was supplied more than once.")
        version_path = self.root / f"version{version}"
        config_path = version_path / f"config{config_version}"
        destination = config_path / "data"
        identities = [p.to_dict() for p in parents]
        with artifact_lock(self.root):
            parent_path = version_path / "parents.json"
            if (
                parent_path.exists()
                and json.loads(parent_path.read_text()) != identities
            ):
                raise ValueError(
                    "This Transformer version has different parents; choose a new version."
                )
            if destination.exists():
                existing = self.artifact(version=version, config_version=config_version)
                if (
                    existing.manifest["specification"]
                    != json.loads(canonical_json(specification))
                    or existing.manifest["parents"] != identities
                ):
                    raise ValueError(
                        "Completed configuration differs; choose a new configuration."
                    )
                return existing
            states = [load_artifact(p) for p in parents]
            for state in states:
                state.pairs.require_supported(operation="Transformer")
            exclusions = []
            if specification["kind"] == "columns":
                state, audit = apply_operations(
                    states[0],
                    [operation_from_dict(op) for op in specification["operations"]],
                )
            elif specification["kind"] == "operations":
                state, audit, exclusions = apply_mixed_operations(
                    states[0],
                    [operation_from_dict(op) for op in specification["operations"]],
                )
            else:
                state, audit = concatenate_states(
                    states,
                    mappings=specification["mappings"],
                    identity_columns=specification["identity_columns"],
                    cohort_column=specification["cohort_column"],
                )
            group = self.dataset_group.require_group("transformer")
            version_group = group.require_group(f"version{version}")
            config_group = version_group.require_group(f"config{config_version}")
            if not parent_path.exists():
                parent_path.write_text(canonical_json(identities), encoding="utf-8")
            identity = {
                "project": self.project_name,
                "dataset": self.dataset_name,
                "pipeline": "transformer",
                "version": version,
                "config_version": config_version,
            }
            reference = publish_artifact(
                project_root=self.project_root,
                destination=destination,
                state=state,
                identity=identity,
                parents=parents,
                specification=specification,
                audit=audit,
                exclusions=exclusions,
            )
            config_group.attrs["meta"] = {
                "status": "completed",
                "artifact_id": reference.artifact_id,
                "manifest": "data/manifest.json",
            }
            return reference
