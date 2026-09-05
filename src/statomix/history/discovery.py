"""Read-only discovery of Statomix project sources and artifacts."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from statomix.core.artifacts import digest_json, safe_relative
from statomix.curation.columns import ColReport
from statomix.curation.survival.report import SurvPairs
from statomix.history.model import (
    HistoryEdge,
    HistoryNode,
    HistoryWarning,
    ProjectHistory,
)
from statomix.history.validation import validate_graph
from statomix.storage.hashing import sha256_file
from statomix.storage.parquet_metadata import load_category_rank_metadata

VERSION_PATTERN = re.compile(r"^version(?P<value>[1-9][0-9]*)$")
CONFIG_PATTERN = re.compile(r"^config(?P<value>[1-9][0-9]*)$")


def _number(path: Path, pattern: re.Pattern[str]) -> int:
    match = pattern.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Invalid versioned path component: {path.name!r}.")
    return int(match.group("value"))


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str, allow_nan=False))


def _parquet_shape(path: Path) -> tuple[int, int]:
    """Return the logical pandas shape stored in a Parquet file.

    Parquet's physical schema can contain serialized pandas index fields.
    Those fields are storage metadata, not DataFrame columns, and therefore
    must not be included in the reported analytical shape.
    """

    parquet_file = pq.ParquetFile(path)
    row_count = int(parquet_file.metadata.num_rows)

    schema = parquet_file.schema_arrow
    pandas_metadata = schema.pandas_metadata or {}

    physical_index_columns = {
        index_column
        for index_column in pandas_metadata.get("index_columns", [])
        if isinstance(index_column, str)
    }

    logical_column_count = sum(
        column_name not in physical_index_columns for column_name in schema.names
    )

    return row_count, logical_column_count


def _semantic_summary(
    *,
    profiles_path: Path | None,
    pairs_path: Path | None,
    dataframe_path: Path | None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}

    if profiles_path is not None and profiles_path.is_file():
        profiles = ColReport.load_col_profiles(path=profiles_path)
        summary["datatype_counts"] = dict(
            sorted(
                Counter(profile.col_type.value for profile in profiles.values()).items()
            )
        )

    if pairs_path is not None and pairs_path.is_file():
        pairs = SurvPairs.load(pairs_path)
        summary["survival_endpoints"] = {
            label: {
                "event": pair.event_profile.col_name,
                "duration": pair.time_profile.col_name,
            }
            for label, pair in sorted(pairs.pairs.items())
        }

    if dataframe_path is not None and dataframe_path.is_file():
        ranks = load_category_rank_metadata(dataframe_path)
        summary["ranked_categorical_columns"] = sorted(ranks.get("columns", {}))

    return summary


class ProjectHistoryDiscovery:
    """Build a normalized graph without writing to the project store."""

    def __init__(
        self,
        *,
        project,
        verify_checksums: bool,
        include_files: bool,
    ) -> None:
        if type(verify_checksums) is not bool:
            raise TypeError("verify_checksums must be Boolean.")
        if type(include_files) is not bool:
            raise TypeError("include_files must be Boolean.")

        self.project = project
        self.project_root = (Path(project.project_dir) / project.project_name).resolve()
        self.verify_checksums = verify_checksums
        self.include_files = include_files
        self.nodes: dict[str, HistoryNode] = {}
        self.edges: dict[tuple[str, str, str], HistoryEdge] = {}
        self.warnings: list[HistoryWarning] = []

    def discover(self) -> ProjectHistory:
        for dataset_name in sorted(self.project.datasets):
            dataset = self.project.datasets[dataset_name]
            self._add_source(dataset)
            self._add_cleaner_configurations(dataset)
            self._add_published_artifacts(dataset)
            self._add_analyzer_configurations(dataset)

        nodes = tuple(sorted(self.nodes.values(), key=lambda item: item.node_id))
        edges = tuple(
            sorted(
                self.edges.values(),
                key=lambda item: (
                    item.source,
                    item.target,
                    item.relationship,
                ),
            )
        )
        structural = validate_graph(nodes=nodes, edges=edges)
        warnings = tuple(
            sorted(
                {*self.warnings, *structural},
                key=lambda item: (
                    item.severity,
                    item.code,
                    item.node_id or "",
                    item.message,
                ),
            )
        )
        return ProjectHistory(
            project_name=self.project.project_name,
            nodes=nodes,
            edges=edges,
            warnings=warnings,
        )

    def _relative(self, path: Path) -> str:
        return path.resolve().relative_to(self.project_root).as_posix()

    def _warn(
        self,
        *,
        code: str,
        severity: str,
        message: str,
        node_id: str | None = None,
        path: Path | None = None,
    ) -> None:
        self.warnings.append(
            HistoryWarning(
                code=code,
                severity=severity,
                message=message,
                node_id=node_id,
                path=(self._relative(path) if path is not None else None),
            )
        )

    def _add_node(self, node: HistoryNode) -> None:
        existing = self.nodes.get(node.node_id)
        if existing is None:
            self.nodes[node.node_id] = node
            return

        status_priority = {
            "unknown": 0,
            "not_applicable": 1,
            "completed": 2,
            "computed": 2,
            "verified": 3,
            "incomplete": 4,
            "invalid": 5,
        }
        status = max(
            (existing.status, node.status),
            key=lambda value: status_priority.get(value, 0),
        )
        self.nodes[node.node_id] = replace(
            existing,
            label=node.label or existing.label,
            display_label=node.display_label or existing.display_label,
            dataset_role=node.dataset_role or existing.dataset_role,
            status=status,
            rows=node.rows if node.rows is not None else existing.rows,
            columns=(node.columns if node.columns is not None else existing.columns),
            reason=node.reason or existing.reason,
            path=node.path or existing.path,
            attributes={**existing.attributes, **node.attributes},
        )

    def _add_edge(
        self,
        *,
        source: str,
        target: str,
        relationship: str,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        key = source, target, relationship
        self.edges[key] = HistoryEdge(
            source=source,
            target=target,
            relationship=relationship,
            attributes=dict(attributes or {}),
        )

    def _dataset_fields(self, dataset_name: str) -> dict[str, str | None]:
        dataset = self.project.datasets.get(dataset_name)
        if dataset is None:
            return {
                "display_label": dataset_name,
                "dataset_role": None,
            }
        return {
            "display_label": dataset.display_label,
            "dataset_role": dataset.dataset_role,
        }

    def _source_id(self, dataset_name: str) -> str:
        return f"source:{dataset_name}"

    def _cleaner_id(
        self,
        *,
        dataset_name: str,
        version: int,
        config_version: int,
    ) -> str:
        return f"cleaner:{dataset_name}:v{version}:c{config_version}"

    def _manifest_id(self, manifest: dict[str, Any]) -> str:
        if manifest.get("pipeline") == "cleaner":
            return self._cleaner_id(
                dataset_name=str(manifest["dataset"]),
                version=int(manifest["version"]),
                config_version=int(manifest["config_version"]),
            )
        return f"artifact:{digest_json(manifest)}"

    def _add_source(self, dataset) -> None:
        path = Path(dataset.paths["df"]["source"]).resolve()
        node_id = self._source_id(dataset.dataset_name)
        rows = columns = None
        status = "completed"
        attributes: dict[str, Any] = {
            "dataset_role_history": _json_safe(
                list(
                    dataset.groups["root"].attrs.get(
                        "dataset_role_history",
                        [],
                    )
                )
            )
        }

        if path.is_file():
            try:
                rows, columns = _parquet_shape(path)
                if self.verify_checksums:
                    attributes["sha256"] = sha256_file(path=path)
                    attributes["checksum_status"] = "computed"
                else:
                    attributes["checksum_status"] = "not_checked"
            except Exception as exc:  # noqa: BLE001 - discovery records failures
                status = "invalid"
                self._warn(
                    code="source_read_failure",
                    severity="error",
                    message=str(exc),
                    node_id=node_id,
                    path=path,
                )
        else:
            status = "incomplete"
            self._warn(
                code="missing_source_dataframe",
                severity="error",
                message=f"Source dataframe is missing for {dataset.dataset_name!r}.",
                node_id=node_id,
                path=path,
            )

        self._add_node(
            HistoryNode(
                node_id=node_id,
                node_type="source",
                label=f"{dataset.display_label}\nSource",
                dataset=dataset.dataset_name,
                display_label=dataset.display_label,
                dataset_role=dataset.dataset_role,
                pipeline="source",
                status=status,
                rows=rows,
                columns=columns,
                path=self._relative(path),
                attributes=attributes,
            )
        )

    def _group_metadata(
        self,
        *,
        dataset,
        pipeline: str,
        version: int,
        config_version: int,
    ) -> dict[str, Any]:
        key = f"{pipeline}/version{version}/config{config_version}"
        try:
            if key in dataset.groups["root"]:
                return _json_safe(
                    dict(dataset.groups["root"][key].attrs.get("meta", {}))
                )
        except Exception as exc:  # noqa: BLE001 - discovery records failures
            self._warn(
                code="zarr_metadata_read_failure",
                severity="warning",
                message=f"Could not read {key!r} metadata: {exc}",
            )
        return {}

    def _add_cleaner_configurations(self, dataset) -> None:
        cleaner_root = self.project_root / "datasets" / dataset.dataset_name / "cleaner"
        for dataframe_path in sorted(
            cleaner_root.glob("version*/config*/curated_data/df.parquet")
        ):
            curated_root = dataframe_path.parent
            config_path = curated_root.parent
            version_path = config_path.parent
            try:
                version = _number(version_path, VERSION_PATTERN)
                config_version = _number(config_path, CONFIG_PATTERN)
            except ValueError as exc:
                self._warn(
                    code="invalid_cleaner_path",
                    severity="error",
                    message=str(exc),
                    path=curated_root,
                )
                continue

            node_id = self._cleaner_id(
                dataset_name=dataset.dataset_name,
                version=version,
                config_version=config_version,
            )
            profiles_path = curated_root / "col_profiles.parquet"
            pairs_path = curated_root / "surv_pairs.parquet"
            required = [dataframe_path, profiles_path, pairs_path]
            missing = [path for path in required if not path.is_file()]
            status = "incomplete" if missing else "completed"
            rows = columns = None
            attributes: dict[str, Any] = {
                "configuration_metadata": self._group_metadata(
                    dataset=dataset,
                    pipeline="cleaner",
                    version=version,
                    config_version=config_version,
                )
            }

            if missing:
                self._warn(
                    code="incomplete_cleaner_artifact",
                    severity="error",
                    message=(
                        "Cleaner artifact is missing required files: "
                        f"{[self._relative(path) for path in missing]!r}."
                    ),
                    node_id=node_id,
                    path=curated_root,
                )
            try:
                rows, columns = _parquet_shape(dataframe_path)
                attributes.update(
                    _semantic_summary(
                        profiles_path=profiles_path,
                        pairs_path=pairs_path,
                        dataframe_path=dataframe_path,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - discovery records failures
                status = "invalid"
                self._warn(
                    code="cleaner_artifact_read_failure",
                    severity="error",
                    message=str(exc),
                    node_id=node_id,
                    path=curated_root,
                )

            if self.verify_checksums:
                attributes["file_sha256"] = {
                    path.name: sha256_file(path=path)
                    for path in required
                    if path.is_file()
                }
                attributes["checksum_status"] = "computed"
            else:
                attributes["checksum_status"] = "not_checked"

            config_meta = attributes["configuration_metadata"]
            reason = config_meta.get("reason")
            name = config_meta.get("name") or f"Cleaner v{version}/c{config_version}"
            self._add_node(
                HistoryNode(
                    node_id=node_id,
                    node_type="cleaner",
                    label=f"{name}\n{rows or '?'} × {columns or '?'}",
                    dataset=dataset.dataset_name,
                    display_label=dataset.display_label,
                    dataset_role=dataset.dataset_role,
                    pipeline="cleaner",
                    version=version,
                    config_version=config_version,
                    status=status,
                    rows=rows,
                    columns=columns,
                    reason=reason,
                    path=self._relative(curated_root),
                    attributes=attributes,
                )
            )
            self._add_edge(
                source=self._source_id(dataset.dataset_name),
                target=node_id,
                relationship="cleaned_from",
            )

    def _add_published_artifacts(self, dataset) -> None:
        dataset_root = self.project_root / "datasets" / dataset.dataset_name
        manifests = [
            *dataset_root.glob("reference/version*/config*/data/manifest.json"),
            *dataset_root.glob("transformer/version*/config*/data/manifest.json"),
        ]
        for manifest_path in sorted(manifests):
            self._add_manifest_path(manifest_path)

    def _add_manifest_path(self, manifest_path: Path) -> str | None:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - discovery records failures
            self._warn(
                code="manifest_read_failure",
                severity="error",
                message=str(exc),
                path=manifest_path,
            )
            return None
        return self._add_manifest(
            manifest=manifest,
            manifest_path=manifest_path,
            expected_artifact_id=None,
        )

    def _add_manifest(
        self,
        *,
        manifest: dict[str, Any],
        manifest_path: Path | None,
        expected_artifact_id: str | None,
    ) -> str:
        artifact_id = digest_json(manifest)
        node_id = self._manifest_id(manifest)
        pipeline = str(manifest.get("pipeline", "unknown"))
        dataset_name = str(manifest.get("dataset", "unknown"))
        version = manifest.get("version")
        config_version = manifest.get("config_version")
        fields = self._dataset_fields(dataset_name)

        if expected_artifact_id is not None and artifact_id != expected_artifact_id:
            self._warn(
                code="parent_artifact_id_mismatch",
                severity="error",
                message=(
                    f"Embedded parent declares {expected_artifact_id}, but its "
                    f"manifest hashes to {artifact_id}."
                ),
                node_id=node_id,
                path=manifest_path,
            )

        status = str(manifest.get("status", "unknown"))
        attributes: dict[str, Any] = {
            "artifact_id": artifact_id,
            "manifest_status": status,
        }
        specification = dict(manifest.get("specification", {}))
        specification.pop("execution", None)
        if specification:
            attributes["specification"] = _json_safe(specification)
            attributes["operation_kind"] = specification.get("kind")

        metadata = manifest.get("metadata", {})
        if isinstance(metadata, dict):
            attributes["endpoint_definitions"] = _json_safe(
                metadata.get("endpoint_definitions", {})
            )
            attributes["unit_columns"] = {
                name: record.get("unit")
                for name, record in metadata.get("columns", {}).items()
                if isinstance(record, dict) and record.get("unit") is not None
            }

        file_summary, dataframe_path, profiles_path, pairs_path, integrity = (
            self._inspect_manifest_files(
                manifest=manifest,
                node_id=node_id,
            )
        )
        attributes["checksum_status"] = integrity
        attributes["file_count"] = len(file_summary)
        if self.include_files:
            attributes["files"] = file_summary

        rows = columns = None
        if dataframe_path is not None and dataframe_path.is_file():
            try:
                rows, columns = _parquet_shape(dataframe_path)
                attributes.update(
                    _semantic_summary(
                        profiles_path=profiles_path,
                        pairs_path=pairs_path,
                        dataframe_path=dataframe_path,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - discovery records failures
                status = "invalid"
                self._warn(
                    code="artifact_semantic_read_failure",
                    severity="error",
                    message=str(exc),
                    node_id=node_id,
                    path=dataframe_path,
                )

        if integrity == "failed":
            status = "invalid"

        reason = (
            specification.get("reason") or manifest.get("declaration_reason") or None
        )
        name = specification.get("name") or pipeline.title()
        path = (
            self._relative(manifest_path.parent) if manifest_path is not None else None
        )
        self._add_node(
            HistoryNode(
                node_id=node_id,
                node_type=pipeline,
                label=f"{name}\nv{version}/c{config_version}",
                dataset=dataset_name,
                display_label=fields["display_label"],
                dataset_role=fields["dataset_role"],
                pipeline=pipeline,
                version=int(version) if version is not None else None,
                config_version=(
                    int(config_version) if config_version is not None else None
                ),
                status=status,
                rows=rows,
                columns=columns,
                reason=reason,
                path=path,
                attributes=attributes,
            )
        )

        if pipeline in {"cleaner", "reference"}:
            relationship = "cleaned_from" if pipeline == "cleaner" else "reference_from"
            self._add_edge(
                source=self._source_id(dataset_name),
                target=node_id,
                relationship=relationship,
            )

        parents = manifest.get("parents", [])
        for index, parent in enumerate(parents):
            try:
                parent_id = self._add_manifest(
                    manifest=dict(parent["manifest"]),
                    manifest_path=None,
                    expected_artifact_id=parent.get("artifact_id"),
                )
            except Exception as exc:  # noqa: BLE001 - discovery records failures
                self._warn(
                    code="embedded_parent_failure",
                    severity="error",
                    message=str(exc),
                    node_id=node_id,
                    path=manifest_path,
                )
                continue
            self._add_edge(
                source=parent_id,
                target=node_id,
                relationship=self._parent_relationship(
                    pipeline=pipeline,
                    specification=specification,
                    parent_index=index,
                ),
                attributes={"parent_index": index},
            )
        return node_id

    def _inspect_manifest_files(
        self,
        *,
        manifest: dict[str, Any],
        node_id: str,
    ) -> tuple[list[dict[str, Any]], Path | None, Path | None, Path | None, str]:
        summaries = []
        paths: dict[str, Path] = {}
        failed = False

        for name, record in sorted(manifest.get("files", {}).items()):
            try:
                path = safe_relative(self.project_root, record["path"])
            except Exception as exc:  # noqa: BLE001 - discovery records failures
                failed = True
                self._warn(
                    code="unsafe_artifact_path",
                    severity="error",
                    message=str(exc),
                    node_id=node_id,
                )
                continue

            paths[name] = path
            summary = {
                "name": name,
                "path": record["path"],
                "expected_sha256": record.get("sha256"),
                "status": "not_checked",
            }
            if not path.is_file():
                failed = True
                summary["status"] = "missing"
                self._warn(
                    code="missing_artifact_file",
                    severity="error",
                    message=f"Artifact file {name!r} is missing.",
                    node_id=node_id,
                    path=path,
                )
            elif self.verify_checksums:
                observed = sha256_file(path=path)
                summary["observed_sha256"] = observed
                if observed == record.get("sha256"):
                    summary["status"] = "verified"
                else:
                    failed = True
                    summary["status"] = "checksum_mismatch"
                    self._warn(
                        code="artifact_checksum_mismatch",
                        severity="error",
                        message=f"Checksum mismatch for artifact file {name!r}.",
                        node_id=node_id,
                        path=path,
                    )
            summaries.append(summary)

        integrity = (
            "failed"
            if failed
            else ("verified" if self.verify_checksums else "not_checked")
        )
        return (
            summaries,
            paths.get("df"),
            paths.get("col_profiles"),
            paths.get("surv_pairs"),
            integrity,
        )

    @staticmethod
    def _parent_relationship(
        *,
        pipeline: str,
        specification: dict[str, Any],
        parent_index: int,
    ) -> str:
        if pipeline != "transformer":
            return "derived_from"
        kind = specification.get("kind")
        if kind == "keyed_update":
            return "base_parent" if parent_index == 0 else "update_parent"
        if kind == "concatenate":
            return "concatenated_from"
        return "transformed_from"

    def _add_analyzer_configurations(self, dataset) -> None:
        analyzer_root = (
            self.project_root / "datasets" / dataset.dataset_name / "analyzer"
        )
        for config_path in sorted(analyzer_root.glob("version*/config*")):
            if not config_path.is_dir():
                continue
            try:
                version = _number(config_path.parent, VERSION_PATTERN)
                config_version = _number(config_path, CONFIG_PATTERN)
            except ValueError as exc:
                self._warn(
                    code="invalid_analyzer_path",
                    severity="error",
                    message=str(exc),
                    path=config_path,
                )
                continue

            node_id = f"analyzer:{dataset.dataset_name}:v{version}:c{config_version}"
            metadata = self._group_metadata(
                dataset=dataset,
                pipeline="analyzer",
                version=version,
                config_version=config_version,
            )
            binding_path = config_path / "input_artifact.json"
            legacy_path = config_path / "group_analyzer_path.yaml"
            status = (
                "completed"
                if binding_path.is_file() or legacy_path.is_file()
                else "incomplete"
            )
            attributes: dict[str, Any] = {
                "configuration_metadata": metadata,
                "input_kind": (
                    "artifact"
                    if binding_path.is_file()
                    else ("legacy_cleaner" if legacy_path.is_file() else "missing")
                ),
            }
            reason = metadata.get("reason")
            self._add_node(
                HistoryNode(
                    node_id=node_id,
                    node_type="analyzer",
                    label=f"Analyzer\nv{version}/c{config_version}",
                    dataset=dataset.dataset_name,
                    display_label=dataset.display_label,
                    dataset_role=dataset.dataset_role,
                    pipeline="analyzer",
                    version=version,
                    config_version=config_version,
                    status=status,
                    reason=reason,
                    path=self._relative(config_path),
                    attributes=attributes,
                )
            )

            if binding_path.is_file():
                self._add_analyzer_binding(
                    node_id=node_id,
                    binding_path=binding_path,
                )
            elif legacy_path.is_file():
                self._add_edge(
                    source=self._cleaner_id(
                        dataset_name=dataset.dataset_name,
                        version=version,
                        config_version=config_version,
                    ),
                    target=node_id,
                    relationship="analyzed_from",
                )
            else:
                self._warn(
                    code="missing_analyzer_input",
                    severity="warning",
                    message="Analyzer configuration has no recorded input binding.",
                    node_id=node_id,
                    path=config_path,
                )

            summary_path = config_path / "summary.xlsx"
            if summary_path.is_file():
                report_id = f"report:{node_id}:summary"
                self._add_node(
                    HistoryNode(
                        node_id=report_id,
                        node_type="report",
                        label="Summary report",
                        dataset=dataset.dataset_name,
                        display_label=dataset.display_label,
                        dataset_role=dataset.dataset_role,
                        pipeline="report",
                        version=version,
                        config_version=config_version,
                        status="completed",
                        path=self._relative(summary_path),
                        attributes={"kind": "summary_workbook"},
                    )
                )
                self._add_edge(
                    source=node_id,
                    target=report_id,
                    relationship="reported_by",
                )

            self._add_survival_report(
                dataset=dataset,
                analyzer_node_id=node_id,
                config_path=config_path,
                version=version,
                config_version=config_version,
            )

    def _add_analyzer_binding(self, *, node_id: str, binding_path: Path) -> None:
        try:
            binding = json.loads(binding_path.read_text(encoding="utf-8"))
            source = binding["source"]
            parent_id = self._add_manifest(
                manifest=dict(source["manifest"]),
                manifest_path=None,
                expected_artifact_id=source.get("artifact_id"),
            )
            self._add_edge(
                source=parent_id,
                target=node_id,
                relationship="analyzed_from",
            )
            existing = self.nodes[node_id]
            self.nodes[node_id] = replace(
                existing,
                attributes={
                    **existing.attributes,
                    "binding_sha256": digest_json(binding),
                    "survival_evaluation": _json_safe(
                        binding.get("survival_evaluation", {})
                    ),
                },
            )
        except Exception as exc:  # noqa: BLE001 - discovery records failures
            self._warn(
                code="analyzer_binding_failure",
                severity="error",
                message=str(exc),
                node_id=node_id,
                path=binding_path,
            )

    def _add_survival_report(
        self,
        *,
        dataset,
        analyzer_node_id: str,
        config_path: Path,
        version: int,
        config_version: int,
    ) -> None:
        survival_root = config_path / "surv"
        manifest_path = survival_root / "report_manifest.json"
        descriptives_path = survival_root / "descriptives.xlsx"
        if not manifest_path.is_file() and not descriptives_path.is_file():
            return

        node_id = f"report:{analyzer_node_id}:survival"
        status = "completed"
        attributes: dict[str, Any] = {"kind": "survival"}
        path = manifest_path if manifest_path.is_file() else descriptives_path

        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                recorded_id = manifest.get("report_id")
                unsigned = dict(manifest)
                unsigned.pop("report_id", None)
                if recorded_id != digest_json(unsigned):
                    status = "invalid"
                    self._warn(
                        code="survival_report_id_mismatch",
                        severity="error",
                        message="Survival report manifest identity is invalid.",
                        node_id=node_id,
                        path=manifest_path,
                    )
                attributes.update(
                    {
                        "report_id": recorded_id,
                        "input_artifact_id": manifest.get("input_artifact_id"),
                        "plots": _json_safe(manifest.get("plots", [])),
                        "file_count": len(manifest.get("files", [])),
                    }
                )
                if self.include_files:
                    attributes["files"] = _json_safe(manifest.get("files", []))
                if self.verify_checksums:
                    for record in manifest.get("files", []):
                        file_path = safe_relative(survival_root, record["path"])
                        if not file_path.is_file() or sha256_file(
                            path=file_path
                        ) != record.get("sha256"):
                            status = "invalid"
                            self._warn(
                                code="survival_report_file_mismatch",
                                severity="error",
                                message=(
                                    "A survival report file is missing or has "
                                    "a checksum mismatch."
                                ),
                                node_id=node_id,
                                path=file_path,
                            )
            except Exception as exc:  # noqa: BLE001 - discovery records failures
                status = "invalid"
                self._warn(
                    code="survival_report_read_failure",
                    severity="error",
                    message=str(exc),
                    node_id=node_id,
                    path=manifest_path,
                )

        self._add_node(
            HistoryNode(
                node_id=node_id,
                node_type="report",
                label="Survival reports",
                dataset=dataset.dataset_name,
                display_label=dataset.display_label,
                dataset_role=dataset.dataset_role,
                pipeline="report",
                version=version,
                config_version=config_version,
                status=status,
                path=self._relative(path),
                attributes=attributes,
            )
        )
        self._add_edge(
            source=analyzer_node_id,
            target=node_id,
            relationship="reported_by",
        )


def discover_project_history(
    *,
    project,
    verify_checksums: bool = True,
    include_files: bool = False,
) -> ProjectHistory:
    """Return a read-only normalized history graph for one project."""

    return ProjectHistoryDiscovery(
        project=project,
        verify_checksums=verify_checksums,
        include_files=include_files,
    ).discover()
