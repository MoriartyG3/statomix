"""Typed, deterministic project-history graph records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from statomix.core.artifacts import digest_json


def _sorted_mapping(value: dict[str, Any]) -> dict[str, Any]:
    return {key: value[key] for key in sorted(value)}


@dataclass(frozen=True, slots=True, kw_only=True)
class HistoryNode:
    """One persisted source, artifact, analysis configuration, or report."""

    node_id: str
    node_type: str
    label: str
    dataset: str | None
    display_label: str | None
    dataset_role: str | None
    pipeline: str
    version: int | None = None
    config_version: int | None = None
    status: str = "unknown"
    rows: int | None = None
    columns: int | None = None
    reason: str | None = None
    path: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "label": self.label,
            "dataset": self.dataset,
            "display_label": self.display_label,
            "dataset_role": self.dataset_role,
            "pipeline": self.pipeline,
            "version": self.version,
            "config_version": self.config_version,
            "status": self.status,
            "rows": self.rows,
            "columns": self.columns,
            "reason": self.reason,
            "path": self.path,
            "attributes": _sorted_mapping(dict(self.attributes)),
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class HistoryEdge:
    """A typed provenance relationship directed parent to child."""

    source: str
    target: str
    relationship: str
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relationship": self.relationship,
            "attributes": _sorted_mapping(dict(self.attributes)),
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class HistoryWarning:
    """A non-silent discovery or integrity problem."""

    code: str
    severity: str
    message: str
    node_id: str | None = None
    path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "node_id": self.node_id,
            "path": self.path,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class ProjectHistory:
    """A deterministic snapshot of the discovered project lineage."""

    project_name: str
    nodes: tuple[HistoryNode, ...]
    edges: tuple[HistoryEdge, ...]
    warnings: tuple[HistoryWarning, ...] = ()
    schema_version: int = 1

    def __post_init__(self) -> None:
        node_ids = [node.node_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("Project history contains duplicate node identifiers.")

        edge_keys = [
            (edge.source, edge.target, edge.relationship) for edge in self.edges
        ]
        if len(edge_keys) != len(set(edge_keys)):
            raise ValueError("Project history contains duplicate typed edges.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "project_name": self.project_name,
            "nodes": [
                node.to_dict()
                for node in sorted(self.nodes, key=lambda item: item.node_id)
            ],
            "edges": [
                edge.to_dict()
                for edge in sorted(
                    self.edges,
                    key=lambda item: (
                        item.source,
                        item.target,
                        item.relationship,
                    ),
                )
            ],
            "warnings": [
                warning.to_dict()
                for warning in sorted(
                    self.warnings,
                    key=lambda item: (
                        item.severity,
                        item.code,
                        item.node_id or "",
                        item.message,
                    ),
                )
            ],
        }

    @property
    def history_id(self) -> str:
        return digest_json(self.to_dict())

    def nodes_frame(self) -> pd.DataFrame:
        rows = []
        for node in sorted(self.nodes, key=lambda item: item.node_id):
            record = node.to_dict()
            record.pop("attributes")
            rows.append(record)
        return pd.DataFrame(rows)

    def edges_frame(self) -> pd.DataFrame:
        rows = []
        for edge in sorted(
            self.edges,
            key=lambda item: (item.source, item.target, item.relationship),
        ):
            record = edge.to_dict()
            record.pop("attributes")
            rows.append(record)
        return pd.DataFrame(rows)

    def warnings_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [warning.to_dict() for warning in self.warnings],
            columns=["severity", "code", "message", "node_id", "path"],
        )
