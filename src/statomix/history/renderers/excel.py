"""Tabular project-history audit export."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

from statomix.core.artifacts import canonical_json
from statomix.history.model import ProjectHistory


def _attributes_frame(history: ProjectHistory) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "node_id": node.node_id,
                "attributes_json": canonical_json(node.attributes),
            }
            for node in history.nodes
        ],
        columns=["node_id", "attributes_json"],
    )


def _files_frame(history: ProjectHistory) -> pd.DataFrame:
    rows = []
    for node in history.nodes:
        for record in node.attributes.get("files", []):
            rows.append(
                {
                    "node_id": node.node_id,
                    "name": record.get("name"),
                    "path": record.get("path"),
                    "status": record.get("status"),
                    "expected_sha256": record.get("expected_sha256"),
                    "observed_sha256": record.get("observed_sha256"),
                    "record_json": json.dumps(record, sort_keys=True),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "node_id",
            "name",
            "path",
            "status",
            "expected_sha256",
            "observed_sha256",
            "record_json",
        ],
    )


def render_history_excel(*, history: ProjectHistory, destination: Path) -> None:
    """Write nodes, edges, warnings, attributes, and optional files."""

    with pd.ExcelWriter(destination, engine="openpyxl") as writer:
        history.nodes_frame().to_excel(writer, sheet_name="Nodes", index=False)
        history.edges_frame().to_excel(writer, sheet_name="Edges", index=False)
        history.warnings_frame().to_excel(writer, sheet_name="Warnings", index=False)
        _attributes_frame(history).to_excel(
            writer,
            sheet_name="Node Attributes",
            index=False,
        )
        files = _files_frame(history)
        if not files.empty:
            files.to_excel(writer, sheet_name="Files", index=False)

    workbook = load_workbook(destination)
    try:
        for worksheet in workbook.worksheets:
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for column_cells in worksheet.columns:
                values = [
                    "" if cell.value is None else str(cell.value)
                    for cell in column_cells
                ]
                width = min(max((len(value) for value in values), default=0) + 2, 80)
                worksheet.column_dimensions[column_cells[0].column_letter].width = width
        workbook.save(destination)
    finally:
        workbook.close()
