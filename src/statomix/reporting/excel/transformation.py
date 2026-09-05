"""Human-readable audit of deterministic dataset transformations."""

from __future__ import annotations

import pandas as pd

from statomix.core.artifacts import canonical_json


def write_transformation_report(
    *,
    path,
    audit,
    parents,
    specification,
    exclusions=(),
    column_updates=(),
    unused_updates=(),
):
    parent_rows = []
    for position, parent in enumerate(parents):
        manifest = parent.manifest
        for name, artifact in manifest["files"].items():
            parent_rows.append(
                {
                    "parent_order": position,
                    "dataset": manifest["dataset"],
                    "artifact_id": parent.artifact_id,
                    "file": name,
                    **artifact,
                }
            )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(audit).to_excel(writer, sheet_name="Operations", index=False)
        if exclusions:
            pd.DataFrame(exclusions).to_excel(
                writer,
                sheet_name="Excluded Rows",
                index=False,
            )

        if column_updates:
            pd.DataFrame(column_updates).to_excel(
                writer,
                sheet_name="Column Updates",
                index=False,
            )

        if unused_updates:
            pd.DataFrame(unused_updates).to_excel(
                writer,
                sheet_name="Unused Update Rows",
                index=False,
            )
        pd.DataFrame(parent_rows).to_excel(writer, sheet_name="Parents", index=False)
        # Split long specifications before Excel's cell-length limit.
        records = []
        for key, value in specification.items():
            serialized = canonical_json(value)
            for start in range(0, len(serialized), 30000):
                records.append(
                    {
                        "field": key,
                        "part": start // 30000 + 1,
                        "value": serialized[start : start + 30000],
                    }
                )
        pd.DataFrame(records).to_excel(writer, sheet_name="Specification", index=False)
        for sheet in writer.book.worksheets:
            sheet.freeze_panes = "A2"
            for row in sheet.iter_rows():
                for cell in row:
                    if isinstance(cell.value, str):
                        cell.data_type = "s"
            for column in sheet.columns:
                sheet.column_dimensions[column[0].column_letter].width = 24
