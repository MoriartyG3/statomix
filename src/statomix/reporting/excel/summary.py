"""Render the stable dataset summary workbook."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from fileverse.formats.excel import BaseExcel

from statomix.storage.atomic import atomic_output_path


@dataclass(frozen=True, slots=True, kw_only=True)
class SummaryWorkbook:
    """The three tables written to ``summary.xlsx``."""

    numerical: pd.DataFrame
    normality: pd.DataFrame
    categorical: pd.DataFrame


class SummaryWorkbookRenderer:
    """Excel adapter; statistical services do not depend on openpyxl."""

    @staticmethod
    def render(*, workbook: SummaryWorkbook, path: Path) -> None:
        destination = Path(path)
        with atomic_output_path(destination=destination) as temporary_path:
            with pd.ExcelWriter(path=temporary_path, engine="openpyxl") as writer:
                workbook.numerical.to_excel(
                    excel_writer=writer,
                    index=True,
                    sheet_name="Numerical",
                )
                workbook.normality.to_excel(
                    excel_writer=writer,
                    index=True,
                    sheet_name="Normality Diagnostics",
                )
                workbook.categorical.to_excel(
                    excel_writer=writer,
                    index=True,
                    sheet_name="Categorical",
                )
            BaseExcel.format_cell_length(path=temporary_path)
