"""Dataset-level analysis configuration workbook renderer."""

from pathlib import Path

import pandas as pd
from fileverse.formats.excel import BaseExcel

from statomix.analysis.survival import MultiClassSurv
from statomix.analysis.survival.thresholds import MinimumPValue
from statomix.reporting.excel.validation import add_datatype_list_validations
from statomix.storage.atomic import atomic_output_path

SURVIVAL_MODULES = (MultiClassSurv, MinimumPValue)


class AnalysisConfig:
    """Render the stable dataset analysis configuration workbook."""

    @staticmethod
    def create_analysis_config(
        path: Path,
        datatype_map_df: pd.DataFrame,
    ) -> None:
        destination = Path(path)
        with atomic_output_path(destination=destination) as temporary_path:
            with pd.ExcelWriter(path=temporary_path, engine="openpyxl") as writer:
                datatype_map_df.to_excel(
                    excel_writer=writer,
                    sheet_name="Datatype Map",
                    index=False,
                )

            BaseExcel.format_cell_length(path=temporary_path)
            AnalysisConfig.add_survival_modules(path=temporary_path)

    @staticmethod
    def add_survival_modules(path: Path) -> None:
        with pd.ExcelWriter(
            path=path,
            engine="openpyxl",
            mode="a",
        ) as writer:
            for survival_module in SURVIVAL_MODULES:
                survival_module.get_config_df().to_excel(
                    excel_writer=writer,
                    sheet_name=survival_module.MODULE_NAME,
                    index=False,
                )

        for survival_module in SURVIVAL_MODULES:
            AnalysisConfig._add_validation_to_analysis_config_file(
                path=path,
                sheet_name=survival_module.MODULE_NAME,
            )

    @staticmethod
    def _add_validation_to_analysis_config_file(
        path: Path,
        sheet_name: str,
        max_row: int = 500,
    ) -> None:
        add_datatype_list_validations(
            path=Path(path),
            sheet_name=sheet_name,
            max_row=max_row,
            style_sheet=True,
        )
