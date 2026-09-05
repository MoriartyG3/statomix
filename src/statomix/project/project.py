"""Top-level Statomix project workflow."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from fileverse.formats.excel import BaseExcel
from fileverse.formats.zarr import BaseZARR
from pandas.testing import assert_frame_equal

from statomix.dataset.base import normalize_display_label
from statomix.dataset.dataset import Dataset
from statomix.dataset.roles import normalize_dataset_role
from statomix.logging import get_logger
from statomix.project.analyzer.analyzer import Analyzer

DEFAULT_PROJECT_DIR = Path("multiomix") / "statomix"
logger = get_logger(name="Project")


class Project:
    """Persist datasets and their workflows in one Zarr v3 project store."""

    def __init__(
        self,
        project_name: str,
        project_dir: str | Path | None = None,
    ) -> None:
        if not project_name.strip():
            raise ValueError("project_name must not be empty")

        self.project_name = project_name
        self.project_dir = (
            Path.cwd() / DEFAULT_PROJECT_DIR
            if project_dir is None
            else Path(project_dir)
        )
        self._create_groups()
        self._init_meta()
        self._discover_datasets()
        self.analyzer = Analyzer(
            root_group=self.groups["analyzer_root"],
            dataset_name="project_datasets",
        )

    def _create_groups(self) -> None:
        project_path = self.project_dir / self.project_name
        self._zarr_storage = BaseZARR(path=project_path)
        root_group = self._zarr_storage.root_group
        self.groups: dict[str, Any] = {
            "root": root_group,
            "datasets_root": root_group.require_group("datasets"),
            "analyzer_root": root_group.require_group("project_analyzer"),
        }

    def _init_meta(self) -> None:
        stored_name = self.groups["root"].attrs.get("project_name")
        if stored_name is not None and stored_name != self.project_name:
            raise RuntimeError(
                f"Project store contains project_name={stored_name!r}, but "
                f"{self.project_name!r} was requested."
            )
        self.groups["root"].attrs["project_name"] = self.project_name

    def add_dataset(
        self,
        df: pd.DataFrame | None,
        dataset_name: str,
        display_label: str | None = None,
        dataset_role: str = "analysis",
    ) -> Dataset | None:

        resolved_display_label = normalize_display_label(
            dataset_name if display_label is None else display_label
        )
        resolved_dataset_role = normalize_dataset_role(dataset_role)

        project_datasets_meta = dict(self.groups["root"].attrs.get("datasets", {}))
        existing_meta = project_datasets_meta.get(dataset_name)
        if existing_meta and existing_meta.get("created_successfully"):
            existing_role = self.datasets[dataset_name].dataset_role
            if resolved_dataset_role != existing_role:
                raise RuntimeError(
                    f"Dataset {dataset_name!r} already has "
                    f"dataset_role={existing_role!r}."
                )
            self._report_existing_dataset(df=df, dataset_name=dataset_name)
            return None

        project_datasets_meta[dataset_name] = {"created_successfully": False}
        self.groups["root"].attrs["datasets"] = project_datasets_meta

        try:
            dataset = Dataset(
                df=df,
                dataset_name=dataset_name,
                display_label=resolved_display_label,
                dataset_role=resolved_dataset_role,
                root_group=self.groups["datasets_root"],
            )
        except Exception as exc:
            project_datasets_meta[dataset_name] = {
                "created_successfully": False,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            self.groups["root"].attrs["datasets"] = project_datasets_meta
            raise

        self.datasets[dataset_name] = dataset
        project_datasets_meta[dataset_name] = {
            "created_successfully": True,
            "dataset_role": dataset.dataset_role,
        }
        self.groups["root"].attrs["datasets"] = project_datasets_meta
        logger.info(
            "Dataset '%s' successfully initialized and registered.",
            dataset_name,
        )
        return dataset

    def _report_existing_dataset(
        self,
        *,
        df: pd.DataFrame | None,
        dataset_name: str,
    ) -> None:
        message = (
            f"Dataset '{dataset_name}' already exists in this project. "
            "Choose a unique name or delete the existing dataset."
        )
        if df is None:
            logger.warning(message)
            return

        existing_df = pd.read_parquet(
            path=self.datasets[dataset_name].paths["df"]["source"]
        ).fillna(value=pd.NA)
        supplied_df = df.fillna(value=pd.NA)
        try:
            assert_frame_equal(left=existing_df, right=supplied_df)
            logger.warning(message)
        except AssertionError as exc:
            logger.warning(
                "%s The supplied DataFrame is not identical to the saved " "DataFrame.",
                message,
            )
            logger.debug(str(exc))

    def set_dataset_role(
        self,
        *,
        dataset_name: str,
        dataset_role: str,
        reason: str,
    ) -> Dataset:
        """Persist an audited change to a dataset's analytical purpose.

        A role transition changes eligibility for future operations. It does
        not rewrite the source dataframe or remove existing pipeline artifacts.
        """

        if not isinstance(dataset_name, str):
            raise TypeError("dataset_name must be a string")

        normalized_name = dataset_name.strip()
        if not normalized_name:
            raise ValueError("dataset_name must not be empty")

        if not isinstance(reason, str):
            raise TypeError("reason must be a string")

        normalized_reason = reason.strip()
        if not normalized_reason:
            raise ValueError("A dataset-role transition reason is required.")

        resolved_role = normalize_dataset_role(dataset_role)

        if normalized_name not in self.datasets:
            raise KeyError(
                f"Dataset {normalized_name!r} is not registered in "
                f"project {self.project_name!r}."
            )

        dataset = self.datasets[normalized_name]
        previous_role = dataset.dataset_role

        if previous_role == resolved_role:
            logger.info(
                "Dataset '%s' already has dataset_role=%r; no role "
                "transition was recorded.",
                normalized_name,
                resolved_role,
            )
            return dataset

        dataset_group = dataset.groups["root"]
        existing_history = list(dataset_group.attrs.get("dataset_role_history", []))
        transition = {
            "sequence": len(existing_history) + 1,
            "changed_at_utc": datetime.now(UTC).isoformat(),
            "previous_role": previous_role,
            "dataset_role": resolved_role,
            "reason": normalized_reason,
        }

        project_datasets_meta = dict(self.groups["root"].attrs.get("datasets", {}))
        dataset_meta = dict(project_datasets_meta.get(normalized_name, {}))

        dataset_group.attrs["dataset_role"] = resolved_role
        dataset_group.attrs["dataset_role_history"] = [
            *existing_history,
            transition,
        ]

        dataset._dataset_role = resolved_role
        dataset.analyzer.dataset_role = resolved_role
        dataset.reference.dataset_role = resolved_role

        dataset_meta["dataset_role"] = resolved_role
        project_datasets_meta[normalized_name] = dataset_meta
        self.groups["root"].attrs["datasets"] = project_datasets_meta

        logger.info(
            "Changed dataset '%s' role from %r to %r. Reason: %s",
            normalized_name,
            previous_role,
            resolved_role,
            normalized_reason,
        )
        return dataset

    def _discover_datasets(self) -> None:
        self.datasets: dict[str, Dataset] = {}
        project_datasets_meta = self.groups["root"].attrs.get("datasets", {})
        for dataset_name, dataset_meta in project_datasets_meta.items():
            if dataset_meta.get("created_successfully"):
                self.datasets[dataset_name] = Dataset(
                    df=None,
                    dataset_name=dataset_name,
                    root_group=self.groups["datasets_root"],
                )
                logger.info(
                    "Discovered and loaded existing dataset: '%s'",
                    dataset_name,
                )

    def create_datatype_map_overview(
        self,
        version: int,
        config_version: int,
    ) -> Path:
        group_bundle = self.analyzer._get_group_bundle(
            version=version,
            config_version=config_version,
        )
        output_path = group_bundle["config"]["path"] / "datatype_map_overview.xlsx"
        if output_path.exists():
            logger.info("Datatype map overview already exists at %s.", output_path)
            return output_path

        analysis_datasets = {
            name: dataset
            for name, dataset in self.datasets.items()
            if dataset.dataset_role == "analysis"
        }
        if not analysis_datasets:
            raise ValueError("No analysis datasets are available for this overview.")

        with pd.ExcelWriter(path=output_path, engine="openpyxl") as writer:
            for dataset_name, dataset in analysis_datasets.items():
                group_analyzer = dataset.analyzer._get_group_analyzer(
                    version=version,
                    config_version=config_version,
                )
                datatype_map_df = group_analyzer._get_datatype_map_df()
                datatype_map_df.to_excel(
                    excel_writer=writer,
                    sheet_name=dataset_name,
                    index=False,
                )
        BaseExcel.format_cell_length(path=output_path)
        return output_path

    def combine_datasets(
        self,
        *,
        sources,
        mappings,
        identity_columns,
        dataset_name,
        display_label,
        reason,
        cohort_column="source_cohort",
    ):
        """Create a new dataset from exact artifact parents, with strict validation."""
        from statomix.pipelines.transformer.project import combine_datasets

        return combine_datasets(
            self,
            sources=sources,
            mappings=mappings,
            identity_columns=identity_columns,
            dataset_name=dataset_name,
            display_label=display_label,
            reason=reason,
            cohort_column=cohort_column,
        )
