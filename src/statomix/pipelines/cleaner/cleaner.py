"""Cleaner workflow coordinating human-in-the-loop curation artifacts."""

from __future__ import annotations

import shutil
from collections.abc import Collection, Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd
from fileverse.formats.yaml import BaseYAML
from fileverse.formats.zarr import BaseZARR

from statomix.core.contracts import ProcedureState, ProcedureStatus
from statomix.curation import apply_curation_schemas
from statomix.curation.categorical import CatMetaEditSchema, CatMetaReport
from statomix.curation.columns import (
    ColEditSchema,
    ColReport,
    DatatypeInventory,
    DataTypes,
)
from statomix.curation.columns.audit import (
    DEFAULT_VALUE_COUNT_UNIQUE_THRESHOLD,
)
from statomix.curation.survival import SurvivalDataTypes
from statomix.curation.survival.report import (
    SurvCatMetaEditSchema,
    SurvEditSchema,
    SurvMetaReport,
    SurvPairs,
)
from statomix.logging import get_logger
from statomix.storage.atomic import atomic_output_path
from statomix.storage.parquet_metadata import (
    write_dataframe_with_category_ranks,
)
from statomix.storage.versioning import BasePipeline

logger = get_logger(name="cleaner")


def calculate_file_sha256(path: Path) -> str:
    """Calculate an artifact hash without modifying the file."""

    digest = sha256()

    with path.open("rb") as file_handle:
        while True:
            chunk = file_handle.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


class Cleaner(BasePipeline):
    """Create versioned reports, edit schemas, and curated data."""

    def __init__(self, df_path: Path, root_group: Any, dataset_name: str) -> None:
        super().__init__(
            root_group=root_group, dataset_name=dataset_name, pipeline_name="cleaner"
        )

        self.df_path = df_path

        self.col_report = ColReport()
        self.cat_meta_report = CatMetaReport()
        self.surv_meta_report = SurvMetaReport()

    def inherit_curated_state(
        self,
        *,
        source_cleaner: Cleaner,
        source_version: int,
        source_config_version: int,
        target_version: int | None = None,
        target_config_version: int | None = None,
        column_mapping: Mapping[str, str] | None = None,
        changed_columns: Collection[str],
        row_key: str | None = None,
        strict: bool = True,
        apply_parent_category_edits: bool = True,
        replace: bool = False,
    ) -> dict[str, Any]:
        """Inherit semantic decisions from an already-curated parent dataset.

        The target source dataframe is treated as post-curation data. Existing
        rename, removal, and category transformations are therefore represented
        by identity schemas instead of being applied a second time. Column and
        survival profiles are rebuilt from the target values while curated
        datatypes and endpoint labels are inherited from the parent.

        ``column_mapping`` maps parent curated column names to target names.
        ``changed_columns`` uses target names and makes every intentional value
        change explicit. In strict mode, every undeclared target column must be
        identical to its parent counterpart. Parent category recoding is
        applied only to changed columns unless explicitly disabled.
        """

        from statomix.pipelines.cleaner.inheritance import (
            inherit_curated_state,
        )

        result = inherit_curated_state(
            target_cleaner=self,
            source_cleaner=source_cleaner,
            source_version=source_version,
            source_config_version=source_config_version,
            target_version=target_version,
            target_config_version=target_config_version,
            column_mapping=column_mapping,
            changed_columns=changed_columns,
            row_key=row_key,
            strict=strict,
            apply_parent_category_edits=apply_parent_category_edits,
            replace=replace,
        )
        logger.info(
            "Inherited curated state from dataset '%s' version:%s "
            "config_version:%s into dataset '%s' version:%s "
            "config_version:%s.",
            source_cleaner.dataset_name,
            source_version,
            source_config_version,
            self.dataset_name,
            result["version"],
            result["config_version"],
        )
        return result

    def _get_default_version_meta(self) -> dict[str, Any]:
        return {}

    def _get_default_config_meta(self) -> dict[str, Any]:
        return {}

    def _get_curated_datatype_inventory(self, group_bundle):
        col_profiles_path = (
            group_bundle["version"]["path"] / "col_profiles_curated.parquet"
        )
        col_profiles = self.col_report.load_col_profiles(path=col_profiles_path)
        inventory = DatatypeInventory.from_profiles(profiles=col_profiles)
        return col_profiles, inventory

    @staticmethod
    def _record_procedure_status(
        group_bundle,
        procedure,
        status,
        reason,
        input_count,
        output_count,
        inventory=None,
    ):
        config_info = group_bundle["config"]
        config_group = config_info["group"]
        config_meta = dict(config_group.attrs.get("meta", {}))

        procedure_status = dict(config_meta.get("procedure_status", {}))
        procedure_status[procedure] = ProcedureStatus(
            status=ProcedureState(status),
            reason=reason,
            input_count=int(input_count),
            output_count=int(output_count),
        ).to_dict()
        config_meta["procedure_status"] = procedure_status

        if inventory is not None:
            config_meta["curated_datatype_counts"] = inventory.counts_by_name()

        config_group.attrs["meta"] = config_meta
        config_info["meta"] = config_meta

    def create_col_report(
        self,
        version=None,
        version_name=None,
        config_version=None,
        config_name=None,
        create_new=False,
    ):
        """Create the integrated column report and audit artifacts."""

        group_bundle = self._get_group_bundle(
            version=version,
            version_name=version_name,
            version_create_new=create_new,
            config_version=config_version,
            config_name=config_name,
            config_version_create_new=False,
        )

        version_meta = group_bundle["version"]["meta"]
        resolved_version = version_meta["version"]

        config_meta = group_bundle["config"]["meta"]
        resolved_config_version = config_meta["version"]

        base_path = group_bundle["version"]["path"]

        col_report_path = base_path / "col_report.xlsx"
        col_profiles_path = base_path / "col_profiles.parquet"
        col_audit_path = base_path / "col_audit.parquet"
        col_value_counts_path = base_path / "col_value_counts.parquet"

        user_config_path = self.paths["user_config"] / (
            f"{self.dataset_name.replace(' ', '_')}"
            f"_col_report_version{resolved_version}"
            f"_config{resolved_config_version}.xlsx"
        )

        core_artifacts = {
            "column report": col_report_path,
            "column profiles": col_profiles_path,
            "column audit": col_audit_path,
            "column value counts": (col_value_counts_path),
        }

        artifact_exists = {
            artifact_name: artifact_path.exists()
            for artifact_name, artifact_path in core_artifacts.items()
        }

        if all(artifact_exists.values()) and not create_new:
            if not user_config_path.exists():
                shutil.copy2(
                    src=col_report_path,
                    dst=user_config_path,
                )

            logger.info(
                "Integrated column report already exists "
                "for version:%s and config_version:%s.",
                resolved_version,
                resolved_config_version,
            )
            return

        if any(artifact_exists.values()) and not create_new:
            present_artifacts = [
                artifact_name
                for artifact_name, exists in artifact_exists.items()
                if exists
            ]
            missing_artifacts = [
                artifact_name
                for artifact_name, exists in artifact_exists.items()
                if not exists
            ]

            raise RuntimeError(
                "Incomplete column-report artifact set for "
                f"version:{resolved_version} and "
                f"config_version:{resolved_config_version}.\n"
                f"Present: {present_artifacts}\n"
                f"Missing: {missing_artifacts}\n"
                "Create a new Cleaner version instead of "
                "repairing this version in place."
            )

        df = pd.read_parquet(self.df_path)

        self.col_report.create_col_profiles(
            df=df,
            path=col_profiles_path,
            replace=create_new,
        )

        source_sha256 = calculate_file_sha256(path=self.df_path)

        self.col_report.create_col_report(
            df=df,
            report_path=col_report_path,
            profiles_path=col_profiles_path,
            audit_profiles_path=col_audit_path,
            value_frequencies_path=(col_value_counts_path),
            value_count_unique_threshold=(DEFAULT_VALUE_COUNT_UNIQUE_THRESHOLD),
            report_metadata={
                "dataset_name": self.dataset_name,
                "cleaner_version": resolved_version,
                "cleaner_config_version": (resolved_config_version),
                "source_row_count": len(df),
                "source_column_count": len(df.columns),
                "source_path": str(self.df_path),
                "source_sha256": source_sha256,
            },
            replace=create_new,
        )

        shutil.copy2(
            src=col_report_path,
            dst=user_config_path,
        )

        version_meta["col_report_exists"] = True
        version_meta["col_audit_exists"] = True
        version_meta["col_value_counts_exists"] = True
        version_meta["value_count_unique_threshold"] = (
            DEFAULT_VALUE_COUNT_UNIQUE_THRESHOLD
        )

        group_bundle["version"]["group"].attrs["meta"] = version_meta

        logger.info(
            "Created integrated column report for " "version:%s and config_version:%s.",
            resolved_version,
            resolved_config_version,
        )

    def create_col_edit_schema(self, version=None):

        group_bundle = self._get_group_bundle(version=version, config_version=None)
        version_meta = group_bundle["version"]["meta"]
        version = version_meta["version"]

        config_meta = group_bundle["config"]["meta"]
        config_version = config_meta["version"]

        base_path = group_bundle["version"]["path"]

        curated_col_report_path = base_path / "col_report_curated.xlsx"
        user_config_path = (
            self.paths["user_config"]
            / f"{self.dataset_name.replace(" ", "_")}_col_report_version{version}_config{config_version}_curated.xlsx"
        )

        if curated_col_report_path.exists():
            pass
        elif user_config_path.exists():
            shutil.copy2(src=user_config_path, dst=curated_col_report_path)
        else:
            error_msg = (
                f"Column Edit Report Curated file not found.\n"
                f"Checked:\n- {curated_col_report_path}\n- {user_config_path}"
            )
            raise FileNotFoundError(error_msg)

        col_profiles_path = base_path / "col_profiles.parquet"
        rename_mapping_path = base_path / "rename_mapping.yaml"
        col_edit_schema_path = base_path / "col_edit_schema.parquet"
        col_profiles_curated_path = base_path / "col_profiles_curated.parquet"

        if (
            rename_mapping_path.exists()
            and col_edit_schema_path.exists()
            and col_profiles_curated_path.exists()
        ):
            version = version_meta["version"]
            logger.info(f"Column edit schema already exists for version:{version}.")
            return

        curated_col_report = pd.ExcelFile(curated_col_report_path)
        rename_mapping, col_edit_schema = self.col_report.get_col_edit_schema(
            curated_col_report=curated_col_report
        )

        BaseYAML.save(data=rename_mapping, path=rename_mapping_path)
        col_edit_schema.save(path=col_edit_schema_path)

        col_profiles = self.col_report.load_col_profiles(path=col_profiles_path)
        col_profiles_curated = self.col_report.get_curated_col_profiles(
            col_profiles=col_profiles, col_edit_schema=col_edit_schema
        )
        self.col_report.save_col_profiles(
            col_profiles=col_profiles_curated, path=col_profiles_curated_path
        )

        version_meta["col_edit_schema_exists"] = True
        group_bundle["version"]["group"].attrs["meta"] = version_meta

    def create_cat_meta_report(
        self, version=None, config_version=None, config_name=None, create_new=False
    ):

        group_bundle = self._get_group_bundle(
            version=version,
            version_name=None,
            version_create_new=False,
            config_name=config_name,
            config_version=config_version,
            config_version_create_new=create_new,
        )

        version_meta = group_bundle["version"]["meta"]
        version = version_meta["version"]

        config_meta = group_bundle["config"]["meta"]
        config_version = config_meta["version"]

        base_path = group_bundle["config"]["path"]
        req_base_path = group_bundle["version"]["path"]

        col_profiles_curated, inventory = self._get_curated_datatype_inventory(
            group_bundle=group_bundle
        )
        categorical_count = inventory.count(datatype=DataTypes.CATEGORICAL)

        if categorical_count == 0:
            self._record_procedure_status(
                group_bundle=group_bundle,
                procedure="categorical_meta_report",
                status="not_applicable",
                reason="no_curated_categorical_columns",
                input_count=0,
                output_count=0,
                inventory=inventory,
            )
            logger.info(
                "No categorical columns found. "
                "Categorical metadata report is not applicable."
            )
            return

        rename_mapping_path = req_base_path / "rename_mapping.yaml"
        rename_mapping = BaseYAML.load(path=rename_mapping_path)

        meta_report_path = base_path / "cat_meta_report.xlsx"

        if meta_report_path.exists():
            self._record_procedure_status(
                group_bundle=group_bundle,
                procedure="categorical_meta_report",
                status="completed",
                reason="existing_report_reused",
                input_count=categorical_count,
                output_count=1,
                inventory=inventory,
            )
            version = version_meta["version"]
            config_version = config_meta["version"]
            logger.info(
                f"Categorical metadata report already exists for version:{version} and config_version:{config_version}"
            )
            return

        df = pd.read_parquet(self.df_path)

        self.cat_meta_report.create_meta_report(
            df=df,
            col_profiles=col_profiles_curated,
            rename_mapping=rename_mapping,
            report_path=meta_report_path,
        )
        user_config_path = (
            self.paths["user_config"]
            / f"{self.dataset_name.replace(" ", "_")}_cat_meta_report_version{version}_config{config_version}.xlsx"
        )
        shutil.copy2(src=meta_report_path, dst=user_config_path)

        config_meta["cat_meta_report_exists"] = True
        group_bundle["config"]["group"].attrs["meta"] = config_meta
        self._record_procedure_status(
            group_bundle=group_bundle,
            procedure="categorical_meta_report",
            status="completed",
            reason="report_created",
            input_count=categorical_count,
            output_count=1,
            inventory=inventory,
        )

    def create_cat_meta_edit_schema(self, version=None, config_version=None):
        group_bundle = self._get_group_bundle(
            version=version, config_version=config_version
        )

        version_meta = group_bundle["version"]["meta"]
        version = version_meta["version"]

        config_meta = group_bundle["config"]["meta"]
        config_version = config_meta["version"]

        base_path = group_bundle["config"]["path"]

        meta_edit_schema_path = base_path / "cat_meta_edit_schema.parquet"
        _, inventory = self._get_curated_datatype_inventory(group_bundle=group_bundle)
        categorical_count = inventory.count(datatype=DataTypes.CATEGORICAL)

        if categorical_count == 0:
            CatMetaEditSchema.empty().save(path=meta_edit_schema_path)
            self._record_procedure_status(
                group_bundle=group_bundle,
                procedure="categorical_meta_edit_schema",
                status="not_applicable",
                reason="no_curated_categorical_columns",
                input_count=0,
                output_count=0,
                inventory=inventory,
            )
            logger.info(
                "No categorical columns found. "
                "Created an empty categorical edit schema."
            )
            return

        if meta_edit_schema_path.exists():
            existing_schema = CatMetaEditSchema.load(path=meta_edit_schema_path)
            self._record_procedure_status(
                group_bundle=group_bundle,
                procedure="categorical_meta_edit_schema",
                status="completed",
                reason="existing_schema_reused",
                input_count=categorical_count,
                output_count=existing_schema.edit_count,
                inventory=inventory,
            )
            version = version_meta["version"]
            config_version = config_meta["version"]
            logger.info(
                f"Categorical metadata edit schema already exists for version: {version} and config_version:{config_version}"
            )
            return

        curated_meta_report_path = base_path / "cat_meta_report_curated.xlsx"
        user_config_path = (
            self.paths["user_config"]
            / f"{self.dataset_name.replace(" ", "_")}_cat_meta_report_version{version}_config{config_version}_curated.xlsx"
        )

        if curated_meta_report_path.exists():
            pass
        elif user_config_path.exists():
            shutil.copy2(src=user_config_path, dst=curated_meta_report_path)
        else:
            error_msg = (
                f"Categorical Metadata Edit Curated file not found.\n"
                f"Checked:\n- {curated_meta_report_path}\n- {user_config_path}"
            )
            raise FileNotFoundError(error_msg)

        curated_meta_report = pd.ExcelFile(curated_meta_report_path)

        meta_edit_schema = self.cat_meta_report.get_meta_edit_schema(
            curated_meta_report
        )
        meta_edit_schema.save(path=meta_edit_schema_path)
        self._record_procedure_status(
            group_bundle=group_bundle,
            procedure="categorical_meta_edit_schema",
            status="completed",
            reason="curated_schema_created",
            input_count=categorical_count,
            output_count=meta_edit_schema.edit_count,
            inventory=inventory,
        )

    def create_surv_meta_report(
        self, version=None, config_version=None, config_name=None, create_new=False
    ):

        group_bundle = self._get_group_bundle(
            version=version,
            version_name=None,
            version_create_new=False,
            config_name=config_name,
            config_version=config_version,
            config_version_create_new=create_new,
        )

        version_meta = group_bundle["version"]["meta"]
        version = version_meta["version"]

        config_meta = group_bundle["config"]["meta"]
        config_version = config_meta["version"]

        base_path = group_bundle["config"]["path"]

        surv_profiles_path = base_path / "surv_profiles.parquet"
        meta_report_path = base_path / "surv_meta_report.xlsx"

        _, inventory = self._get_curated_datatype_inventory(group_bundle=group_bundle)
        survival_count = inventory.count(datatype=DataTypes.SURVIVAL)

        if survival_count == 0:
            self.surv_meta_report.save_semantic_profiles(
                semantic_profiles={},
                path=surv_profiles_path,
            )
            self._record_procedure_status(
                group_bundle=group_bundle,
                procedure="survival_meta_report",
                status="not_applicable",
                reason="no_curated_survival_columns",
                input_count=0,
                output_count=0,
                inventory=inventory,
            )
            logger.info(
                "No survival columns found. "
                "Survival metadata report is not applicable."
            )
            return

        if surv_profiles_path.exists() and meta_report_path.exists():
            self._record_procedure_status(
                group_bundle=group_bundle,
                procedure="survival_meta_report",
                status="completed",
                reason="existing_report_reused",
                input_count=survival_count,
                output_count=1,
                inventory=inventory,
            )
            version = version_meta["version"]
            config_version = config_meta["version"]
            logger.info(
                f"Survival metadata report already exists for version: {version} and config_version:{config_version}"
            )
            return

        col_names = inventory.columns(datatype=DataTypes.SURVIVAL)

        self.surv_meta_report.create_surv_report(
            col_names=col_names,
            report_path=meta_report_path,
            profiles_path=surv_profiles_path,
        )

        user_config_path = (
            self.paths["user_config"]
            / f"{self.dataset_name.replace(" ", "_")}_surv_meta_report_version{version}_config{config_version}.xlsx"
        )
        shutil.copy2(src=meta_report_path, dst=user_config_path)
        self._record_procedure_status(
            group_bundle=group_bundle,
            procedure="survival_meta_report",
            status="completed",
            reason="report_created",
            input_count=survival_count,
            output_count=1,
            inventory=inventory,
        )

    def create_surv_meta_edit_schema(self, version=None, config_version=None):

        group_bundle = self._get_group_bundle(
            version=version, config_version=config_version
        )

        version_meta = group_bundle["version"]["meta"]
        version = version_meta["version"]

        config_meta = group_bundle["config"]["meta"]
        config_version = config_meta["version"]

        base_path = group_bundle["config"]["path"]

        surv_pairs_path = base_path / "surv_pairs.parquet"
        surv_profiles_path = base_path / "surv_profiles.parquet"
        surv_profiles_curated_path = base_path / "surv_profiles_curated.parquet"

        meta_edit_schema_path = base_path / "surv_meta_edit_schema.parquet"
        curated_meta_report_path = base_path / "surv_meta_report_curated.xlsx"

        _, inventory = self._get_curated_datatype_inventory(group_bundle=group_bundle)
        survival_count = inventory.count(datatype=DataTypes.SURVIVAL)

        if survival_count == 0:
            self.surv_meta_report.save_semantic_profiles(
                semantic_profiles={},
                path=surv_profiles_path,
            )
            SurvEditSchema.empty().save(path=meta_edit_schema_path)
            self.surv_meta_report.save_semantic_profiles(
                semantic_profiles={},
                path=surv_profiles_curated_path,
            )
            SurvPairs.empty().save(path=surv_pairs_path)

            self._record_procedure_status(
                group_bundle=group_bundle,
                procedure="survival_meta_edit_schema",
                status="not_applicable",
                reason="no_curated_survival_columns",
                input_count=0,
                output_count=0,
                inventory=inventory,
            )
            logger.info(
                "No survival columns found. Created empty survival "
                "profiles, edit schema, and pairs."
            )
            return

        if (
            surv_pairs_path.exists()
            and surv_profiles_curated_path.exists()
            and meta_edit_schema_path.exists()
        ):
            existing_pairs = SurvPairs.load(path=surv_pairs_path)
            self._record_procedure_status(
                group_bundle=group_bundle,
                procedure="survival_meta_edit_schema",
                status="completed",
                reason="existing_schema_reused",
                input_count=survival_count,
                output_count=len(existing_pairs.pairs),
                inventory=inventory,
            )
            version = version_meta["version"]
            config_version = config_meta["version"]
            logger.info(
                f"Surival meta data already exists for version:{version} and config_version:{config_version}"
            )
            return

        user_config_path = (
            self.paths["user_config"]
            / f"{self.dataset_name.replace(" ", "_")}_surv_meta_report_version{version}_config{config_version}_curated.xlsx"
        )

        if curated_meta_report_path.exists():
            pass
        elif user_config_path.exists():
            shutil.copy2(src=user_config_path, dst=curated_meta_report_path)
        else:
            error_msg = (
                f"Survival Categorical Metadata Edit Curated file not found.\n"
                f"Checked:\n- {curated_meta_report_path}\n- {user_config_path}"
            )
            raise FileNotFoundError(error_msg)

        surv_profiles = self.surv_meta_report.load_semantic_profiles(
            path=surv_profiles_path
        )
        curated_meta_report = pd.ExcelFile(curated_meta_report_path)

        meta_edit_schema = self.surv_meta_report.get_surv_edit_schema(
            curated_meta_report=curated_meta_report
        )
        meta_edit_schema.save(path=meta_edit_schema_path)

        surv_profiles_curated = self.surv_meta_report.get_curated_surv_profiles(
            meta_edit_schema=meta_edit_schema, surv_profiles=surv_profiles
        )
        self.surv_meta_report.save_semantic_profiles(
            semantic_profiles=surv_profiles_curated, path=surv_profiles_curated_path
        )

        surv_meta_df = curated_meta_report.parse(sheet_name="SurvMeta")
        surv_pairs = self.surv_meta_report.get_surv_pairs(
            surv_meta_df=surv_meta_df, surv_profiles=surv_profiles_curated
        )
        surv_pairs.save(path=surv_pairs_path)
        self._record_procedure_status(
            group_bundle=group_bundle,
            procedure="survival_meta_edit_schema",
            status="completed",
            reason="curated_schema_created",
            input_count=survival_count,
            output_count=len(surv_pairs.pairs),
            inventory=inventory,
        )

    def create_surv_cat_meta_report(
        self, version=None, config_version=None, config_name=None, create_new=False
    ):

        group_bundle = self._get_group_bundle(
            version=version,
            version_name=None,
            version_create_new=False,
            config_name=config_name,
            config_version=config_version,
            config_version_create_new=create_new,
        )
        version_meta = group_bundle["version"]["meta"]
        version = version_meta["version"]

        config_meta = group_bundle["config"]["meta"]
        config_version = config_meta["version"]

        base_path = group_bundle["config"]["path"]
        req_base_path = group_bundle["version"]["path"]

        # version_meta = group_bundle['version']['meta']
        # config_meta = group_bundle['config']['meta']

        # base_path =  group_bundle['config']['path']
        # req_base_path = group_bundle['version']['path']

        profiles_path = base_path / "surv_profiles_curated.parquet"
        meta_report_path = base_path / "surv_cat_meta_report.xlsx"

        _, inventory = self._get_curated_datatype_inventory(group_bundle=group_bundle)
        survival_count = inventory.count(datatype=DataTypes.SURVIVAL)

        if survival_count == 0:
            self._record_procedure_status(
                group_bundle=group_bundle,
                procedure="survival_categorical_meta_report",
                status="not_applicable",
                reason="no_curated_survival_columns",
                input_count=0,
                output_count=0,
                inventory=inventory,
            )
            logger.info(
                "No curated survival columns found. Survival "
                "categorical metadata report is not applicable."
            )
            return

        survival_profiles = self.surv_meta_report.load_semantic_profiles(
            path=profiles_path
        )
        event_count = sum(
            profile.col_type == SurvivalDataTypes.EVENT
            for profile in survival_profiles.values()
        )

        if event_count == 0:
            self._record_procedure_status(
                group_bundle=group_bundle,
                procedure="survival_categorical_meta_report",
                status="not_applicable",
                reason="no_curated_survival_event_columns",
                input_count=0,
                output_count=0,
                inventory=inventory,
            )
            logger.info(
                "No curated survival event columns found. Survival "
                "categorical metadata report is not applicable."
            )
            return

        rename_mapping_path = req_base_path / "rename_mapping.yaml"
        # col_profiles_path = req_base_path/"col_profiles_curated.parquet"

        rename_mapping = BaseYAML.load(path=rename_mapping_path)

        if meta_report_path.exists():
            self._record_procedure_status(
                group_bundle=group_bundle,
                procedure="survival_categorical_meta_report",
                status="completed",
                reason="existing_report_reused",
                input_count=event_count,
                output_count=1,
                inventory=inventory,
            )
            version = version_meta["version"]
            config_version = config_meta["version"]
            logger.info(
                f"Survival categorical metadata report already exists for version: {version} and config_version:{config_version}"
            )
            return

        df = pd.read_parquet(self.df_path)
        self.surv_meta_report.create_cat_meta_report(
            df=df,
            rename_mapping=rename_mapping,
            profiles_path=profiles_path,
            report_path=meta_report_path,
        )

        user_config_path = (
            self.paths["user_config"]
            / f"{self.dataset_name.replace(" ", "_")}_surv_cat_meta_report_version{version}_config{config_version}.xlsx"
        )
        shutil.copy2(src=meta_report_path, dst=user_config_path)
        self._record_procedure_status(
            group_bundle=group_bundle,
            procedure="survival_categorical_meta_report",
            status="completed",
            reason="report_created",
            input_count=event_count,
            output_count=1,
            inventory=inventory,
        )

    def create_surv_cat_meta_edit_schema(self, version=None, config_version=None):
        group_bundle = self._get_group_bundle(
            version=version, config_version=config_version
        )

        # version_meta = group_bundle['version']['meta']
        # config_meta = group_bundle['config']['meta']

        version_meta = group_bundle["version"]["meta"]
        version = version_meta["version"]

        config_meta = group_bundle["config"]["meta"]
        config_version = config_meta["version"]

        base_path = group_bundle["config"]["path"]

        curated_meta_report_path = base_path / "surv_cat_meta_report_curated.xlsx"
        meta_edit_schema_path = base_path / "surv_cat_meta_edit_schema.parquet"

        profiles_path = base_path / "surv_profiles_curated.parquet"
        _, inventory = self._get_curated_datatype_inventory(group_bundle=group_bundle)
        survival_count = inventory.count(datatype=DataTypes.SURVIVAL)

        if survival_count == 0:
            SurvCatMetaEditSchema.empty().save(path=meta_edit_schema_path)
            self._record_procedure_status(
                group_bundle=group_bundle,
                procedure="survival_categorical_meta_edit_schema",
                status="not_applicable",
                reason="no_curated_survival_columns",
                input_count=0,
                output_count=0,
                inventory=inventory,
            )
            logger.info(
                "No curated survival columns found. Created an empty "
                "survival categorical edit schema."
            )
            return

        survival_profiles = self.surv_meta_report.load_semantic_profiles(
            path=profiles_path
        )
        event_count = sum(
            profile.col_type == SurvivalDataTypes.EVENT
            for profile in survival_profiles.values()
        )

        if event_count == 0:
            SurvCatMetaEditSchema.empty().save(path=meta_edit_schema_path)
            self._record_procedure_status(
                group_bundle=group_bundle,
                procedure="survival_categorical_meta_edit_schema",
                status="not_applicable",
                reason="no_curated_survival_event_columns",
                input_count=0,
                output_count=0,
                inventory=inventory,
            )
            logger.info(
                "No curated survival event columns found. Created an "
                "empty survival categorical edit schema."
            )
            return

        if meta_edit_schema_path.exists():
            existing_schema = SurvCatMetaEditSchema.load(path=meta_edit_schema_path)
            self._record_procedure_status(
                group_bundle=group_bundle,
                procedure="survival_categorical_meta_edit_schema",
                status="completed",
                reason="existing_schema_reused",
                input_count=event_count,
                output_count=existing_schema.edit_count,
                inventory=inventory,
            )
            version = version_meta["version"]
            config_version = config_meta["version"]
            logger.info(
                f"Survival categorical metadata edit schema already exists for version: {version} and config_version:{config_version}"
            )
            return

        user_config_path = (
            self.paths["user_config"]
            / f"{self.dataset_name.replace(" ", "_")}_surv_cat_meta_report_version{version}_config{config_version}_curated.xlsx"
        )

        if curated_meta_report_path.exists():
            pass
        elif user_config_path.exists():
            shutil.copy2(src=user_config_path, dst=curated_meta_report_path)
        else:
            error_msg = (
                f"Survival Categorical Metadata Edit Curated file not found.\n"
                f"Checked:\n- {curated_meta_report_path}\n- {user_config_path}"
            )
            raise FileNotFoundError(error_msg)

        curated_meta_report = pd.ExcelFile(curated_meta_report_path)

        meta_edit_schema = self.surv_meta_report.get_surv_cat_meta_edit_schema(
            curated_meta_report=curated_meta_report
        )
        meta_edit_schema.save(path=meta_edit_schema_path)
        self._record_procedure_status(
            group_bundle=group_bundle,
            procedure="survival_categorical_meta_edit_schema",
            status="completed",
            reason="curated_schema_created",
            input_count=event_count,
            output_count=meta_edit_schema.edit_count,
            inventory=inventory,
        )

    def create_curated_data(self, version=None, config_version=None):

        group_bundle = self._get_group_bundle(
            version=version, config_version=config_version
        )

        version_meta = group_bundle["version"]["meta"]
        version = version_meta["version"]

        config_meta = group_bundle["config"]["meta"]
        config_version = config_meta["version"]

        base_path = group_bundle["config"]["path"]
        req_base_path = group_bundle["version"]["path"]

        config_group = group_bundle["config"]["group"]

        curated_data_group = config_group.require_group("curated_data")
        curated_base_path = BaseZARR.get_abs_path(curated_data_group)

        curated_df_path = curated_base_path / "df.parquet"
        curated_surv_pairs_path = curated_base_path / "surv_pairs.parquet"
        curated_col_profiles_path = curated_base_path / "col_profiles.parquet"

        col_profiles_curated, inventory = self._get_curated_datatype_inventory(
            group_bundle=group_bundle
        )

        if (
            curated_df_path.exists()
            and curated_surv_pairs_path.exists()
            and curated_col_profiles_path.exists()
        ):
            self._record_procedure_status(
                group_bundle=group_bundle,
                procedure="curated_data",
                status="completed",
                reason="existing_artifacts_reused",
                input_count=len(col_profiles_curated),
                output_count=3,
                inventory=inventory,
            )
            version = version_meta["version"]
            config_version = config_meta["version"]
            logger.info(
                f"Curated data already exists for version:{version} and config_version:{config_version}"
            )
            return

        # curated_data_meta = curated_data_group.attrs.get("meta", {})
        # curated_data_meta["curated_data_exists"] = False
        # curated_data_group.attrs["meta"] = curated_data_meta

        surv_pairs_path = base_path / "surv_pairs.parquet"
        rename_mapping_path = req_base_path / "rename_mapping.yaml"
        col_edit_schema_path = req_base_path / "col_edit_schema.parquet"
        cat_meta_edit_schema_path = base_path / "cat_meta_edit_schema.parquet"
        surv_cat_meta_edit_schema_path = base_path / "surv_cat_meta_edit_schema.parquet"

        rename_mapping = BaseYAML.load(path=rename_mapping_path)

        col_edit_schema = ColEditSchema.load(path=col_edit_schema_path)

        cat_meta_edit_schema = CatMetaEditSchema.load(cat_meta_edit_schema_path)

        surv_cat_meta_edit_schema = SurvCatMetaEditSchema.load(
            path=surv_cat_meta_edit_schema_path
        )

        surv_pairs = SurvPairs.load(path=surv_pairs_path)

        df = pd.read_parquet(path=self.df_path)
        df = apply_curation_schemas(
            df=df,
            rename_mapping=rename_mapping,
            col_edit_schema=col_edit_schema,
            cat_meta_edit_schema=cat_meta_edit_schema,
            surv_cat_meta_edit_schema=surv_cat_meta_edit_schema,
        )
        category_ranks = cat_meta_edit_schema.category_ranks

        with (
            atomic_output_path(destination=curated_df_path) as temporary_df,
            atomic_output_path(
                destination=curated_surv_pairs_path
            ) as temporary_surv_pairs,
            atomic_output_path(
                destination=curated_col_profiles_path
            ) as temporary_col_profiles,
        ):
            # df.to_parquet(path=temporary_df)
            rank_metadata = write_dataframe_with_category_ranks(
                df=df,
                path=temporary_df,
                category_ranks=category_ranks,
            )
            surv_pairs.save(path=temporary_surv_pairs)
            self.col_report.save_col_profiles(
                col_profiles=col_profiles_curated,
                path=temporary_col_profiles,
            )
        config_meta = dict(
            config_group.attrs.get(
                "meta",
                {},
            )
        )

        config_meta["categorical_rank_metadata"] = rank_metadata

        config_group.attrs["meta"] = config_meta
        group_bundle["config"]["meta"] = config_meta

        self._record_procedure_status(
            group_bundle=group_bundle,
            procedure="curated_data",
            status="completed",
            reason="artifacts_created",
            input_count=len(col_profiles_curated),
            output_count=3,
            inventory=inventory,
        )

        # curated_data_meta["curated_data_exists"] = True
        # curated_data_group.attrs["meta"] = curated_data_meta

    def get_curated_data_group(
        self,
        version=None,
        config_version=None,
    ):

        group_bundle = self._find_group_bundle(
            version=version,
            config_version=config_version,
        )

        version = group_bundle["version"]["meta"]["version"]
        config_version = group_bundle["config"]["meta"]["version"]
        config_group = group_bundle["config"]["group"]

        curated_data_group = config_group.get("curated_data")

        if curated_data_group is None:
            logger.info(
                f"Curated group does not exist for version:{version} "
                f"and config_version:{config_version}."
            )
            return None

        curated_base_path = BaseZARR.get_abs_path(curated_data_group)

        required_paths = (
            curated_base_path / "df.parquet",
            curated_base_path / "surv_pairs.parquet",
            curated_base_path / "col_profiles.parquet",
        )

        missing_paths = [path for path in required_paths if not path.exists()]

        if missing_paths:
            missing_names = ", ".join(path.name for path in missing_paths)

            logger.info(
                f"Curated data is incomplete for version:{version} "
                f"and config_version:{config_version}. "
                f"Missing: {missing_names}."
            )
            return None

        return curated_data_group

    # def get_curated_data_group(self, version=None, config_version=None):

    #     group_bundle = self._get_group_bundle(
    #         version=version, config_version=config_version
    #     )

    #     version_meta = group_bundle["version"]["meta"]
    #     version = version_meta["version"]

    #     config_meta = group_bundle["config"]["meta"]
    #     config_version = config_meta["version"]

    #     config_group = group_bundle["config"]["group"]

    #     curated_data_group = config_group.get("curated_data")

    #     if curated_data_group is None:
    #         logger.info(
    #             f"Curated group does not exist for version:{version} "
    #             f"and config_version:{config_version}"
    #         )
    #         return None

    #     curated_base_path = BaseZARR.get_abs_path(
    #         curated_data_group
    #     )

    #     required_paths = (
    #         curated_base_path / "df.parquet",
    #         curated_base_path / "surv_pairs.parquet",
    #         curated_base_path / "col_profiles.parquet",
    #     )

    #     all_outputs_exist = (
    #         curated_df_path.exists()
    #         and curated_surv_pairs_path.exists()
    #         and curated_col_profiles_path.exists()
    #     )

    #     if not curated_data_meta["curated_data_exists"]:
    #         version = version_meta["version"]
    #         config_version = config_meta["version"]
    #         logger.info(
    #             f"Curated group does not exist for version:{version} and config_version:{config_version}"
    #         )
    #         return None

    #     return curated_data_group
