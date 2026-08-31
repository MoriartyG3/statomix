"""Persistence adapter for Cleaner curated-parent inheritance."""

from __future__ import annotations

import shutil
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from fileverse.formats.yaml import BaseYAML

from statomix.core.contracts import (
    CuratedStateLineage,
    ProcedureState,
    ProcedureStatus,
)
from statomix.core.errors import CuratedStateInheritanceError
from statomix.curation.categorical import CatMetaEditSchema
from statomix.curation.columns import ColEditSchema, DatatypeInventory, DataTypes
from statomix.curation.inheritance import (
    InheritedCuratedState,
    apply_inherited_category_edits,
    build_inherited_curated_state,
)
from statomix.curation.survival import (
    SurvCatMetaEditSchema,
    SurvEditSchema,
    SurvivalDataTypes,
    SurvPairs,
)
from statomix.storage.atomic import atomic_output_path
from statomix.storage.hashing import sha256_file
from statomix.storage.layout import StatomixLayout

if TYPE_CHECKING:
    from statomix.pipelines.cleaner.cleaner import Cleaner


@dataclass(frozen=True, slots=True, kw_only=True)
class _InheritancePaths:
    version_root: Path
    config_root: Path
    curated_root: Path
    user_config_root: Path
    dataset_stem: str
    version: int
    config_version: int

    @property
    def version_artifacts(self) -> dict[str, Path]:
        return {
            "col_profiles": self.version_root / "col_profiles.parquet",
            "col_profiles_curated": (
                self.version_root / "col_profiles_curated.parquet"
            ),
            "rename_mapping": self.version_root / "rename_mapping.yaml",
            "col_edit_schema": self.version_root / "col_edit_schema.parquet",
            "col_report": self.version_root / "col_report.xlsx",
            "col_report_curated": self.version_root / "col_report_curated.xlsx",
        }

    @property
    def config_artifacts(self) -> dict[str, Path]:
        return {
            "cat_meta_edit_schema": (self.config_root / "cat_meta_edit_schema.parquet"),
            "surv_profiles": self.config_root / "surv_profiles.parquet",
            "surv_profiles_curated": (
                self.config_root / "surv_profiles_curated.parquet"
            ),
            "surv_meta_edit_schema": (
                self.config_root / "surv_meta_edit_schema.parquet"
            ),
            "surv_pairs": self.config_root / "surv_pairs.parquet",
            "surv_cat_meta_edit_schema": (
                self.config_root / "surv_cat_meta_edit_schema.parquet"
            ),
            "cat_meta_report": self.config_root / "cat_meta_report.xlsx",
            "cat_meta_report_curated": (
                self.config_root / "cat_meta_report_curated.xlsx"
            ),
            "surv_meta_report": self.config_root / "surv_meta_report.xlsx",
            "surv_meta_report_curated": (
                self.config_root / "surv_meta_report_curated.xlsx"
            ),
            "surv_cat_meta_report": (self.config_root / "surv_cat_meta_report.xlsx"),
            "surv_cat_meta_report_curated": (
                self.config_root / "surv_cat_meta_report_curated.xlsx"
            ),
        }

    @property
    def curated_artifacts(self) -> dict[str, Path]:
        return {
            "df": self.curated_root / StatomixLayout.CURATED_DF,
            "surv_pairs": (self.curated_root / StatomixLayout.CURATED_SURV_PAIRS),
            "col_profiles": (self.curated_root / StatomixLayout.CURATED_COL_PROFILES),
        }

    def user_report(self, *, report_name: str, curated: bool) -> Path:
        suffix = "_curated" if curated else ""
        filename = (
            f"{self.dataset_stem}_{report_name}_version{self.version}_"
            f"config{self.config_version}{suffix}.xlsx"
        )
        return self.user_config_root / filename

    def managed_artifacts(self) -> tuple[Path, ...]:
        paths = [
            *self.version_artifacts.values(),
            *self.config_artifacts.values(),
            *self.curated_artifacts.values(),
        ]
        for report_name in (
            "col_report",
            "cat_meta_report",
            "surv_meta_report",
            "surv_cat_meta_report",
        ):
            paths.extend(
                [
                    self.user_report(report_name=report_name, curated=False),
                    self.user_report(report_name=report_name, curated=True),
                ]
            )
        return tuple(paths)


def _save_atomic(*, destination: Path, writer: Callable[[Path], None]) -> None:
    with atomic_output_path(destination=destination) as temporary_path:
        writer(temporary_path)


def _copy_atomic(*, source: Path, destination: Path) -> None:
    _save_atomic(
        destination=destination,
        writer=lambda temporary_path: shutil.copy2(
            src=source,
            dst=temporary_path,
        ),
    )


def _ensure_replace_allowed(*, paths: _InheritancePaths, replace: bool) -> None:
    existing = sorted(
        (path for path in paths.managed_artifacts() if path.exists()),
        key=str,
    )
    if existing and not replace:
        preview = "\n".join(f"- {path}" for path in existing[:12])
        remainder = len(existing) - 12
        if remainder:
            preview += f"\n- ... and {remainder} more"
        raise FileExistsError(
            "Target cleaner artifacts already exist. Curated-state "
            "inheritance will not overwrite them implicitly. Pass "
            f"replace=True after reviewing the target paths:\n{preview}"
        )


def _write_core_artifacts(
    *,
    cleaner: Cleaner,
    paths: _InheritancePaths,
    target_df: pd.DataFrame,
    state: InheritedCuratedState,
) -> None:
    version = paths.version_artifacts
    config = paths.config_artifacts
    curated = paths.curated_artifacts

    for destination in (
        version["col_profiles"],
        version["col_profiles_curated"],
        curated["col_profiles"],
    ):
        _save_atomic(
            destination=destination,
            writer=lambda temporary_path, profiles=state.col_profiles: (
                cleaner.col_report.save_col_profiles(
                    col_profiles=profiles,
                    path=temporary_path,
                )
            ),
        )

    _save_atomic(
        destination=version["rename_mapping"],
        writer=lambda temporary_path: BaseYAML.save(
            data={},
            path=temporary_path,
            replace=True,
        ),
    )
    _save_atomic(
        destination=version["col_edit_schema"],
        writer=lambda temporary_path: ColEditSchema.empty().save(path=temporary_path),
    )
    _save_atomic(
        destination=config["cat_meta_edit_schema"],
        writer=lambda temporary_path: CatMetaEditSchema.empty().save(
            path=temporary_path
        ),
    )

    for destination in (
        config["surv_profiles"],
        config["surv_profiles_curated"],
    ):
        _save_atomic(
            destination=destination,
            writer=lambda temporary_path, profiles=state.survival_profiles: (
                cleaner.surv_meta_report.save_semantic_profiles(
                    semantic_profiles=profiles,
                    path=temporary_path,
                )
            ),
        )

    _save_atomic(
        destination=config["surv_meta_edit_schema"],
        writer=lambda temporary_path: SurvEditSchema.empty().save(path=temporary_path),
    )
    for destination in (config["surv_pairs"], curated["surv_pairs"]):
        _save_atomic(
            destination=destination,
            writer=lambda temporary_path, pairs=state.survival_pairs: pairs.save(
                path=temporary_path
            ),
        )

    _save_atomic(
        destination=config["surv_cat_meta_edit_schema"],
        writer=lambda temporary_path: SurvCatMetaEditSchema.empty().save(
            path=temporary_path
        ),
    )
    _save_atomic(
        destination=curated["df"],
        writer=lambda temporary_path: target_df.to_parquet(
            path=temporary_path,
            index=False,
        ),
    )


def _write_report_pair(
    *,
    source: Path,
    curated_destination: Path,
    user_default: Path,
    user_curated: Path,
) -> None:
    _copy_atomic(source=source, destination=curated_destination)
    _copy_atomic(source=source, destination=user_default)
    _copy_atomic(source=source, destination=user_curated)


def _remove_report_pair(
    *,
    paths: _InheritancePaths,
    report_name: str,
    default_path: Path,
    curated_path: Path,
) -> None:
    for path in (
        default_path,
        curated_path,
        paths.user_report(report_name=report_name, curated=False),
        paths.user_report(report_name=report_name, curated=True),
    ):
        path.unlink(missing_ok=True)


def _write_reports(
    *,
    cleaner: Cleaner,
    paths: _InheritancePaths,
    target_df: pd.DataFrame,
    state: InheritedCuratedState,
) -> None:
    version = paths.version_artifacts
    config = paths.config_artifacts

    _save_atomic(
        destination=version["col_report"],
        writer=lambda temporary_path: cleaner.col_report.create_col_report(
            df=target_df,
            report_path=temporary_path,
            profiles_path=version["col_profiles"],
            replace=True,
            rename_mapping={},
        ),
    )
    _write_report_pair(
        source=version["col_report"],
        curated_destination=version["col_report_curated"],
        user_default=paths.user_report(
            report_name="col_report",
            curated=False,
        ),
        user_curated=paths.user_report(
            report_name="col_report",
            curated=True,
        ),
    )

    categorical_columns = [
        name
        for name, profile in state.col_profiles.items()
        if profile.col_type == DataTypes.CATEGORICAL
    ]
    if categorical_columns:
        _save_atomic(
            destination=config["cat_meta_report"],
            writer=lambda temporary_path: cleaner.cat_meta_report.create_meta_report(
                df=target_df,
                col_profiles=state.col_profiles,
                rename_mapping={},
                report_path=temporary_path,
            ),
        )
        _write_report_pair(
            source=config["cat_meta_report"],
            curated_destination=config["cat_meta_report_curated"],
            user_default=paths.user_report(
                report_name="cat_meta_report",
                curated=False,
            ),
            user_curated=paths.user_report(
                report_name="cat_meta_report",
                curated=True,
            ),
        )
    else:
        _remove_report_pair(
            paths=paths,
            report_name="cat_meta_report",
            default_path=config["cat_meta_report"],
            curated_path=config["cat_meta_report_curated"],
        )

    if state.survival_profiles:
        _save_atomic(
            destination=config["surv_meta_report"],
            writer=lambda temporary_path: (
                cleaner.surv_meta_report.create_surv_report_from_profiles(
                    col_names=list(state.survival_profiles),
                    semantic_profiles=state.survival_profiles,
                    report_path=temporary_path,
                    survival_labels=state.survival_labels,
                )
            ),
        )
        _write_report_pair(
            source=config["surv_meta_report"],
            curated_destination=config["surv_meta_report_curated"],
            user_default=paths.user_report(
                report_name="surv_meta_report",
                curated=False,
            ),
            user_curated=paths.user_report(
                report_name="surv_meta_report",
                curated=True,
            ),
        )
    else:
        _remove_report_pair(
            paths=paths,
            report_name="surv_meta_report",
            default_path=config["surv_meta_report"],
            curated_path=config["surv_meta_report_curated"],
        )

    event_columns = [
        name
        for name, profile in state.survival_profiles.items()
        if profile.col_type == SurvivalDataTypes.EVENT
    ]
    if event_columns:
        _save_atomic(
            destination=config["surv_cat_meta_report"],
            writer=lambda temporary_path: (
                cleaner.surv_meta_report.create_cat_meta_report(
                    df=target_df,
                    rename_mapping={},
                    profiles_path=config["surv_profiles_curated"],
                    report_path=temporary_path,
                )
            ),
        )
        _write_report_pair(
            source=config["surv_cat_meta_report"],
            curated_destination=config["surv_cat_meta_report_curated"],
            user_default=paths.user_report(
                report_name="surv_cat_meta_report",
                curated=False,
            ),
            user_curated=paths.user_report(
                report_name="surv_cat_meta_report",
                curated=True,
            ),
        )
    else:
        _remove_report_pair(
            paths=paths,
            report_name="surv_cat_meta_report",
            default_path=config["surv_cat_meta_report"],
            curated_path=config["surv_cat_meta_report_curated"],
        )


def _procedure_status(
    *,
    status: ProcedureState,
    reason: str,
    input_count: int,
    output_count: int,
) -> dict[str, str | int]:
    return ProcedureStatus(
        status=status,
        reason=reason,
        input_count=input_count,
        output_count=output_count,
    ).to_dict()


def _record_metadata(
    *,
    group_bundle: dict[str, dict[str, Any]],
    curated_data_group: Any,
    paths: _InheritancePaths,
    state: InheritedCuratedState,
    lineage: CuratedStateLineage,
) -> None:
    inventory = DatatypeInventory.from_profiles(profiles=state.col_profiles)
    categorical_count = inventory.count(datatype=DataTypes.CATEGORICAL)
    survival_count = inventory.count(datatype=DataTypes.SURVIVAL)
    event_count = sum(
        profile.col_type == SurvivalDataTypes.EVENT
        for profile in state.survival_profiles.values()
    )

    version_group = group_bundle["version"]["group"]
    version_meta = dict(version_group.attrs.get("meta", {}))
    version_meta["col_report_exists"] = True
    version_meta["col_edit_schema_exists"] = True
    version_group.attrs["meta"] = version_meta

    config_group = group_bundle["config"]["group"]
    config_meta = dict(config_group.attrs.get("meta", {}))
    procedure_status = dict(config_meta.get("procedure_status", {}))
    procedure_status["curated_state_inheritance"] = _procedure_status(
        status=ProcedureState.COMPLETED,
        reason="curated_parent_state_inherited",
        input_count=len(state.col_profiles),
        output_count=3,
    )
    procedure_status["categorical_meta_report"] = _procedure_status(
        status=(
            ProcedureState.COMPLETED
            if categorical_count
            else ProcedureState.NOT_APPLICABLE
        ),
        reason=(
            "derived_state_report_created"
            if categorical_count
            else "no_curated_categorical_columns"
        ),
        input_count=categorical_count,
        output_count=int(bool(categorical_count)),
    )
    procedure_status["categorical_meta_edit_schema"] = _procedure_status(
        status=(
            ProcedureState.COMPLETED
            if categorical_count
            else ProcedureState.NOT_APPLICABLE
        ),
        reason="inherited_curated_identity_schema",
        input_count=categorical_count,
        output_count=0,
    )
    procedure_status["survival_meta_report"] = _procedure_status(
        status=(
            ProcedureState.COMPLETED
            if survival_count
            else ProcedureState.NOT_APPLICABLE
        ),
        reason=(
            "derived_state_report_created"
            if survival_count
            else "no_curated_survival_columns"
        ),
        input_count=survival_count,
        output_count=int(bool(survival_count)),
    )
    procedure_status["survival_meta_edit_schema"] = _procedure_status(
        status=(
            ProcedureState.COMPLETED
            if survival_count
            else ProcedureState.NOT_APPLICABLE
        ),
        reason="inherited_curated_identity_schema",
        input_count=survival_count,
        output_count=len(state.survival_pairs.pairs),
    )
    procedure_status["survival_categorical_meta_report"] = _procedure_status(
        status=(
            ProcedureState.COMPLETED if event_count else ProcedureState.NOT_APPLICABLE
        ),
        reason=(
            "derived_state_report_created"
            if event_count
            else "no_curated_survival_event_columns"
        ),
        input_count=event_count,
        output_count=int(bool(event_count)),
    )
    procedure_status["survival_categorical_meta_edit_schema"] = _procedure_status(
        status=(
            ProcedureState.COMPLETED if event_count else ProcedureState.NOT_APPLICABLE
        ),
        reason="inherited_curated_identity_schema",
        input_count=event_count,
        output_count=0,
    )
    procedure_status["curated_data"] = _procedure_status(
        status=ProcedureState.COMPLETED,
        reason="derived_curated_artifacts_created",
        input_count=len(state.col_profiles),
        output_count=3,
    )

    curated_hashes = {
        name: sha256_file(path=path) for name, path in paths.curated_artifacts.items()
    }
    config_meta.update(
        {
            "cat_meta_report_exists": bool(categorical_count),
            "curated_artifact_sha256": curated_hashes,
            "curated_datatype_counts": inventory.counts_by_name(),
            "curated_parent": lineage.to_dict(),
            "curation_mode": "inherited_curated_state",
            "procedure_status": procedure_status,
            "surv_meta_report_exists": bool(survival_count),
        }
    )
    config_group.attrs["meta"] = config_meta

    curated_data_group.attrs["meta"] = {
        "artifact_sha256": curated_hashes,
        "curated_data_exists": True,
        "curation_mode": "inherited_curated_state",
        "parent": lineage.to_dict()["source"],
    }


def inherit_curated_state(
    *,
    target_cleaner: Cleaner,
    source_cleaner: Cleaner,
    source_version: int,
    source_config_version: int,
    target_version: int | None,
    target_config_version: int | None,
    column_mapping: Mapping[str, str] | None,
    changed_columns: Collection[str],
    row_key: str | None,
    strict: bool,
    apply_parent_category_edits: bool,
    replace: bool,
) -> dict[str, Any]:
    """Materialize a target Cleaner state from a curated parent dataset."""

    if source_cleaner is target_cleaner:
        raise ValueError("source_cleaner and target cleaner must be different")
    if not isinstance(apply_parent_category_edits, bool):
        raise CuratedStateInheritanceError(
            "apply_parent_category_edits must be a boolean value."
        )
    if not isinstance(replace, bool):
        raise CuratedStateInheritanceError("replace must be a boolean value.")
    if isinstance(changed_columns, (str, bytes)):
        raise CuratedStateInheritanceError(
            "changed_columns must be a collection of complete column names, "
            "not one string."
        )
    changed_columns = tuple(changed_columns)

    source_bundle = source_cleaner._find_group_bundle(
        version=source_version,
        config_version=source_config_version,
    )
    source_curated_group = source_cleaner.get_curated_data_group(
        version=source_version,
        config_version=source_config_version,
    )
    if source_curated_group is None:
        raise FileNotFoundError(
            "The selected parent Cleaner configuration has no complete "
            "curated_data group."
        )

    source_curated_paths = StatomixLayout(
        root=source_bundle["config"]["path"]
    ).curated_artifacts()
    source_survival_profiles_path = (
        source_bundle["config"]["path"] / "surv_profiles_curated.parquet"
    )
    source_cat_schema_path = (
        source_bundle["config"]["path"] / "cat_meta_edit_schema.parquet"
    )
    source_surv_cat_schema_path = (
        source_bundle["config"]["path"] / "surv_cat_meta_edit_schema.parquet"
    )
    required_source_paths = {
        **source_curated_paths,
        "surv_profiles": source_survival_profiles_path,
    }
    missing_source = [
        path for path in required_source_paths.values() if not path.is_file()
    ]
    if missing_source:
        missing_text = "\n".join(f"- {path}" for path in missing_source)
        raise FileNotFoundError(
            "Parent curated state is incomplete. Missing artifacts:\n" f"{missing_text}"
        )

    source_df = pd.read_parquet(source_curated_paths["df"])
    target_df = pd.read_parquet(target_cleaner.df_path)
    source_col_profiles = source_cleaner.col_report.load_col_profiles(
        path=source_curated_paths["col_profiles"]
    )
    source_survival_profiles = source_cleaner.surv_meta_report.load_semantic_profiles(
        path=source_survival_profiles_path
    )
    parent_pairs = SurvPairs.load(path=source_curated_paths["surv_pairs"])
    source_cat_schema = (
        CatMetaEditSchema.load(path=source_cat_schema_path)
        if source_cat_schema_path.is_file()
        else CatMetaEditSchema.empty()
    )
    source_surv_cat_schema = (
        SurvCatMetaEditSchema.load(path=source_surv_cat_schema_path)
        if source_surv_cat_schema_path.is_file()
        else SurvCatMetaEditSchema.empty()
    )
    if apply_parent_category_edits:
        curated_target_df = apply_inherited_category_edits(
            target_df=target_df,
            source_cat_meta_edit_schema=source_cat_schema,
            source_surv_cat_meta_edit_schema=source_surv_cat_schema,
            column_mapping=column_mapping,
            changed_columns=changed_columns,
        )
    else:
        curated_target_df = target_df

    state = build_inherited_curated_state(
        source_df=source_df,
        target_df=curated_target_df,
        source_col_profiles=source_col_profiles,
        source_survival_profiles=source_survival_profiles,
        source_survival_pairs=parent_pairs,
        column_mapping=column_mapping,
        changed_columns=changed_columns,
        row_key=row_key,
        strict=strict,
    )

    target_bundle = target_cleaner._get_group_bundle(
        version=target_version,
        config_version=target_config_version,
    )
    resolved_target_version = int(target_bundle["version"]["meta"]["version"])
    resolved_target_config = int(target_bundle["config"]["meta"]["version"])
    if target_version is not None and resolved_target_version != int(target_version):
        raise RuntimeError(
            f"Requested target version {target_version}, resolved "
            f"{resolved_target_version}."
        )
    if target_config_version is not None and resolved_target_config != int(
        target_config_version
    ):
        raise RuntimeError(
            f"Requested target config {target_config_version}, resolved "
            f"{resolved_target_config}."
        )

    target_config_group = target_bundle["config"]["group"]
    target_curated_root = target_bundle["config"]["path"] / StatomixLayout.CURATED_GROUP
    paths = _InheritancePaths(
        version_root=target_bundle["version"]["path"],
        config_root=target_bundle["config"]["path"],
        curated_root=target_curated_root,
        user_config_root=target_cleaner.paths["user_config"],
        dataset_stem=target_cleaner.dataset_name.replace(" ", "_"),
        version=resolved_target_version,
        config_version=resolved_target_config,
    )
    _ensure_replace_allowed(paths=paths, replace=replace)

    target_curated_group = target_config_group.require_group(
        StatomixLayout.CURATED_GROUP
    )

    source_hashes = {
        name: sha256_file(path=path) for name, path in required_source_paths.items()
    }
    optional_source_paths = {
        "cat_meta_edit_schema": source_cat_schema_path,
        "surv_cat_meta_edit_schema": source_surv_cat_schema_path,
    }
    source_hashes.update(
        {
            name: sha256_file(path=path)
            for name, path in optional_source_paths.items()
            if path.is_file()
        }
    )
    lineage = CuratedStateLineage(
        source_project=source_cleaner.project_name,
        source_dataset=source_cleaner.dataset_name,
        source_version=int(source_version),
        source_config_version=int(source_config_version),
        target_dataset=target_cleaner.dataset_name,
        target_version=resolved_target_version,
        target_config_version=resolved_target_config,
        column_mapping=state.column_mapping,
        changed_columns=state.changed_columns,
        row_key=row_key,
        strict=strict,
        applied_parent_category_edits=apply_parent_category_edits,
        source_artifact_sha256=source_hashes,
        target_source_df_sha256=sha256_file(path=target_cleaner.df_path),
    )

    _write_core_artifacts(
        cleaner=target_cleaner,
        paths=paths,
        target_df=curated_target_df,
        state=state,
    )
    _write_reports(
        cleaner=target_cleaner,
        paths=paths,
        target_df=curated_target_df,
        state=state,
    )
    _record_metadata(
        group_bundle=target_bundle,
        curated_data_group=target_curated_group,
        paths=paths,
        state=state,
        lineage=lineage,
    )

    return {
        "version": resolved_target_version,
        "config_version": resolved_target_config,
        "curated_data": {
            name: str(path) for name, path in paths.curated_artifacts.items()
        },
        "lineage": lineage.to_dict(),
        "survival_pairs": list(state.survival_pairs.pairs),
    }
