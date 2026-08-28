from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest
from fileverse.formats.zarr import BaseZARR
from pandas.testing import assert_frame_equal

from statomix.core.errors import CuratedStateInheritanceError
from statomix.curation.categorical import CatEdit, CatMetaEditSchema
from statomix.curation.columns import ColProfiler, ColReport, DataTypes
from statomix.curation.inheritance import (
    apply_inherited_category_edits,
    build_inherited_curated_state,
)
from statomix.curation.survival import (
    SurvCatMetaEditSchema,
    SurvMetaReport,
    SurvPair,
    SurvPairs,
    get_survival_semantic_col_profile,
)
from statomix.curation.survival.report import SurvCatEdit
from statomix.storage.layout import StatomixLayout


def _source_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Patient ID": ["P1", "P2", "P3"],
            "Age": [51, 62, 47],
            "Group": ["A", "B", "A"],
            "OS Event": [0, 1, 1],
            "OS Months": [12.0, 6.0, 9.0],
        }
    )


def _target_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Patient ID": ["P3", "P1", "P2"],
            "Age": [47, 51, 62],
            "Group": ["A", "A", "B"],
            "OS Event": [0, 0, 1],
            "OS_months": [10.0, 13.0, float("nan")],
        }
    )


def _parent_semantics(source_df: pd.DataFrame):
    col_profiler = ColProfiler()
    col_profiles = {
        name: col_profiler.get_col_profile(
            col_name=name,
            col_series=source_df[name],
        )
        for name in source_df.columns
    }
    survival_profiles = {
        name: get_survival_semantic_col_profile(col_name=name)
        for name in ("OS Event", "OS Months")
    }
    survival_pairs = SurvPairs(
        pairs={
            "OS": SurvPair(
                surv_label="OS",
                event_profile=survival_profiles["OS Event"],
                time_profile=survival_profiles["OS Months"],
            )
        }
    )
    return col_profiles, survival_profiles, survival_pairs


def test_build_inherited_state_preserves_decisions_and_reprofiles_values():
    source_df = _source_df()
    target_df = _target_df()
    col_profiles, survival_profiles, survival_pairs = _parent_semantics(source_df)

    state = build_inherited_curated_state(
        source_df=source_df,
        target_df=target_df,
        source_col_profiles=col_profiles,
        source_survival_profiles=survival_profiles,
        source_survival_pairs=survival_pairs,
        column_mapping={"OS Months": "OS_months"},
        changed_columns=["OS Event", "OS_months"],
        row_key="Patient ID",
        strict=True,
    )

    assert state.column_mapping == {"OS Months": "OS_months"}
    assert state.col_profiles["OS_months"].missing_n == 1
    assert (
        state.col_profiles["OS_months"].col_type == col_profiles["OS Months"].col_type
    )
    assert state.survival_labels == {
        "OS Event": "OS",
        "OS_months": "OS",
    }
    assert state.survival_pairs.pairs["OS"].time_profile.col_name == "OS_months"


def test_build_inherited_state_rejects_undeclared_changes():
    source_df = _source_df()
    target_df = _target_df()
    target_df.loc[target_df["Patient ID"] == "P1", "Age"] = 99
    col_profiles, survival_profiles, survival_pairs = _parent_semantics(source_df)

    with pytest.raises(
        CuratedStateInheritanceError,
        match="Age.*not declared",
    ):
        build_inherited_curated_state(
            source_df=source_df,
            target_df=target_df,
            source_col_profiles=col_profiles,
            source_survival_profiles=survival_profiles,
            source_survival_pairs=survival_pairs,
            column_mapping={"OS Months": "OS_months"},
            changed_columns=["OS Event", "OS_months"],
            row_key="Patient ID",
            strict=True,
        )


def test_build_inherited_state_preserves_orphan_survival_datatype():
    source_df = _source_df().assign(**{"Alive/Dead": [0, 1, 0]})
    target_df = _target_df().assign(**{"Alive/Dead": [0, 0, 1]})
    col_profiles, survival_profiles, survival_pairs = _parent_semantics(source_df)
    col_profiles["Alive/Dead"] = replace(
        col_profiles["Alive/Dead"],
        col_type=DataTypes.SURVIVAL,
    )

    state = build_inherited_curated_state(
        source_df=source_df,
        target_df=target_df,
        source_col_profiles=col_profiles,
        source_survival_profiles=survival_profiles,
        source_survival_pairs=survival_pairs,
        column_mapping={"OS Months": "OS_months"},
        changed_columns=["OS Event", "OS_months"],
        row_key="Patient ID",
        strict=True,
    )

    assert state.col_profiles["Alive/Dead"].col_type == DataTypes.SURVIVAL
    assert "Alive/Dead" not in state.survival_profiles
    assert "Alive/Dead" not in state.survival_labels
    assert list(state.survival_pairs.pairs) == ["OS"]


def test_build_inherited_state_requires_profiles_for_paired_columns():
    source_df = _source_df()
    target_df = _target_df()
    col_profiles, survival_profiles, survival_pairs = _parent_semantics(source_df)
    survival_profiles.pop("OS Event")

    with pytest.raises(
        CuratedStateInheritanceError,
        match=r"Parent curated survival profiles are incomplete: \['OS Event'\]",
    ):
        build_inherited_curated_state(
            source_df=source_df,
            target_df=target_df,
            source_col_profiles=col_profiles,
            source_survival_profiles=survival_profiles,
            source_survival_pairs=survival_pairs,
            column_mapping={"OS Months": "OS_months"},
            changed_columns=["OS Event", "OS_months"],
            row_key="Patient ID",
            strict=True,
        )


def test_parent_category_edits_apply_only_to_changed_columns():
    target_df = pd.DataFrame(
        {
            "Group": ["A", "B"],
            "OS Event": ["alive", "dead"],
        }
    )
    category_schema = CatMetaEditSchema(
        cat_edits={
            "Group": {
                "A": CatEdit(
                    col_name="Group",
                    category="A",
                    rename_to="renamed",
                    remove=False,
                )
            }
        }
    )
    survival_schema = SurvCatMetaEditSchema(
        cat_edits={
            "OS Event": {
                "alive": SurvCatEdit(
                    col_name="OS Event",
                    category="alive",
                    rename_to="0",
                    remove=False,
                ),
                "dead": SurvCatEdit(
                    col_name="OS Event",
                    category="dead",
                    rename_to="1",
                    remove=False,
                ),
            }
        }
    )

    curated_df = apply_inherited_category_edits(
        target_df=target_df,
        source_cat_meta_edit_schema=category_schema,
        source_surv_cat_meta_edit_schema=survival_schema,
        column_mapping={},
        changed_columns=["OS Event"],
    )

    assert curated_df["Group"].tolist() == ["A", "B"]
    assert curated_df["OS Event"].tolist() == ["0", "1"]


def _materialize_parent_cleaner_state(*, dataset) -> None:
    source_df = dataset._get_source_df()
    col_profiles, survival_profiles, survival_pairs = _parent_semantics(source_df)
    bundle = dataset.cleaner._get_group_bundle(version=1, config_version=1)
    curated_group = bundle["config"]["group"].require_group(
        StatomixLayout.CURATED_GROUP
    )
    curated_root = BaseZARR.get_abs_path(curated_group)

    source_df.to_parquet(
        path=curated_root / StatomixLayout.CURATED_DF,
        index=False,
    )
    ColReport().save_col_profiles(
        col_profiles=col_profiles,
        path=curated_root / StatomixLayout.CURATED_COL_PROFILES,
    )
    survival_pairs.save(path=curated_root / StatomixLayout.CURATED_SURV_PAIRS)
    SurvMetaReport().save_semantic_profiles(
        semantic_profiles=survival_profiles,
        path=bundle["config"]["path"] / "surv_profiles_curated.parquet",
    )


def test_cleaner_materializes_independent_inherited_state(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MPLBACKEND", "Agg")

    from statomix import Project

    project = Project(
        project_name="inheritance_test",
        project_dir=tmp_path / "projects",
    )
    source_dataset = project.add_dataset(
        df=_source_df(),
        dataset_name="Discovery Cohort",
    )
    target_df = _target_df()
    target_dataset = project.add_dataset(
        df=target_df,
        dataset_name="Validation_Cohort_Corrected_Survival",
    )
    assert source_dataset is not None
    assert target_dataset is not None
    _materialize_parent_cleaner_state(dataset=source_dataset)

    result = target_dataset.cleaner.inherit_curated_state(
        source_cleaner=source_dataset.cleaner,
        source_version=1,
        source_config_version=1,
        target_version=1,
        target_config_version=1,
        column_mapping={"OS Months": "OS_months"},
        changed_columns=["OS Event", "OS_months"],
        row_key="Patient ID",
        strict=True,
        replace=False,
    )

    target_bundle = target_dataset.cleaner._find_group_bundle(
        version=1,
        config_version=1,
    )
    curated_paths = StatomixLayout(
        root=target_bundle["config"]["path"]
    ).curated_artifacts()
    assert_frame_equal(
        left=pd.read_parquet(curated_paths["df"]),
        right=target_df,
        check_dtype=True,
        check_exact=True,
    )

    inherited_pairs = SurvPairs.load(path=curated_paths["surv_pairs"])
    assert list(inherited_pairs.pairs) == ["OS"]
    assert inherited_pairs.pairs["OS"].event_profile.col_name == "OS Event"
    assert inherited_pairs.pairs["OS"].time_profile.col_name == "OS_months"

    lineage = target_bundle["config"]["group"].attrs["meta"]["curated_parent"]
    assert lineage["source"]["dataset"] == "Discovery Cohort"
    assert lineage["column_mapping"] == {"OS Months": "OS_months"}
    assert result["survival_pairs"] == ["OS"]

    curated_report = target_bundle["config"]["path"] / ("surv_meta_report_curated.xlsx")
    report_df = pd.read_excel(curated_report, sheet_name="SurvMeta")
    labels = report_df.set_index("col_name")["survival_label"].dropna().to_dict()
    assert labels == {"OS Event": "OS", "OS_months": "OS"}

    with pytest.raises(FileExistsError, match="replace=True"):
        target_dataset.cleaner.inherit_curated_state(
            source_cleaner=source_dataset.cleaner,
            source_version=1,
            source_config_version=1,
            target_version=1,
            target_config_version=1,
            column_mapping={"OS Months": "OS_months"},
            changed_columns=["OS Event", "OS_months"],
            row_key="Patient ID",
            strict=True,
            replace=False,
        )
