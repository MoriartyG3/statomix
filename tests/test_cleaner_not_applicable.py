from __future__ import annotations

import pandas as pd

from statomix.pipelines.cleaner.cat_meta_report import (
    CatMetaEditSchema,
    CatMetaReport,
)
from statomix.pipelines.cleaner.cleaner import Cleaner
from statomix.pipelines.cleaner.col.col_profiler import ColProfile
from statomix.pipelines.cleaner.col.col_report import ColReport
from statomix.pipelines.cleaner.col.col_semantic_rules import DataTypes
from statomix.pipelines.cleaner.surv.surv_report import (
    SurvCatMetaEditSchema,
    SurvEditSchema,
    SurvMetaReport,
    SurvPairs,
)


class _FakeGroup:
    def __init__(self, meta):
        self.attrs = {"meta": dict(meta)}


def _numerical_profile() -> ColProfile:
    return ColProfile(
        col_name="age",
        col_type=DataTypes.NUMERICAL,
        missing_n=0,
        missing_pct=0.0,
        num_conversion_pct=100.0,
        unique_n=3,
        tokens=["age"],
        normalized_name="age",
    )


def _cleaner_without_categorical_or_survival_columns(tmp_path):
    version_path = tmp_path / "version1"
    config_path = version_path / "config1"
    user_config_path = tmp_path / "user_config"

    version_path.mkdir(parents=True)
    config_path.mkdir(parents=True)
    user_config_path.mkdir(parents=True)

    source_df_path = tmp_path / "source_df.parquet"
    pd.DataFrame({"age": [41.0, 52.0, 63.0]}).to_parquet(
        path=source_df_path,
        index=False,
    )

    col_report = ColReport()
    col_report.save_col_profiles(
        col_profiles={"age": _numerical_profile()},
        path=version_path / "col_profiles_curated.parquet",
    )

    version_group = _FakeGroup(
        meta={
            "version": 1,
            "name": "default",
            "config": {
                "latest_version": 1,
                "version_history": [1],
            },
        }
    )
    config_group = _FakeGroup(meta={"version": 1, "name": "default"})

    group_bundle = {
        "version": {
            "group": version_group,
            "path": version_path,
            "meta": version_group.attrs["meta"],
        },
        "config": {
            "group": config_group,
            "path": config_path,
            "meta": config_group.attrs["meta"],
        },
    }

    cleaner = Cleaner.__new__(Cleaner)
    cleaner.df_path = source_df_path
    cleaner.dataset_name = "test_dataset"
    cleaner.paths = {"user_config": user_config_path}
    cleaner.col_report = col_report
    cleaner.cat_meta_report = CatMetaReport()
    cleaner.surv_meta_report = SurvMetaReport()
    cleaner._get_group_bundle = lambda **kwargs: group_bundle

    return cleaner, config_path, config_group


def test_cleaner_materializes_noop_contracts_and_moves_forward(tmp_path):
    cleaner, config_path, config_group = (
        _cleaner_without_categorical_or_survival_columns(tmp_path=tmp_path)
    )

    cleaner.create_cat_meta_report(version=1, config_version=1)
    cleaner.create_cat_meta_edit_schema(version=1, config_version=1)
    cleaner.create_surv_meta_report(version=1, config_version=1)
    cleaner.create_surv_meta_edit_schema(version=1, config_version=1)
    cleaner.create_surv_cat_meta_report(version=1, config_version=1)
    cleaner.create_surv_cat_meta_edit_schema(
        version=1,
        config_version=1,
    )

    assert not (config_path / "cat_meta_report.xlsx").exists()
    assert not (config_path / "surv_meta_report.xlsx").exists()
    assert not (config_path / "surv_cat_meta_report.xlsx").exists()

    assert CatMetaEditSchema.load(
        path=config_path / "cat_meta_edit_schema.parquet"
    ).is_empty
    assert SurvEditSchema.load(
        path=config_path / "surv_meta_edit_schema.parquet"
    ).is_empty
    assert SurvPairs.load(path=config_path / "surv_pairs.parquet").is_empty
    assert SurvCatMetaEditSchema.load(
        path=config_path / "surv_cat_meta_edit_schema.parquet"
    ).is_empty

    survival_profiles = cleaner.surv_meta_report.load_semantic_profiles(
        path=config_path / "surv_profiles.parquet"
    )
    curated_survival_profiles = cleaner.surv_meta_report.load_semantic_profiles(
        path=config_path / "surv_profiles_curated.parquet"
    )
    assert survival_profiles == {}
    assert curated_survival_profiles == {}

    config_meta = config_group.attrs["meta"]
    assert config_meta["curated_datatype_counts"]["Numerical"] == 1
    assert config_meta["curated_datatype_counts"]["Categorical"] == 0
    assert config_meta["curated_datatype_counts"]["Survival"] == 0

    expected_not_applicable = {
        "categorical_meta_report",
        "categorical_meta_edit_schema",
        "survival_meta_report",
        "survival_meta_edit_schema",
        "survival_categorical_meta_report",
        "survival_categorical_meta_edit_schema",
    }
    statuses = config_meta["procedure_status"]

    assert expected_not_applicable <= set(statuses)
    assert all(
        statuses[procedure]["status"] == "not_applicable"
        for procedure in expected_not_applicable
    )
