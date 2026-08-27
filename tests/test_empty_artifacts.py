from __future__ import annotations

import pandas as pd
from pandas.testing import assert_frame_equal

from statomix.pipelines.cleaner.cat_meta_report import (
    CatMetaEditSchema,
)
from statomix.pipelines.cleaner.col.col_profiler import ColProfile
from statomix.pipelines.cleaner.col.col_report import (
    ColEditSchema,
    ColReport,
)
from statomix.pipelines.cleaner.curation import (
    apply_curation_schemas,
)
from statomix.pipelines.cleaner.surv.surv_profiler import (
    SurvivalSemanticProfile,
)
from statomix.pipelines.cleaner.surv.surv_report import (
    SurvCatMetaEditSchema,
    SurvEditSchema,
    SurvMetaReport,
    SurvPairs,
)


def test_empty_edit_artifacts_round_trip_with_stable_columns(tmp_path):
    artifacts = [
        (
            ColEditSchema.empty(),
            ColEditSchema,
            ColEditSchema.PARQUET_SCHEMA,
        ),
        (
            CatMetaEditSchema.empty(),
            CatMetaEditSchema,
            CatMetaEditSchema.PARQUET_SCHEMA,
        ),
        (
            SurvEditSchema.empty(),
            SurvEditSchema,
            SurvEditSchema.PARQUET_SCHEMA,
        ),
        (
            SurvPairs.empty(),
            SurvPairs,
            SurvPairs.PARQUET_SCHEMA,
        ),
        (
            SurvCatMetaEditSchema.empty(),
            SurvCatMetaEditSchema,
            SurvCatMetaEditSchema.PARQUET_SCHEMA,
        ),
    ]

    for index, (artifact, artifact_type, schema) in enumerate(artifacts):
        path = tmp_path / f"artifact_{index}.parquet"
        artifact.save(path=path)

        stored_frame = pd.read_parquet(path=path)
        assert stored_frame.empty
        assert list(stored_frame.columns) == list(schema)

        loaded_artifact = artifact_type.load(path=path)
        assert loaded_artifact.is_empty


def test_empty_profile_artifacts_keep_their_contracts(tmp_path):
    col_profiles_path = tmp_path / "col_profiles.parquet"
    ColReport().save_col_profiles(
        col_profiles={},
        path=col_profiles_path,
    )
    col_profiles_df = pd.read_parquet(path=col_profiles_path)

    assert col_profiles_df.empty
    assert list(col_profiles_df.columns) == list(ColProfile.PARQUET_SCHEMA)
    assert ColReport.load_col_profiles(path=col_profiles_path) == {}

    surv_profiles_path = tmp_path / "surv_profiles.parquet"
    surv_report = SurvMetaReport()
    surv_report.save_semantic_profiles(
        semantic_profiles={},
        path=surv_profiles_path,
    )
    surv_profiles_df = pd.read_parquet(path=surv_profiles_path)

    assert surv_profiles_df.empty
    assert list(surv_profiles_df.columns) == list(
        SurvivalSemanticProfile.PARQUET_SCHEMA
    )
    assert surv_report.load_semantic_profiles(path=surv_profiles_path) == {}


def test_empty_schemas_are_dataframe_identity_elements():
    source_df = pd.DataFrame(
        {
            "age": pd.Series([41, 52, 63], dtype="int64"),
            "group": pd.Series(["A", "B", "A"], dtype="object"),
        }
    )

    curated_df = apply_curation_schemas(
        df=source_df,
        rename_mapping={},
        col_edit_schema=ColEditSchema.empty(),
        cat_meta_edit_schema=CatMetaEditSchema.empty(),
        surv_cat_meta_edit_schema=SurvCatMetaEditSchema.empty(),
    )

    assert_frame_equal(
        left=curated_df,
        right=source_df,
        check_dtype=True,
        check_exact=True,
    )
