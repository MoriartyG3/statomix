from __future__ import annotations

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from statomix.curation.categorical import (
    CatEdit,
    CatMetaEditSchema,
    CatMetaReport,
)
from statomix.curation.columns import ColEditSchema
from statomix.curation.service import apply_curation_schemas
from statomix.curation.survival import (
    SurvCatMetaEditSchema,
)
from statomix.storage.parquet_metadata import (
    load_category_rank_metadata,
    select_lowest_rank_reference,
    write_dataframe_with_category_ranks,
)


def make_ranked_stage_schema() -> CatMetaEditSchema:
    return CatMetaEditSchema(
        cat_edits={
            "T Stage": {
                stage: CatEdit(
                    col_name="T Stage",
                    category=stage,
                    rename_to=None,
                    remove=False,
                    rank=rank,
                )
                for rank, stage in enumerate(("T0", "T1", "T2", "T3"))
            }
        }
    )


def test_rank_only_schema_round_trip_preserves_ranks(
    tmp_path,
):
    schema = make_ranked_stage_schema()
    path = tmp_path / "cat_meta_edit_schema.parquet"

    schema.save(path=path)

    loaded_schema = CatMetaEditSchema.load(path=path)

    assert loaded_schema.category_ranks == {
        "T Stage": {
            "T0": 0,
            "T1": 1,
            "T2": 2,
            "T3": 3,
        }
    }

    assert loaded_schema.edit_count == 4


def test_report_keeps_rank_only_rows():
    index = pd.MultiIndex.from_tuples(
        [
            ("T Stage", "T0"),
            ("T Stage", "T1"),
        ],
        names=["col_name", "category"],
    )

    report_df = pd.DataFrame(
        {
            "rename_to": [None, None],
            "remove": [None, None],
            "rank": [0, 1],
        },
        index=index,
    )

    edits = CatMetaReport._get_cat_edits(
        cat_meta_df=report_df,
    )

    assert edits["T Stage"]["T0"].rank == 0
    assert edits["T Stage"]["T1"].rank == 1


def test_partial_ranking_is_rejected():
    schema = CatMetaEditSchema(
        cat_edits={
            "T Stage": {
                "T0": CatEdit(
                    col_name="T Stage",
                    category="T0",
                    rename_to=None,
                    remove=False,
                    rank=0,
                ),
                "T1": CatEdit(
                    col_name="T Stage",
                    category="T1",
                    rename_to=None,
                    remove=False,
                    rank=None,
                ),
            }
        }
    )

    with pytest.raises(
        ValueError,
        match="categories without ranks",
    ):
        _ = schema.category_ranks


def test_duplicate_ranks_are_rejected():
    schema = CatMetaEditSchema(
        cat_edits={
            "T Stage": {
                stage: CatEdit(
                    col_name="T Stage",
                    category=stage,
                    rename_to=None,
                    remove=False,
                    rank=0,
                )
                for stage in ("T0", "T1")
            }
        }
    )

    with pytest.raises(
        ValueError,
        match="must be unique",
    ):
        _ = schema.category_ranks


def test_rank_only_edits_do_not_change_dataframe_values():
    source_df = pd.DataFrame(
        {
            "T Stage": [
                "T0",
                "T2",
                "T1",
            ]
        }
    )

    curated_df = apply_curation_schemas(
        df=source_df,
        rename_mapping={},
        col_edit_schema=ColEditSchema.empty(),
        cat_meta_edit_schema=make_ranked_stage_schema(),
        surv_cat_meta_edit_schema=(SurvCatMetaEditSchema.empty()),
    )

    assert_frame_equal(
        curated_df,
        source_df,
        check_exact=True,
        check_dtype=True,
    )


def test_parquet_contains_typed_rank_metadata_without_changing_values(
    tmp_path,
):
    source_df = pd.DataFrame(
        {
            "T Stage": [0, 2, 1, 3],
            "Treatment": [
                "RT",
                "CT+RT",
                "RT",
                "CT+RT",
            ],
        }
    )

    path = tmp_path / "df.parquet"

    payload = write_dataframe_with_category_ranks(
        df=source_df,
        path=path,
        category_ranks={
            "T Stage": {
                "0": 0,
                "1": 1,
                "2": 2,
                "3": 3,
            }
        },
    )

    reloaded_df = pd.read_parquet(
        path=path,
    )

    reloaded_payload = load_category_rank_metadata(
        path=path,
    )

    assert_frame_equal(
        reloaded_df,
        source_df,
        check_exact=True,
        check_dtype=True,
    )

    assert reloaded_payload == payload

    assert payload["columns"]["T Stage"] == [
        {
            "category": {
                "type": "integer",
                "value": 0,
            },
            "rank": 0,
        },
        {
            "category": {
                "type": "integer",
                "value": 1,
            },
            "rank": 1,
        },
        {
            "category": {
                "type": "integer",
                "value": 2,
            },
            "rank": 2,
        },
        {
            "category": {
                "type": "integer",
                "value": 3,
            },
            "rank": 3,
        },
    ]


def test_reference_is_lowest_ranked_observed_category(
    tmp_path,
):
    full_df = pd.DataFrame(
        {
            "T Stage": [
                "T0",
                "T1",
                "T2",
                "T3",
            ]
        }
    )

    path = tmp_path / "df.parquet"

    write_dataframe_with_category_ranks(
        df=full_df,
        path=path,
        category_ranks={
            "T Stage": {
                "T0": 0,
                "T1": 1,
                "T2": 2,
                "T3": 3,
            }
        },
    )

    metadata = load_category_rank_metadata(
        path=path,
    )

    analysis_df = pd.DataFrame(
        {
            "T Stage": [
                "T1",
                "T2",
                "T3",
            ]
        }
    )

    with pytest.warns(
        UserWarning,
        match="selected 'T1'",
    ):
        reference = select_lowest_rank_reference(
            df=analysis_df,
            column_name="T Stage",
            rank_metadata=metadata,
        )

    assert reference.category == "T1"
    assert reference.rank == 1

    assert reference.absent_lower_ranked_categories == ("T0",)


def test_observed_category_without_rank_is_rejected(
    tmp_path,
):
    df = pd.DataFrame(
        {
            "T Stage": [
                "T0",
                "T1",
                "T2",
            ]
        }
    )

    with pytest.raises(
        ValueError,
        match="observed categories without ranks",
    ):
        write_dataframe_with_category_ranks(
            df=df,
            path=tmp_path / "df.parquet",
            category_ranks={
                "T Stage": {
                    "T0": 0,
                    "T1": 1,
                }
            },
        )
