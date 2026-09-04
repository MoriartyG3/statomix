from __future__ import annotations

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from statomix.curation.categorical import (
    CatEdit,
    CatMetaEditSchema,
)
from statomix.curation.columns import ColEditSchema
from statomix.curation.service import apply_curation_schemas
from statomix.curation.survival import SurvCatMetaEditSchema
from statomix.storage.parquet_metadata import (
    write_dataframe_with_category_ranks,
)


def make_stage_schema(*, rank_only: bool) -> CatMetaEditSchema:
    edits = {}

    for rank, stage in enumerate((1, 2, 3, 4)):
        edits[str(stage)] = CatEdit(
            col_name="T Stage",
            category=str(stage),
            rename_to=None if rank_only else f"T{stage}",
            remove=False,
            rank=rank,
        )

    return CatMetaEditSchema(
        cat_edits={
            "T Stage": edits,
        }
    )


def apply_stage_schema(
    *,
    source_df: pd.DataFrame,
    schema: CatMetaEditSchema,
) -> pd.DataFrame:
    return apply_curation_schemas(
        df=source_df,
        rename_mapping={},
        col_edit_schema=ColEditSchema.empty(),
        cat_meta_edit_schema=schema,
        surv_cat_meta_edit_schema=SurvCatMetaEditSchema.empty(),
    )


def test_numeric_categories_are_renamed_before_rank_metadata_write(
    tmp_path,
):
    source_df = pd.DataFrame(
        {
            "T Stage": pd.Series(
                [1, 2, 3, 4, 2, 1],
                dtype="int64",
            )
        }
    )

    schema = make_stage_schema(rank_only=False)

    curated_df = apply_stage_schema(
        source_df=source_df,
        schema=schema,
    )

    assert curated_df["T Stage"].tolist() == [
        "T1",
        "T2",
        "T3",
        "T4",
        "T2",
        "T1",
    ]

    path = tmp_path / "curated.parquet"

    payload = write_dataframe_with_category_ranks(
        df=curated_df,
        path=path,
        category_ranks=schema.category_ranks,
    )

    assert [
        (
            record["category"]["type"],
            record["category"]["value"],
            record["rank"],
        )
        for record in payload["columns"]["T Stage"]
    ] == [
        ("string", "T1", 0),
        ("string", "T2", 1),
        ("string", "T3", 2),
        ("string", "T4", 3),
    ]


def test_rank_only_numeric_categories_remain_unchanged():
    source_df = pd.DataFrame(
        {
            "T Stage": pd.Series(
                [1, 2, 3, 4],
                dtype="int64",
            )
        }
    )

    schema = make_stage_schema(rank_only=True)

    curated_df = apply_stage_schema(
        source_df=source_df,
        schema=schema,
    )

    assert_frame_equal(
        curated_df,
        source_df,
        check_dtype=True,
    )


def test_string_categories_are_still_renamed():
    source_df = pd.DataFrame(
        {
            "T Stage": ["1", "2", "3", "4"],
        }
    )

    schema = make_stage_schema(rank_only=False)

    curated_df = apply_stage_schema(
        source_df=source_df,
        schema=schema,
    )

    assert curated_df["T Stage"].tolist() == [
        "T1",
        "T2",
        "T3",
        "T4",
    ]


def test_mixed_numeric_and_string_category_labels_are_rejected():
    source_df = pd.DataFrame(
        {
            "T Stage": pd.Series(
                [1, "1"],
                dtype="object",
            )
        }
    )

    schema = CatMetaEditSchema(
        cat_edits={
            "T Stage": {
                "1": CatEdit(
                    col_name="T Stage",
                    category="1",
                    rename_to="T1",
                    remove=False,
                    rank=0,
                )
            }
        }
    )

    with pytest.raises(
        ValueError,
        match="ambiguous",
    ):
        apply_stage_schema(
            source_df=source_df,
            schema=schema,
        )


def test_unobserved_category_edit_is_rejected():
    source_df = pd.DataFrame(
        {
            "T Stage": pd.Series(
                [1, 2],
                dtype="int64",
            )
        }
    )

    schema = CatMetaEditSchema(
        cat_edits={
            "T Stage": {
                "3": CatEdit(
                    col_name="T Stage",
                    category="3",
                    rename_to="T3",
                    remove=False,
                    rank=2,
                )
            }
        }
    )

    with pytest.raises(
        ValueError,
        match="no observed value matches",
    ):
        apply_stage_schema(
            source_df=source_df,
            schema=schema,
        )
