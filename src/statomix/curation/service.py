"""Pure application of persisted curation schemas to a dataframe."""

from __future__ import annotations

from collections import defaultdict

import pandas as pd

from statomix.curation.categorical.report import CatMetaEditSchema
from statomix.curation.columns.report import ColEditSchema
from statomix.curation.survival.report import SurvCatMetaEditSchema


def _get_category_rename_mapping(category_edits):
    rename_mapping = defaultdict(dict)

    for col_name, column_edits in category_edits.items():
        for schema in column_edits.values():
            if schema.category is not None and schema.rename_to is not None:
                rename_mapping[col_name][schema.category] = schema.rename_to
            # elif schema.remove:
            #    rename_mapping[col_name][schema.category] = pd.NA
            elif schema.remove:
                rename_mapping[col_name][schema.category] = pd.NA
            elif getattr(schema, "rank", None) is not None:
                # Ranks are metadata only. They must never replace values
                # in the curated DataFrame.
                continue
            else:
                raise ValueError(
                    f"Column {col_name} has an edit with neither a rename "
                    "target nor a removal instruction."
                )

    return rename_mapping


def apply_curation_schemas(
    df: pd.DataFrame,
    rename_mapping: dict[str, str],
    col_edit_schema: ColEditSchema,
    cat_meta_edit_schema: CatMetaEditSchema,
    surv_cat_meta_edit_schema: SurvCatMetaEditSchema,
) -> pd.DataFrame:
    """Apply every Cleaner edit contract to a dataframe.

    Empty schemas are identity elements: when all supplied schemas and the
    rename mapping are empty, the returned dataframe is equal to ``df``.
    """

    remove_cols = [
        col_name
        for col_name, col_edit in col_edit_schema.edits.items()
        if col_edit.remove
    ]
    rename_mapping_swapped = {
        old_name: new_name for new_name, old_name in rename_mapping.items()
    }

    curated_df = df.drop(columns=remove_cols)
    curated_df = curated_df.rename(columns=rename_mapping_swapped)

    cat_rename_mapping = _get_category_rename_mapping(
        category_edits=cat_meta_edit_schema.cat_edits
    )
    for col_name, category_mapping in cat_rename_mapping.items():
        curated_df[col_name] = curated_df[col_name].replace(category_mapping)

    surv_cat_rename_mapping = _get_category_rename_mapping(
        category_edits=surv_cat_meta_edit_schema.cat_edits
    )
    for col_name, category_mapping in surv_cat_rename_mapping.items():
        curated_df[col_name] = curated_df[col_name].replace(category_mapping)

    return curated_df
