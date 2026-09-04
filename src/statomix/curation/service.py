"""Pure application of persisted curation schemas to a dataframe."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection

import pandas as pd

from statomix.curation.categorical.report import CatMetaEditSchema
from statomix.curation.columns.report import ColEditSchema
from statomix.curation.survival.events import (
    apply_survival_event_edits,
    normalize_survival_event_columns,
)
from statomix.curation.survival.report import SurvCatMetaEditSchema


def _get_category_rename_mapping(category_edits):
    rename_mapping = defaultdict(dict)

    for col_name, column_edits in category_edits.items():
        for schema in column_edits.values():
            if schema.category is not None and schema.rename_to is not None:
                rename_mapping[col_name][schema.category] = schema.rename_to
            elif schema.remove:
                rename_mapping[col_name][schema.category] = pd.NA
            elif getattr(schema, "rank", None) is not None:
                # Ranks remain metadata; they never replace category values.
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
    survival_event_columns: Collection[str] | None = None,
) -> pd.DataFrame:
    """Apply Cleaner instructions without modifying the supplied dataframe.

    Empty schemas preserve the dataframe when no event columns are declared.
    Declared event columns are normalized even when they need no value mapping.
    """

    if isinstance(survival_event_columns, (str, bytes)):
        raise TypeError("survival_event_columns must be a collection, not one string.")

    if survival_event_columns is not None:
        unexpected_columns = set(surv_cat_meta_edit_schema.cat_edits) - set(
            survival_event_columns
        )

        if unexpected_columns:
            raise ValueError(
                "Survival-event edits target columns not declared as events: "
                f"{sorted(unexpected_columns)!r}."
            )

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

    category_mapping = _get_category_rename_mapping(
        category_edits=cat_meta_edit_schema.cat_edits
    )
    for col_name, mapping in category_mapping.items():
        curated_df[col_name] = curated_df[col_name].replace(mapping)

    curated_df = apply_survival_event_edits(
        df=curated_df,
        category_edits=surv_cat_meta_edit_schema.cat_edits,
    )

    if survival_event_columns is not None:
        curated_df = normalize_survival_event_columns(
            df=curated_df,
            event_columns=survival_event_columns,
        )

    return curated_df
