"""Pure application of persisted curation schemas to a dataframe."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Collection, Mapping
from typing import Any

import pandas as pd

from statomix.curation.categorical.report import CatMetaEditSchema
from statomix.curation.columns.report import ColEditSchema
from statomix.curation.survival.events import (
    apply_survival_event_edits,
    normalize_survival_event_columns,
)
from statomix.curation.survival.report import SurvCatMetaEditSchema


def _python_scalar(value: object) -> object:
    """Convert a NumPy scalar into its Python scalar equivalent."""

    item_method = getattr(value, "item", None)

    if callable(item_method):
        try:
            return item_method()
        except (TypeError, ValueError):
            pass

    return value


def _category_labels(value: object) -> set[str]:
    """Return textual labels that may represent one category value."""

    scalar = _python_scalar(value)
    labels = {str(scalar)}

    if isinstance(scalar, float) and math.isfinite(scalar) and scalar.is_integer():
        labels.add(str(int(scalar)))

    return labels


def _resolve_category_value(
    *,
    column_name: str,
    requested_category: object,
    observed_values: tuple[object, ...],
) -> object:
    """Resolve an edit-schema label against one observed source value.

    The Excel edit schema stores category labels as text. This resolver
    therefore permits an unambiguous match such as ``"1"`` to numeric
    source value ``1``. If multiple typed source values share the same
    displayed label, the operation fails instead of guessing.
    """

    requested_labels = _category_labels(requested_category)

    matches = [
        observed_value
        for observed_value in observed_values
        if requested_labels.intersection(_category_labels(observed_value))
    ]

    if len(matches) == 1:
        return matches[0]

    formatted_observed = [
        f"{value!r} ({type(value).__name__})" for value in observed_values
    ]

    if not matches:
        raise ValueError(
            f"Categorical edit for column {column_name!r} refers to "
            f"category {requested_category!r}, but no observed value "
            f"matches it. Observed values: {formatted_observed!r}."
        )

    formatted_matches = [f"{value!r} ({type(value).__name__})" for value in matches]

    raise ValueError(
        f"Categorical edit for column {column_name!r} has ambiguous "
        f"category {requested_category!r}; it matches multiple observed "
        f"values: {formatted_matches!r}. Statomix cannot safely apply "
        "this edit because the source values have distinct types."
    )


def _get_category_rename_mapping(
    category_edits: Mapping[str, Mapping[object, Any]],
    *,
    df: pd.DataFrame | None = None,
) -> dict[str, dict[object, object]]:
    """Build source-aware rename/removal mappings.

    When ``df`` is supplied, persisted edit-schema labels are resolved
    against the actual observed scalar values in each DataFrame column.

    Omitting ``df`` preserves the historical helper behavior for callers
    that use this private function directly. The production curation path
    always supplies ``df``.
    """

    rename_mapping: defaultdict[str, dict[object, object]] = defaultdict(dict)

    for col_name, column_edits in category_edits.items():
        if df is not None:
            if col_name not in df.columns:
                raise KeyError(
                    f"Categorical edits refer to missing column " f"{col_name!r}."
                )

            observed_values = tuple(
                _python_scalar(value) for value in pd.unique(df[col_name].dropna())
            )
        else:
            observed_values = ()

        for schema in column_edits.values():
            if schema.category is not None and schema.rename_to is not None:
                replacement = schema.rename_to
            elif schema.category is not None and schema.remove:
                replacement = pd.NA
            elif getattr(schema, "rank", None) is not None:
                # Ranks remain metadata; they never replace category values.
                continue
            else:
                raise ValueError(
                    f"Column {col_name} has an edit with neither a "
                    "rename target nor a removal instruction."
                )

            if df is None:
                source_value = _python_scalar(schema.category)
            else:
                source_value = _resolve_category_value(
                    column_name=col_name,
                    requested_category=schema.category,
                    observed_values=observed_values,
                )

            if source_value in rename_mapping[col_name]:
                raise ValueError(
                    f"Multiple categorical edits for column {col_name!r} "
                    f"resolve to the same observed value "
                    f"{source_value!r}."
                )

            rename_mapping[col_name][source_value] = replacement

    return dict(rename_mapping)


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
        category_edits=cat_meta_edit_schema.cat_edits,
        df=curated_df,
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
