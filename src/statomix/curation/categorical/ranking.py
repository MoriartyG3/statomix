"""Validation and interpretation of categorical rank instructions."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral, Real
from typing import Protocol

import pandas as pd


class RankedCategoryEdit(Protocol):
    """Structural contract needed to derive a category-rank mapping."""

    category: object
    rename_to: str | None
    remove: bool
    rank: int | None


def parse_optional_rank(
    value: object,
    *,
    col_name: str,
    category: object,
) -> int | None:
    """Parse one optional Excel rank without silently truncating it."""

    if value is None or pd.isna(value):
        return None

    if isinstance(value, bool):
        raise ValueError(
            f"Rank for {col_name!r}, category {category!r}, must be an "
            "integer, not a Boolean."
        )

    if isinstance(value, Integral):
        rank = int(value)
    elif isinstance(value, Real) and float(value).is_integer():
        rank = int(value)
    elif isinstance(value, str):
        stripped = value.strip()

        try:
            rank = int(stripped)
        except ValueError as exc:
            raise ValueError(
                f"Rank for {col_name!r}, category {category!r}, must be "
                f"an integer; received {value!r}."
            ) from exc
    else:
        raise ValueError(
            f"Rank for {col_name!r}, category {category!r}, must be an "
            f"integer; received {value!r}."
        )

    if rank < 0:
        raise ValueError(
            f"Rank for {col_name!r}, category {category!r}, must be " "zero or greater."
        )

    return rank


def build_category_rank_mapping(
    cat_edits: Mapping[
        str,
        Mapping[object, RankedCategoryEdit],
    ],
) -> dict[str, dict[str, int]]:
    """Validate edits and return final category labels mapped to ranks.

    Category labels in the human-editable report and edit-schema
    Parquet are represented as strings. Their actual scalar types are
    recovered from the curated DataFrame when its Parquet metadata is
    written.
    """

    rank_mapping: dict[str, dict[str, int]] = {}

    for col_name, column_edits in cat_edits.items():
        retained_edits = [edit for edit in column_edits.values() if not edit.remove]

        removed_with_ranks = [
            edit.category
            for edit in column_edits.values()
            if edit.remove and edit.rank is not None
        ]

        if removed_with_ranks:
            raise ValueError(
                f"Removed categories in {col_name!r} must not have "
                f"ranks: {removed_with_ranks!r}."
            )

        ranks = [edit.rank for edit in retained_edits]

        if not any(rank is not None for rank in ranks):
            continue

        missing_categories = [
            edit.category for edit in retained_edits if edit.rank is None
        ]

        if missing_categories:
            raise ValueError(
                f"Ranked categorical column {col_name!r} has "
                f"categories without ranks: {missing_categories!r}. "
                "Rank every retained category or leave every rank blank."
            )

        final_labels_to_ranks: dict[str, int] = {}

        for edit in retained_edits:
            final_label = (
                edit.rename_to if edit.rename_to is not None else edit.category
            )

            label_key = str(final_label)

            if edit.rank is None:
                raise RuntimeError("Rank completeness validation failed internally.")

            rank = edit.rank
            existing_rank = final_labels_to_ranks.get(label_key)

            if existing_rank is not None and existing_rank != rank:
                raise ValueError(
                    f"Categories merged into {final_label!r} in "
                    f"{col_name!r} have conflicting ranks "
                    f"{existing_rank} and {rank}."
                )

            final_labels_to_ranks[label_key] = rank

        rank_to_labels: dict[int, list[str]] = {}

        for label, rank in final_labels_to_ranks.items():
            rank_to_labels.setdefault(rank, []).append(label)

        duplicated_ranks = {
            rank: labels for rank, labels in rank_to_labels.items() if len(labels) > 1
        }

        if duplicated_ranks:
            raise ValueError(
                f"Categorical ranks for {col_name!r} must be unique "
                f"after renaming; duplicated ranks: "
                f"{duplicated_ranks!r}."
            )

        observed_ranks = sorted(rank_to_labels)
        expected_ranks = list(range(len(rank_to_labels)))

        if observed_ranks != expected_ranks:
            raise ValueError(
                f"Categorical ranks for {col_name!r} must be "
                f"contiguous from 0 to "
                f"{len(rank_to_labels) - 1}; received "
                f"{observed_ranks!r}."
            )

        rank_mapping[col_name] = final_labels_to_ranks

    return rank_mapping
