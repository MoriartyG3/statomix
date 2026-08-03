from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
import unicodedata

import numpy as np
import pandas as pd
from great_tables import GT, md


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _normalize_variable_name(value: object) -> str:
    """
    Normalize labels so that, for example:
        Total GC’s
        Total GC's
    are treated as the same variable.
    """
    if pd.isna(value):
        return ""

    value = unicodedata.normalize("NFKC", str(value))

    value = (
        value
        .replace("’", "'")
        .replace("‘", "'")
        .replace("′", "'")
        .strip()
    )

    # Collapse repeated whitespace
    value = re.sub(r"\s+", " ", value)

    return value


def _format_number(
    value: object,
    decimals: int = 2,
    missing_symbol: str = "—",
) -> str:
    """Format a numeric value while safely handling missing values."""
    if pd.isna(value):
        return missing_symbol

    return f"{float(value):.{decimals}f}"


def _format_integer(
    value: object,
    missing_symbol: str = "—",
) -> str:
    """Format integer-like values."""
    if pd.isna(value):
        return missing_symbol

    return f"{int(round(float(value))):,}"


def _format_statistic(
    row: pd.Series,
    statistic: str,
    decimals: int,
    missing_symbol: str,
) -> str:
    """Convert one raw summary-statistic row to publication format."""

    if statistic == "n":
        return _format_integer(
            row.get("n"),
            missing_symbol=missing_symbol,
        )

    if statistic == "mean_sd":
        mean = row.get("mean")
        std = row.get("std")

        if pd.isna(mean) and pd.isna(std):
            return missing_symbol

        return (
            f"{_format_number(mean, decimals, missing_symbol)} "
            f"({_format_number(std, decimals, missing_symbol)})"
        )

    if statistic == "median_quartiles":
        median = row.get("median")
        q1 = row.get("q1")
        q3 = row.get("q3")

        if pd.isna(median) and pd.isna(q1) and pd.isna(q3):
            return missing_symbol

        return (
            f"{_format_number(median, decimals, missing_symbol)} "
            f"[{_format_number(q1, decimals, missing_symbol)}, "
            f"{_format_number(q3, decimals, missing_symbol)}]"
        )

    if statistic == "median_iqr":
        median = row.get("median")
        iqr = row.get("iqr")

        if pd.isna(median) and pd.isna(iqr):
            return missing_symbol

        return (
            f"{_format_number(median, decimals, missing_symbol)} "
            f"({_format_number(iqr, decimals, missing_symbol)})"
        )

    if statistic == "range":
        minimum = row.get("min")
        maximum = row.get("max")

        if pd.isna(minimum) and pd.isna(maximum):
            return missing_symbol

        return (
            f"{_format_number(minimum, decimals, missing_symbol)}"
            f"–"
            f"{_format_number(maximum, decimals, missing_symbol)}"
        )

    if statistic == "missing":
        missing_n = row.get("missing_n")
        missing_pct = row.get("missing_pct")

        if pd.isna(missing_n) and pd.isna(missing_pct):
            return missing_symbol

        return (
            f"{_format_integer(missing_n, missing_symbol)} "
            f"({_format_number(missing_pct, decimals, missing_symbol)}%)"
        )

    if statistic == "unique_n":
        return _format_integer(
            row.get("unique_n"),
            missing_symbol=missing_symbol,
        )

    if statistic in {"p01", "p05", "p95", "p99"}:
        return _format_number(
            row.get(statistic),
            decimals=decimals,
            missing_symbol=missing_symbol,
        )

    raise ValueError(f"Unsupported statistic: {statistic!r}")


# Labels displayed in the final table
STATISTIC_LABELS = {
    "n": "n",
    "unique_n": "Unique",
    "mean_sd": "Mean (SD)",
    "median_quartiles": "Median [Q1, Q3]",
    "median_iqr": "Median (IQR)",
    "range": "Min–Max",
    "missing": "Missing, n (%)",
    "p01": "P1",
    "p05": "P5",
    "p95": "P95",
    "p99": "P99",
}


# Raw columns needed to calculate each displayed statistic
STATISTIC_REQUIRED_COLUMNS = {
    "n": {"n"},
    "unique_n": {"unique_n"},
    "mean_sd": {"mean", "std"},
    "median_quartiles": {"median", "q1", "q3"},
    "median_iqr": {"median", "iqr"},
    "range": {"min", "max"},
    "missing": {"missing_n", "missing_pct"},
    "p01": {"p01"},
    "p05": {"p05"},
    "p95": {"p95"},
    "p99": {"p99"},
}


# ---------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------

def create_journal_summary_table(
    summaries: (
        Mapping[str, pd.DataFrame]
        | Mapping[str, Mapping[str, pd.DataFrame]]
        | Sequence[Mapping[str, pd.DataFrame]]
    ),
    *,
    section_names: Sequence[str] | None = None,
    statistics: Sequence[str] = (
        "n",
        "mean_sd",
        "median_quartiles",
        "missing",
    ),
    dataset_order: Sequence[str] | None = None,
    title: str = "Summary of Continuous Variables",
    subtitle: str | None = None,
    decimals: int = 2,
    missing_symbol: str = "—",
) -> GT:
    """
    Create a publication-ready Great Tables table.

    Accepted input formats
    ----------------------

    1. A single dictionary:

        {
            "N0": n0_dataframe,
            "OCAT": ocat_dataframe,
            "Priyanka": priyanka_dataframe,
        }

    2. Multiple named dictionaries:

        {
            "GC characteristics": {
                "N0": ...,
                "OCAT": ...,
                "Priyanka": ...,
            },
            "Tumor characteristics": {
                "N0": ...,
                "OCAT": ...,
                "Priyanka": ...,
            },
        }

    3. A list of dictionaries:

        [
            summary_dictionary_1,
            summary_dictionary_2,
        ]

       In this case, section_names can be supplied.

    Parameters
    ----------
    summaries:
        Single dictionary, nested dictionary, or list of dictionaries.

    section_names:
        Names used when summaries is a list.

    statistics:
        Statistics to include. Supported values:

        - "n"
        - "unique_n"
        - "mean_sd"
        - "median_quartiles"
        - "median_iqr"
        - "range"
        - "missing"
        - "p01"
        - "p05"
        - "p95"
        - "p99"

    dataset_order:
        Optional dataset ordering, e.g.
        ["N0", "OCAT", "Priyanka"].

    Returns
    -------
    GT
        A Great Tables object.
    """

    if not summaries:
        raise ValueError("No summary dictionaries were supplied.")

    # --------------------------------------------------------------
    # Convert all accepted inputs to:
    #
    # {
    #     section_name: {
    #         dataset_name: dataframe
    #     }
    # }
    # --------------------------------------------------------------

    if isinstance(summaries, Sequence) and not isinstance(
        summaries,
        (str, bytes, Mapping),
    ):
        if section_names is None:
            section_names = [
                f"Section {index + 1}"
                for index in range(len(summaries))
            ]

        if len(section_names) != len(summaries):
            raise ValueError(
                "section_names must have the same length as summaries."
            )

        sections = {
            section_name: summary_dictionary
            for section_name, summary_dictionary
            in zip(section_names, summaries)
        }

    elif isinstance(summaries, Mapping):
        first_value = next(iter(summaries.values()))

        # Single dictionary: dataset -> DataFrame
        if isinstance(first_value, pd.DataFrame):
            sections = {"Characteristics": summaries}

        # Nested dictionary: section -> dataset -> DataFrame
        elif isinstance(first_value, Mapping):
            sections = summaries

        else:
            raise TypeError(
                "Dictionary values must be DataFrames or dictionaries "
                "containing DataFrames."
            )

    else:
        raise TypeError(
            "summaries must be a dictionary or a sequence of dictionaries."
        )

    # Validate requested statistics
    invalid_statistics = (
        set(statistics) - set(STATISTIC_LABELS)
    )

    if invalid_statistics:
        raise ValueError(
            "Unsupported statistics: "
            f"{sorted(invalid_statistics)}"
        )

    # --------------------------------------------------------------
    # Determine dataset order
    # --------------------------------------------------------------

    if dataset_order is None:
        dataset_order = []

        for section_dictionary in sections.values():
            for dataset_name in section_dictionary:
                if dataset_name not in dataset_order:
                    dataset_order.append(dataset_name)

    dataset_order = list(dataset_order)

    # Safe internal identifiers, avoiding problems with spaces/symbols
    dataset_ids = {
        dataset_name: f"dataset_{index + 1}"
        for index, dataset_name in enumerate(dataset_order)
    }

    records: list[dict] = []

    # Maps dataset names to their generated table columns
    dataset_columns: dict[str, list[str]] = {
        dataset_name: []
        for dataset_name in dataset_order
    }

    for dataset_name in dataset_order:
        dataset_id = dataset_ids[dataset_name]

        for statistic in statistics:
            dataset_columns[dataset_name].append(
                f"{dataset_id}__{statistic}"
            )

    # --------------------------------------------------------------
    # Process every section
    # --------------------------------------------------------------

    for section_name, section_dictionary in sections.items():

        prepared_frames: dict[str, pd.DataFrame] = {}
        variable_order: list[str] = []

        for dataset_name in dataset_order:
            if dataset_name not in section_dictionary:
                continue

            dataframe = section_dictionary[dataset_name].copy()

            if "col_name" not in dataframe.columns:
                raise ValueError(
                    f"Section {section_name!r}, dataset "
                    f"{dataset_name!r}: missing 'col_name' column."
                )

            # Columns needed for requested statistics
            required_columns = {"col_name"}

            for statistic in statistics:
                required_columns.update(
                    STATISTIC_REQUIRED_COLUMNS[statistic]
                )

            missing_columns = (
                required_columns - set(dataframe.columns)
            )

            if missing_columns:
                raise ValueError(
                    f"Section {section_name!r}, dataset "
                    f"{dataset_name!r} is missing columns: "
                    f"{sorted(missing_columns)}"
                )

            dataframe["_variable_key"] = (
                dataframe["col_name"]
                .map(_normalize_variable_name)
            )

            dataframe["_variable_label"] = dataframe["_variable_key"]

            # Detect duplicate rows after normalization
            duplicate_mask = dataframe["_variable_key"].duplicated(
                keep=False
            )

            if duplicate_mask.any():
                duplicate_names = (
                    dataframe.loc[
                        duplicate_mask,
                        "_variable_key",
                    ]
                    .unique()
                    .tolist()
                )

                raise ValueError(
                    f"Section {section_name!r}, dataset "
                    f"{dataset_name!r} contains duplicate variables "
                    f"after name normalization: {duplicate_names}"
                )

            dataframe = dataframe.set_index("_variable_key")

            prepared_frames[dataset_name] = dataframe

            # Preserve first-seen variable ordering
            for variable_name in dataframe.index:
                if variable_name not in variable_order:
                    variable_order.append(variable_name)

        # ----------------------------------------------------------
        # Build one output row per variable
        # ----------------------------------------------------------

        for variable_name in variable_order:
            output_row = {
                "_section": section_name,
                "_variable": variable_name,
            }

            for dataset_name in dataset_order:
                dataset_id = dataset_ids[dataset_name]
                dataframe = prepared_frames.get(dataset_name)

                if (
                    dataframe is None
                    or variable_name not in dataframe.index
                ):
                    for statistic in statistics:
                        output_row[
                            f"{dataset_id}__{statistic}"
                        ] = missing_symbol

                    continue

                raw_row = dataframe.loc[variable_name]

                for statistic in statistics:
                    output_row[
                        f"{dataset_id}__{statistic}"
                    ] = _format_statistic(
                        row=raw_row,
                        statistic=statistic,
                        decimals=decimals,
                        missing_symbol=missing_symbol,
                    )

            records.append(output_row)

    final_dataframe = pd.DataFrame(records)

    if final_dataframe.empty:
        raise ValueError("No rows were available to create the table.")

    value_columns = [
        column_name
        for dataset_name in dataset_order
        for column_name in dataset_columns[dataset_name]
    ]

    final_dataframe = final_dataframe[
        ["_section", "_variable", *value_columns]
    ]

    # Labels shown under each dataset spanner
    displayed_labels = {}

    for dataset_name in dataset_order:
        dataset_id = dataset_ids[dataset_name]

        for statistic in statistics:
            displayed_labels[
                f"{dataset_id}__{statistic}"
            ] = STATISTIC_LABELS[statistic]

    if subtitle is None:
        subtitle = (
            "Mean (SD) and median [Q1, Q3] are reported. "
            "n represents non-missing observations."
        )

    # --------------------------------------------------------------
    # Construct Great Tables object
    # --------------------------------------------------------------

    journal_table = (
        GT(
            final_dataframe,
            rowname_col="_variable",
            groupname_col="_section",
            auto_align=False,
        )
        .tab_header(
            title=md(f"**{title}**"),
            subtitle=subtitle,
        )
        # .tab_stubhead(
        #     label="Characteristic",
        # )
        .cols_label(
            **displayed_labels,
        )
        .cols_align(
            align="center",
            columns=value_columns,
        )
        .tab_options(
            table_width="100%",
            table_font_size="12px",
            data_row_padding="4px",
            data_row_padding_horizontal="7px",
            column_labels_font_weight="bold",
            column_labels_padding="6px",
            row_group_font_weight="bold",
            row_group_background_color="#F2F2F2",
            row_group_padding="6px",
            stub_font_weight="normal",
            table_body_hlines_style="solid",
            table_body_hlines_width="0.5px",
            table_body_hlines_color="#D9D9D9",
            table_body_vlines_style="none",
        )
    )

    # Add one spanner per dataset
    for dataset_name in dataset_order:
        journal_table = journal_table.tab_spanner(
            label=dataset_name,
            columns=dataset_columns[dataset_name],
            id=f"spanner_{dataset_ids[dataset_name]}",
        )

    return journal_table
