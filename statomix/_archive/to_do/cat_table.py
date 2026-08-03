from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
import unicodedata

import pandas as pd
from great_tables import GT, md


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def _normalize_label(value: object) -> str:
    """
    Normalize labels so that differences such as curly apostrophes,
    repeated spaces, or Unicode formatting do not create separate rows.
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

    value = re.sub(r"\s+", " ", value)

    return value


def _format_count(value: object) -> str | None:
    """Format count values as integers."""
    if pd.isna(value):
        return None

    numeric_value = float(value)

    if numeric_value.is_integer():
        return f"{int(numeric_value):,}"

    return f"{numeric_value:g}"


def _format_percentage(
    value: object,
    decimals: int,
) -> str | None:
    """Format percentage values."""
    if pd.isna(value):
        return None

    return f"{float(value):.{decimals}f}"


def _format_n_percentage(
    count: object,
    percentage: object,
    decimals: int,
    missing_symbol: str,
) -> str:
    """
    Format a count and percentage as:

        24 (13.19%)
    """
    count_text = _format_count(count)
    percentage_text = _format_percentage(
        percentage,
        decimals=decimals,
    )

    if count_text is None and percentage_text is None:
        return missing_symbol

    if count_text is None:
        return f"({percentage_text}%)"

    if percentage_text is None:
        return count_text

    return f"{count_text} ({percentage_text}%)"


def _prepare_categorical_dataframe(
    dataframe: pd.DataFrame,
    *,
    variable_col: str,
    category_col: str,
    count_col: str,
    percentage_col: str,
) -> pd.DataFrame:
    """
    Convert a categorical summary DataFrame into a standard internal
    format.

    Supports either:

    1. MultiIndex rows:
       index = ['col_name', 'category']

    2. Normal columns:
       columns include 'col_name' and 'category'
    """
    dataframe = dataframe.copy()

    # --------------------------------------------------------------
    # Handle MultiIndex input
    # --------------------------------------------------------------
    if isinstance(dataframe.index, pd.MultiIndex):

        if dataframe.index.nlevels < 2:
            raise ValueError(
                "The categorical DataFrame MultiIndex must have "
                "at least two levels."
            )

        index_dataframe = (
            dataframe.index
            .to_frame(index=False)
            .iloc[:, :2]
            .copy()
        )

        index_dataframe.columns = [
            variable_col,
            category_col,
        ]

        values_dataframe = dataframe.reset_index(drop=True)

        dataframe = pd.concat(
            [
                index_dataframe.reset_index(drop=True),
                values_dataframe.reset_index(drop=True),
            ],
            axis=1,
        )

    # --------------------------------------------------------------
    # Handle normal-column input
    # --------------------------------------------------------------
    else:
        required_index_columns = {
            variable_col,
            category_col,
        }

        missing_index_columns = (
            required_index_columns - set(dataframe.columns)
        )

        if missing_index_columns:
            raise ValueError(
                "The DataFrame must either have a two-level "
                "MultiIndex or contain the columns "
                f"{sorted(required_index_columns)}."
            )

    required_value_columns = {
        count_col,
        percentage_col,
    }

    missing_value_columns = (
        required_value_columns - set(dataframe.columns)
    )

    if missing_value_columns:
        raise ValueError(
            "Missing categorical summary columns: "
            f"{sorted(missing_value_columns)}"
        )

    # Useful when reading merged cells from Excel
    dataframe[variable_col] = dataframe[variable_col].ffill()

    # Normalize variable and category labels
    dataframe["_variable"] = (
        dataframe[variable_col]
        .map(_normalize_label)
    )

    dataframe["_category"] = (
        dataframe[category_col]
        .map(_normalize_label)
    )

    empty_variable = dataframe["_variable"].eq("")
    empty_category = dataframe["_category"].eq("")

    if empty_variable.any():
        raise ValueError(
            "One or more rows have an empty variable name."
        )

    if empty_category.any():
        raise ValueError(
            "One or more rows have an empty category name."
        )

    # Check for duplicate variable-category combinations
    duplicate_mask = dataframe.duplicated(
        subset=["_variable", "_category"],
        keep=False,
    )

    if duplicate_mask.any():
        duplicates = (
            dataframe.loc[
                duplicate_mask,
                ["_variable", "_category"],
            ]
            .drop_duplicates()
            .to_dict("records")
        )

        raise ValueError(
            "Duplicate variable-category combinations were found: "
            f"{duplicates}"
        )

    return dataframe[
        [
            "_variable",
            "_category",
            count_col,
            percentage_col,
        ]
    ].set_index(
        ["_variable", "_category"]
    )


# ---------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------

def create_journal_categorical_table(
    summaries: (
        Mapping[str, pd.DataFrame]
        | Mapping[str, Mapping[str, pd.DataFrame]]
        | Sequence[Mapping[str, pd.DataFrame]]
    ),
    *,
    section_names: Sequence[str] | None = None,
    dataset_order: Sequence[str] | None = None,
    variable_col: str = "col_name",
    category_col: str = "category",
    count_col: str = "count",
    percentage_col: str = "percentage",
    title: str = "Categorical Characteristics Across Cohorts",
    subtitle: str = "Values are presented as n (%).",
    cohort_spanner: str | None = "Cohort",
    percentage_decimals: int = 2,
    missing_symbol: str = "—",
    section_separator: str = " — ",
) -> GT:
    """
    Create a publication-ready Great Tables table for categorical data.

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
            "Clinicopathological characteristics": {
                "N0": ...,
                "OCAT": ...,
                "Priyanka": ...,
            },
            "Progression characteristics": {
                "N0": ...,
                "OCAT": ...,
                "Priyanka": ...,
            },
        }

    3. A list of dictionaries:

        [
            categorical_dictionary_1,
            categorical_dictionary_2,
        ]

       In this case, supply section_names.

    Each dataset DataFrame should contain:

        MultiIndex:
            col_name
            category

        Columns:
            count
            percentage

    Returns
    -------
    GT
        A Great Tables object.
    """

    if not summaries:
        raise ValueError(
            "No categorical summary dictionaries were supplied."
        )

    # --------------------------------------------------------------
    # Convert accepted inputs into:
    #
    # {
    #     section_name: {
    #         dataset_name: dataframe
    #     }
    # }
    # --------------------------------------------------------------

    if (
        isinstance(summaries, Sequence)
        and not isinstance(
            summaries,
            (str, bytes, Mapping),
        )
    ):
        if section_names is None:
            section_names = [
                f"Section {index + 1}"
                for index in range(len(summaries))
            ]

        if len(section_names) != len(summaries):
            raise ValueError(
                "section_names must have the same length "
                "as summaries."
            )

        sections = {
            section_name: summary_dictionary
            for section_name, summary_dictionary
            in zip(section_names, summaries)
        }

        multiple_sections = len(sections) > 1

    elif isinstance(summaries, Mapping):
        first_value = next(iter(summaries.values()))

        # Dataset name -> DataFrame
        if isinstance(first_value, pd.DataFrame):
            sections = {
                "Characteristics": summaries,
            }

            multiple_sections = False

        # Section name -> dataset dictionary
        elif isinstance(first_value, Mapping):
            sections = dict(summaries)
            multiple_sections = len(sections) > 1

        else:
            raise TypeError(
                "Dictionary values must be DataFrames or "
                "dictionaries containing DataFrames."
            )

    else:
        raise TypeError(
            "summaries must be a dictionary or a sequence "
            "of dictionaries."
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

    if not dataset_order:
        raise ValueError("No dataset names were found.")

    # Safe internal column names
    dataset_ids = {
        dataset_name: f"_dataset_{index + 1}"
        for index, dataset_name in enumerate(dataset_order)
    }

    output_records: list[dict] = []

    # --------------------------------------------------------------
    # Process every dictionary/section
    # --------------------------------------------------------------

    for section_name, section_dictionary in sections.items():

        prepared_datasets: dict[str, pd.DataFrame] = {}

        variable_order: list[str] = []
        category_order: dict[str, list[str]] = {}

        # ----------------------------------------------------------
        # Prepare each cohort DataFrame and determine row ordering
        # ----------------------------------------------------------

        for dataset_name in dataset_order:

            if dataset_name not in section_dictionary:
                continue

            prepared_dataframe = _prepare_categorical_dataframe(
                section_dictionary[dataset_name],
                variable_col=variable_col,
                category_col=category_col,
                count_col=count_col,
                percentage_col=percentage_col,
            )

            prepared_datasets[dataset_name] = prepared_dataframe

            for variable_name, category_name in prepared_dataframe.index:

                if variable_name not in variable_order:
                    variable_order.append(variable_name)

                if variable_name not in category_order:
                    category_order[variable_name] = []

                if category_name not in category_order[variable_name]:
                    category_order[variable_name].append(
                        category_name
                    )

        # ----------------------------------------------------------
        # Build one output row per variable-category pair
        # ----------------------------------------------------------

        for variable_name in variable_order:

            if multiple_sections:
                group_label = (
                    f"{section_name}"
                    f"{section_separator}"
                    f"{variable_name}"
                )
            else:
                group_label = variable_name

            for category_name in category_order[variable_name]:

                output_row = {
                    "_group": group_label,
                    "_category": category_name,
                }

                for dataset_name in dataset_order:
                    dataset_id = dataset_ids[dataset_name]

                    dataframe = prepared_datasets.get(dataset_name)

                    row_key = (
                        variable_name,
                        category_name,
                    )

                    if (
                        dataframe is None
                        or row_key not in dataframe.index
                    ):
                        output_row[dataset_id] = missing_symbol
                        continue

                    raw_row = dataframe.loc[row_key]

                    output_row[dataset_id] = (
                        _format_n_percentage(
                            count=raw_row[count_col],
                            percentage=raw_row[percentage_col],
                            decimals=percentage_decimals,
                            missing_symbol=missing_symbol,
                        )
                    )

                output_records.append(output_row)

    final_dataframe = pd.DataFrame(output_records)

    if final_dataframe.empty:
        raise ValueError(
            "No categorical rows were available for the table."
        )

    dataset_columns = [
        dataset_ids[dataset_name]
        for dataset_name in dataset_order
    ]

    final_dataframe = final_dataframe[
        [
            "_group",
            "_category",
            *dataset_columns,
        ]
    ]

    displayed_labels = {
        dataset_ids[dataset_name]: dataset_name
        for dataset_name in dataset_order
    }

    # --------------------------------------------------------------
    # Create Great Tables object
    # --------------------------------------------------------------

    journal_table = (
        GT(
            final_dataframe,
            rowname_col="_category",
            groupname_col="_group",
            auto_align=False,
        )
        .tab_header(
            title=md(f"**{title}**"),
            subtitle=subtitle,
        )
        .tab_stubhead(
            label="Variable / Category",
        )
        .cols_label(
            **displayed_labels,
        )
        .cols_align(
            align="center",
            columns=dataset_columns,
        )
        .tab_options(
            table_width="100%",
            table_font_size="12px",
            data_row_padding="4px",
            column_labels_padding="6px",
            column_labels_font_weight="bold",
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

    if cohort_spanner and len(dataset_columns) > 1:
        journal_table = journal_table.tab_spanner(
            label=cohort_spanner,
            columns=dataset_columns,
            id="cohort_spanner",
        )

    return journal_table
