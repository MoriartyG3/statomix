"""Boolean audit compatibility and numerical-summary regression tests."""

from dataclasses import replace

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from statomix.curation.columns.audit import ColumnAudit
from statomix.curation.columns.profiler import ColProfiler
from statomix.curation.columns.semantic_rules import DataTypes


@pytest.mark.parametrize(
    ("values", "dtype"),
    [
        ([False, True, False], "bool"),
        ([False, True, pd.NA], "boolean"),
        ([pd.NA, pd.NA], "boolean"),
        ([], "boolean"),
    ],
)
@pytest.mark.parametrize(
    "semantic_type",
    [
        DataTypes.SURVIVAL,
        DataTypes.CATEGORICAL,
    ],
)
def test_boolean_audit_preserves_values_without_quantiles(
    values,
    dtype,
    semantic_type,
):
    """Boolean auditing must preserve data and categorical-count policy."""

    column_name = "Event"
    series = pd.Series(values, dtype=dtype, name=column_name)
    source_df = series.to_frame()
    original_df = source_df.copy(deep=True)

    profiler = ColProfiler()
    inferred_profile = profiler.get_col_profile(
        col_name=column_name,
        col_series=series,
    )
    column_profile = replace(
        inferred_profile,
        col_type=semantic_type,
    )

    audit = ColumnAudit.from_dataframe(
        df=source_df,
        col_profiles={column_name: column_profile},
    )
    result = audit.profiles[column_name]

    assert result.source_dtype == dtype
    assert result.missing_n == int(series.isna().sum())
    # The profiler leaves unique_n unset when no observed values exist.
    # The audit must preserve that convention.
    nonmissing_series = series.dropna()

    if nonmissing_series.empty:
        expected_unique_n = None
    else:
        expected_unique_n = int(nonmissing_series.nunique())

    assert column_profile.unique_n == expected_unique_n
    assert result.unique_n == expected_unique_n

    for field_name in ("minimum", "q1", "median", "q3", "maximum"):
        assert getattr(result, field_name) is None

    # Keep the existing meaning of conversion diagnostics unchanged.
    if column_profile.num_conversion_pct is None:
        expected_numeric_n = 0
    else:
        converted = pd.to_numeric(series.dropna(), errors="coerce")
        expected_numeric_n = int(converted.notna().sum())

    assert result.numeric_n == expected_numeric_n
    assert result.nonnumeric_n == (int(series.notna().sum()) - expected_numeric_n)

    expected_counts = semantic_type == DataTypes.CATEGORICAL
    assert result.exact_value_counts_included == expected_counts

    if expected_counts:
        reported_count = sum(frequency.count for frequency in audit.value_frequencies)
        assert reported_count == len(series)
    else:
        assert audit.value_frequencies == ()

    assert_frame_equal(
        source_df,
        original_df,
        check_dtype=True,
        check_exact=True,
    )


def test_numeric_audit_still_calculates_quantiles():
    """The Boolean guard must not suppress genuine numerical summaries."""

    column_name = "Duration"
    series = pd.Series(
        [0.0, 10.0, 20.0, 30.0, 40.0],
        name=column_name,
    )
    source_df = series.to_frame()
    original_df = source_df.copy(deep=True)

    profiler = ColProfiler()
    column_profile = profiler.get_col_profile(
        col_name=column_name,
        col_series=series,
    )

    audit = ColumnAudit.from_dataframe(
        df=source_df,
        col_profiles={column_name: column_profile},
    )
    result = audit.profiles[column_name]

    assert result.numeric_n == 5
    assert result.nonnumeric_n == 0
    assert (
        result.minimum,
        result.q1,
        result.median,
        result.q3,
        result.maximum,
    ) == pytest.approx((0.0, 10.0, 20.0, 30.0, 40.0))

    assert_frame_equal(
        source_df,
        original_df,
        check_dtype=True,
        check_exact=True,
    )
