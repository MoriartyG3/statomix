from __future__ import annotations

from collections.abc import Iterable, Mapping

import pandas as pd


def frame_from_rows(
    rows: Iterable[Mapping[str, object]],
    schema: Mapping[str, str],
) -> pd.DataFrame:
    """Build a dataframe that preserves its columns when ``rows`` is empty.

    Pandas cannot infer a tabular contract from an empty list. Artifact
    serializers therefore use this helper so a zero-row parquet file remains
    loadable through exactly the same code path as a populated artifact.
    """

    materialized_rows = list(rows)
    if materialized_rows:
        return pd.DataFrame(
            data=materialized_rows,
            columns=list(schema),
        )

    return pd.DataFrame(
        {column_name: pd.Series(dtype=dtype) for column_name, dtype in schema.items()}
    )
