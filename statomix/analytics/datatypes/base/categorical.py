import pandas as pd
from dataclasses import dataclass


@dataclass
class CategoricalSummary:

    n: int

    missing_n: int
    missing_pct: float

    unique_n: int

    mode: str | None
    mode_count: int | None
    mode_pct: float | None

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


class BaseCategorical:

    def __init__(self):
        pass

    @staticmethod
    def get_summary(
        series: pd.Series,
    ) -> CategoricalSummary:

        samples_n = len(series)

        missing_n = int(series.isna().sum())

        missing_pct = round(
            (missing_n / samples_n) * 100,
            2,
        )

        series_non_null = series.dropna()

        n = len(series_non_null)

        if n == 0:
            return CategoricalSummary(
                n=0,
                missing_n=missing_n,
                missing_pct=missing_pct,
                unique_n=0,
                mode=None,
                mode_count=None,
                mode_pct=None,
            )

        counts = series_non_null.value_counts()

        mode = str(counts.index[0])

        mode_count = int(counts.iloc[0])

        mode_pct = round(
            (mode_count / n) * 100,
            2,
        )

        return CategoricalSummary(
            n=n,
            missing_n=missing_n,
            missing_pct=missing_pct,
            unique_n=int(series_non_null.nunique()),
            mode=mode,
            mode_count=mode_count,
            mode_pct=mode_pct,
        )

    @staticmethod
    def get_distribution_df(
        series: pd.Series,
    ) -> pd.DataFrame:

        series_non_null = series.dropna()

        counts = series_non_null.value_counts()

        pcts = (
            series_non_null.value_counts(normalize=True)
            * 100
        )

        summary_df = pd.DataFrame(
            {
                "category": counts.index,
                "count": counts.values,
                "percentage": pcts.values,
            }
        )

        summary_df = (
            summary_df
            .sort_values(by="category")
            .reset_index(drop=True)
        )

        return summary_df