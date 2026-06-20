from dataclasses import dataclass, asdict

import pandas as pd


@dataclass(frozen=True)
class CategoricalSummary:
    name: str|None
    n: int

    missing_n: int
    missing_pct: float

    unique_n: int

    mode: str | int | float | None
    mode_count: int | None
    mode_pct: float | None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


class BaseCategorical:

    @staticmethod
    def get_summary(
        series: pd.Series,
    ) -> CategoricalSummary:

        samples_n = len(series)

        missing_n = int(series.isna().sum())

        missing_pct = (
            round((missing_n / samples_n) * 100, 2)
            if samples_n > 0
            else 0.0
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

        mode = counts.index[0]
        mode_count = int(counts.iloc[0])

        mode_pct = round(
            (mode_count / n) * 100,
            2,
        )

        return CategoricalSummary(
            name=series.name,
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
    
        distribution_df = (
            series_non_null
            .value_counts(dropna=False)
            .rename("count")
            .reset_index()
        )
    
        distribution_df.columns = ["category", "count"]
    
        distribution_df["percentage"] = (
            distribution_df["count"]
            / distribution_df["count"].sum()
            * 100
        ).round(2)
    
        distribution_df = (
            distribution_df
            .sort_values(by="category", kind="stable")
            .reset_index(drop=True)
        )
    
        return distribution_df