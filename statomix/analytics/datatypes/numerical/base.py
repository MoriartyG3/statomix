import pandas as pd


class BaseNumerical:

    @staticmethod
    def _safe_float(value) -> float | None:
        return None if pd.isna(value) else float(value)

    @staticmethod
    def get_summary(
        series: pd.Series,
    ) -> dict:

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
            return {
                "n": 0,
                "unique_n": 0,
                "missing_n": missing_n,
                "missing_pct": missing_pct,
                "mean": None,
                "std": None,
                "q1": None,
                "median": None,
                "q3": None,
                "iqr": None,
                "p01": None,
                "p05": None,
                "p95": None,
                "p99": None,
                "min": None,
                "max": None,
                "skew": None,
                "kurtosis": None,
                "outlier_n": None,
                "outlier_pct": None,
            }

        q_vals = series_non_null.quantile(
            [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
        )

        p01, p05, q1, median, q3, p95, p99 = q_vals.values

        iqr = q3 - q1

        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)

        outlier_mask = (
            (series_non_null < lower_bound)
            | (series_non_null > upper_bound)
        )

        outlier_n = int(outlier_mask.sum())

        outlier_pct = round(
            (outlier_n / n) * 100,
            2,
        )

        stats = series_non_null.agg(
            [
                "mean",
                "std",
                "min",
                "max",
                "skew",
            ]
        )

        kurtosis = series_non_null.kurt()

        return {
            "n": n,
            "unique_n": int(series_non_null.nunique()),
            "missing_n": missing_n,
            "missing_pct": missing_pct,
            "mean": BaseNumerical._safe_float(stats["mean"]),
            "std": BaseNumerical._safe_float(stats["std"]),
            "q1": float(q1),
            "median": float(median),
            "q3": float(q3),
            "iqr": float(iqr),
            "p01": float(p01),
            "p05": float(p05),
            "p95": float(p95),
            "p99": float(p99),
            "min": BaseNumerical._safe_float(stats["min"]),
            "max": BaseNumerical._safe_float(stats["max"]),
            "skew": BaseNumerical._safe_float(stats["skew"]),
            "kurtosis": BaseNumerical._safe_float(kurtosis),
            "outlier_n": outlier_n,
            "outlier_pct": outlier_pct,
        }