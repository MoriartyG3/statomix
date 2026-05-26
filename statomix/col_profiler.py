import re
import pandas as pd
from dataclasses import dataclass

from .semantic_rules import (
    DataTypes,
    DATETIME_REGEX,
    IDENTIFIER_REGEX,
    SEMANTIC_RULES,
    MULTI_TOKEN_RULES,
)


@dataclass
class ColProfile:
    col_name: str
    col_type: DataTypes | None
    missing_n: int
    missing_pct: float
    unique_n: int
    tokens: list[str]
    normalized_name: str


@dataclass
class RawColProfile:
    col_name: str
    col_type: DataTypes | None

    unique_n: int
    missing_n: int
    missing_pct: float
    num_conversion_pct: float | None
    is_datetime: bool | None


@dataclass
class SemanticProfile:
    col_name: str
    col_type: DataTypes | None

    score: float
    tokens: list[str]
    normalized_name: str
    matched_rules: list[str]
    all_scores: dict[DataTypes, float]


class ColProfiler:
    def __init__(self, cat_unique_thresh: int = 4, num_conversion_thresh: float = 90.0):
        self.cat_unique_thresh = cat_unique_thresh
        self.num_conversion_thresh = num_conversion_thresh

    def get_col_profile(self, col_name, col_series):
        col_profile_raw = self.get_raw_col_profile(
            col_name=col_name, col_series=col_series
        )
        col_profile_semantic = self.get_semantic_col_profile(col_name=col_name)

        if (
            col_profile_semantic.col_type is not None
            and col_profile_raw.col_type != DataTypes.DATETIME
        ):
            col_type = col_profile_semantic.col_type
        else:
            col_type = col_profile_raw.col_type

        col_profile = ColProfile(
            col_name=col_name,
            col_type=col_type,
            normalized_name=col_profile_semantic.normalized_name,
            tokens=col_profile_semantic.tokens,
            missing_n=col_profile_raw.missing_n,
            missing_pct=col_profile_raw.missing_pct,
            unique_n=col_profile_raw.unique_n,
        )

        return col_profile

    def get_raw_col_profile(
        self,
        col_name,
        col_series,
        # samples_n,
    ):

        samples_n = len(col_series)
        missing_n = col_series.isna().sum()

        missing_pct = round(
            number=(missing_n / samples_n) * 100,
            ndigits=2,
        )

        if missing_n == samples_n:
            col_type = None
            unique_n = None
            num_conversion_pct = None
            is_datetime = None

        else:
            col_series_non_null = col_series.dropna()
            unique_n = col_series_non_null.nunique()

            col_name_normalized = ColProfiler.normalize_col_name(name=col_name)

            is_datetime = bool(
                re.search(
                    pattern=DATETIME_REGEX,
                    string=col_name_normalized,
                    flags=re.IGNORECASE,
                )
            )

            if is_datetime:
                col_type = DataTypes.DATETIME
                num_conversion_pct = None
            else:
                num_converted = pd.to_numeric(
                    arg=col_series_non_null,
                    errors="coerce",
                )

                num_conversion_pct = round(
                    number=(num_converted.notna().sum() / len(col_series_non_null))
                    * 100,
                    ndigits=2,
                )

                if (
                    num_conversion_pct >= self.num_conversion_thresh
                    and unique_n <= self.cat_unique_thresh
                ):
                    col_type = DataTypes.CATEGORICAL

                elif unique_n == len(col_series_non_null) and bool(
                    re.search(
                        pattern=IDENTIFIER_REGEX,
                        string=col_name_normalized,
                        flags=re.IGNORECASE,
                    )
                ):
                    col_type = DataTypes.IDENTIFIER

                elif num_conversion_pct >= self.num_conversion_thresh:
                    col_type = DataTypes.NUMERICAL

                else:
                    col_type = DataTypes.CATEGORICAL

        col_profile_raw = RawColProfile(
            col_name=col_name,
            col_type=col_type,
            missing_n=missing_n,
            missing_pct=missing_pct,
            unique_n=unique_n,
            num_conversion_pct=num_conversion_pct,
            is_datetime=is_datetime,
        )

        return col_profile_raw

    @staticmethod
    def get_semantic_col_profile(col_name: str):

        scores: dict[str, float] = {}
        matched_rules: list[str] = []

        normalized_name = ColProfiler.normalize_col_name(col_name)
        tokens = ColProfiler.tokenize_col_name(col_name)
        token_set = set(tokens)

        for rule in SEMANTIC_RULES:
            overlap = token_set.intersection(rule.keywords)
            if overlap:

                scores[rule.semantic_type] = (
                    scores.get(rule.semantic_type, 0.0) + rule.score
                )

                matched_rules.append(f"{rule.semantic_type}:tokens={sorted(overlap)}")

        for left_group, right_group, semantic_type, score in MULTI_TOKEN_RULES:

            if token_set.intersection(left_group) and token_set.intersection(
                right_group
            ):

                scores[semantic_type] = scores.get(semantic_type, 0.0) + score

                matched_rules.append(f"{semantic_type}:multi_token_rule")

        if not scores:

            return SemanticProfile(
                col_name=col_name,
                col_type=None,
                score=0.0,
                matched_rules=[],
                normalized_name=normalized_name,
                tokens=tokens,
                all_scores={},
            )

        best_type = max(scores.items(), key=lambda x: x[1])

        return SemanticProfile(
            col_name=col_name,
            col_type=best_type[0],
            score=best_type[1],
            matched_rules=matched_rules,
            normalized_name=normalized_name,
            tokens=tokens,
            all_scores=scores,
        )

    @staticmethod
    def normalize_col_name(name: str) -> str:
        """
        Normalize column names into canonical snake_case.

        Examples
        --------
        patientID        -> patient_id
        OSMonths         -> os_months
        FollowUpDays     -> follow_up_days
        tumor-stage      -> tumor_stage
        Patient Number   -> patient_number
        """

        if not isinstance(name, str):
            name = str(name)

        # Strip whitespace
        name = name.strip()

        # Replace separators with underscore
        name = re.sub(r"[^\w]+", "_", name)

        # Handle camelCase / PascalCase
        # patientID -> patient_ID
        name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)

        # Handle ALLCAPS followed by CapitalLower
        # OSMonths -> OS_Months
        name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)

        # Lowercase
        name = name.lower()

        # Collapse repeated underscores
        name = re.sub(r"_+", "_", name)

        # Remove leading/trailing underscores
        name = name.strip("_")

        return name

    @staticmethod
    def tokenize_col_name(name: str) -> list[str]:
        """
        Convert normalized name into semantic tokens.

        Example
        -------
        follow_up_months -> ["follow", "up", "months"]
        """

        normalized = ColProfiler.normalize_col_name(name)

        tokens = [t for t in normalized.split("_") if t]

        return tokens
