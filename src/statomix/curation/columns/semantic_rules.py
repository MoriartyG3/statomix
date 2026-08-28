"""Rules for semantic column datatype inference."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DataTypes(Enum):
    IDENTIFIER = "Identifier"
    NUMERICAL = "Numerical"
    CATEGORICAL = "Categorical"
    SURVIVAL = "Survival"
    DATETIME = "DateTime"
    FREE_TEXT = "Free Text"


@dataclass(frozen=True, slots=True, kw_only=True)
class SemanticRule:
    semantic_type: DataTypes
    keywords: frozenset[str]
    score: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "keywords", frozenset(self.keywords))


DATETIME_REGEX = r"date|time|timestamp|dob"

IDENTIFIER_REGEX = r"(?:^id$|identifier|code|record|" r".*_id$|[a-z0-9]+id$)"
# IDENTIFIER_REGEX = r"id|identifier|code|record" #Need to fix this at some point of time
# IDENTIFIER_REGEX = r"\b(id|identifier|code|record)\b"

SEMANTIC_RULES: list[SemanticRule] = [
    # -----------------------------------------------------
    # IDENTIFIERS
    # -----------------------------------------------------
    SemanticRule(
        semantic_type=DataTypes.IDENTIFIER,
        keywords={
            "id",
            "identifier",
            "patient_id",
            "sample_id",
            "subject_id",
            "code",
            "number",
            "num",
            "record",
        },
        score=0.35,
    ),
    # -----------------------------------------------------
    # DATETIME
    # -----------------------------------------------------
    SemanticRule(
        semantic_type=DataTypes.DATETIME,
        keywords={
            "date",
            "time",
            "datetime",
            "timestamp",
            "year",
            "month",
            "day",
            "week",
            "dob",
            "created",
            "updated",
            "admission",
            "discharge",
        },
        score=0.25,
    ),
    # -----------------------------------------------------    CATEGORICAL
    # SURVIVAL TIME
    # -----------------------------------------------------
    SemanticRule(
        semantic_type=DataTypes.SURVIVAL,
        keywords={
            "os",
            "pfs",
            "dfs",
            "rfs",
            "efs",
            "follow",
            "followup",
            "fu",
            "months",
            "month",
            "days",
            "day",
            "weeks",
            "week",
            "years",
            "year",
            "duration",
            "event",
            "events",
            "status",
            "censor",
            "censored",
            "outcome",
            "death",
            "progression",
            "relapse",
            "alive",
            "dead",
        },
        score=0.30,
    ),
    # -----------------------------------------------------
    # EVENT / CENSOR
    # -----------------------------------------------------
    # SemanticRule(
    #     semantic_type=DataTypes.SURVIVAL,
    #     keywords={"event", "status", "censor", "censored", "outcome"},
    #     score=0.30,
    # ),
    # -----------------------------------------------------
    # NUMERIC
    # -----------------------------------------------------
    SemanticRule(
        semantic_type=DataTypes.NUMERICAL,
        keywords={
            "age",
            "score",
            "count",
            "pct",
            "percent",
            "rate",
            "ratio",
            "index",
            "value",
            "level",
            "concentration",
            "titre",
            "titer",
        },
        score=0.20,
    ),
    # -----------------------------------------------------
    # CATEGORICAL
    # -----------------------------------------------------
    SemanticRule(
        semantic_type=DataTypes.CATEGORICAL,
        keywords={
            "sex",
            "gender",
            "stage",
            "grade",
            "arm",
            "group",
            "cohort",
            "class",
            "type",
            "category",
            "subtype",
            "treatment",
            "therapy",
            "histology",
            "diagnosis",
            "site",
        },
        score=0.25,
    ),
    SemanticRule(
        semantic_type=DataTypes.FREE_TEXT,
        keywords={
            "note",
            "comment",
            "description",
            "remark",
            "narrative",
            "summary",
            "text",
        },
        score=0.35,
    ),
]

MULTI_TOKEN_RULES = [
    # OS_MONTHS
    (
        {"os", "pfs", "dfs", "rfs", "efs"},  # Left Group
        {
            "months",
            "month",
            "days",
            "day",
            "weeks",
            "week",
            "years",
            "year",
            "time",
            "duration",
        },  # Right Group
        DataTypes.SURVIVAL,  # Semantic Type
        0.45,  # Score
    ),
    # OS_EVENT
    (
        {"os", "pfs", "dfs", "rfs", "efs"},  # Left Group
        {
            "event",
            "events",
            "status",
            "censor",
            "censored",
            "outcome",
        },  # Right Group
        DataTypes.SURVIVAL,  # Semantic Type
        0.45,  # Score
    ),
]
