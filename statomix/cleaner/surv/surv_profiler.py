from __future__ import annotations

import re
import pandas as pd
from dataclasses import dataclass

from statomix.cleaner.col.col_profiler import ColProfiler

#from .col_semantic_rules import DataTypes, DATETIME_REGEX
from .surv_semantic_rules import (
    SurvivalDataTypes,
    SurvivalSemanticRule,
    SURVIVAL_SEMANTIC_RULES,
    SURVIVAL_MULTI_TOKEN_RULES,
    # SURVIVAL_DATETIME_FROM_REGEX,
    # SURVIVAL_DATETIME_TO_REGEX,
    # SURVIVAL_EVENT_REGEX,
    # SURVIVAL_TIME_REGEX,
)


@dataclass
class SurvivalSemanticProfile:
    col_name: str
    col_type: SurvivalDataTypes | None
    score: float
    tokens: list[str]
    normalized_name: str
    matched_rules: list[str]
    all_scores: dict

    def to_dict(self) -> dict[str, object]:
        return {
            "col_name": self.col_name,
            "col_type": (
                self.col_type.value
                if self.col_type is not None
                else None
            ),
            "score": self.score,
            "tokens": "|".join(self.tokens),
            "normalized_name": self.normalized_name,
            "matched_rules": "|".join(self.matched_rules),
            "all_scores": {
                k.value: v for k, v in self.all_scores.items()
            },
        }

    @classmethod
    def from_dict(cls, row: pd.Series | dict) -> "SurvivalSemanticProfile":
        return cls(
            col_name=row["col_name"],
            col_type=(
                SurvivalDataTypes(row["col_type"])
                if pd.notna(row["col_type"])
                else None
            ),
            score=float(row["score"]),
            tokens=(
                row["tokens"].split("|")
                if pd.notna(row["tokens"]) and row["tokens"]
                else []
            ),
            normalized_name=row["normalized_name"],
            matched_rules=(
                row["matched_rules"].split("|")
                if pd.notna(row["matched_rules"]) and row["matched_rules"]
                else []
            ),
            all_scores=(
                {
                    SurvivalDataTypes(k): v
                    for k, v in row["all_scores"].items()
                }
                if isinstance(row["all_scores"], dict)
                else {}
            ),
        )

def get_survival_sematic_col_profile(col_name):
    scores: dict[SurvivalDataTypes, float] = {}
    matched_rules: list[str] = []

    normalized_name = ColProfiler.normalize_col_name(col_name)
    tokens = ColProfiler.tokenize_col_name(col_name)
    token_set = set(tokens)

    # Single-token rules
    for rule in SURVIVAL_SEMANTIC_RULES:
        overlap = token_set.intersection(rule.keywords)
        if overlap:
            scores[rule.semantic_type] = (
                scores.get(rule.semantic_type, 0.0) + rule.score
            )
            matched_rules.append(
                f"{rule.semantic_type}:tokens={sorted(overlap)}"
            )
    # Multi-token rules
    for left_group, right_group, semantic_type, score in SURVIVAL_MULTI_TOKEN_RULES:
        if token_set.intersection(left_group) and token_set.intersection(right_group):
            scores[semantic_type] = scores.get(semantic_type, 0.0) + score
            matched_rules.append(f"{semantic_type}:multi_token_rule")

    if not scores:
        return SurvivalSemanticProfile(
            col_name=col_name,
            col_type=None,
            score=0.0,
            matched_rules=[],
            normalized_name=normalized_name,
            tokens=tokens,
            all_scores={},
        )

    best_type, best_score = max(scores.items(), key=lambda x: x[1])

    return SurvivalSemanticProfile(
        col_name=col_name,
        col_type=best_type,
        score=best_score,
        matched_rules=matched_rules,
        normalized_name=normalized_name,
        tokens=tokens,
        all_scores=scores,
    )