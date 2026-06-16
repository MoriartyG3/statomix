from __future__ import annotations

import re
from enum import Enum
from dataclasses import dataclass


class SurvivalDataTypes(Enum):
    EVENT = "Event"
    TIME = "Time"
    DATETIME_FROM = "DateTime From"
    DATETIME_TO = "DateTime To"


@dataclass
class SurvivalSemanticRule:
    semantic_type: SurvivalDataTypes
    keywords: set[str]
    score: float


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

SURVIVAL_TIME_REGEX = (
    r"(?:os|pfs|dfs|rfs|efs|ttf|ttp|ttr|ttd|lfs)"   # endpoint prefix
    r"|(?:time|duration|follow|fu|followup)"            # generic time words
    r"|(?:months?|days?|weeks?|years?)"                 # unit words
)

SURVIVAL_EVENT_REGEX = (
    r"(?:event|status|censor(?:ed)?|outcome|indicator|flag)"
    r"|(?:death|progression|relapse|recurrence|failure|response)"
    r"|(?:alive|dead|deceased)"
)

SURVIVAL_DATETIME_FROM_REGEX = (
    r"(?:start|begin|baseline|index|enrol(?:l)?|rand(?:omis(?:ation|ed))?)"
    r"|(?:diagnosis|dx|surgery|treatment|therapy|registration)"
    r"|(?:date_from|from_date|start_date|date_start|date_of)"
)

SURVIVAL_DATETIME_TO_REGEX = (
    r"(?:end|last|final|contact|death|progression|relapse)"
    r"|(?:date_to|to_date|end_date|date_end|date_last|last_date)"
    r"|(?:censor(?:ed)?|follow(?:_?up)?|fu)"
)


# ---------------------------------------------------------------------------
# Semantic rules
# ---------------------------------------------------------------------------

SURVIVAL_SEMANTIC_RULES: list[SurvivalSemanticRule] = [
    # -----------------------------------------------------------------------
    # TIME — numeric duration columns (e.g. os_months, pfs_days, fu_weeks)
    # -----------------------------------------------------------------------
    SurvivalSemanticRule(
        semantic_type=SurvivalDataTypes.TIME,
        keywords={
            "os", "pfs", "dfs", "rfs", "efs",   # clinical endpoint prefixes
            "ttf", "ttp", "ttr", "ttd", "lfs",  # time-to-* prefixes
            "time", "duration",
            "follow", "followup", "fu",
            "months", "month",
            "days", "day",
            "weeks", "week",
            "years", "year",
        },
        score=0.30,
    ),

    # -----------------------------------------------------------------------
    # EVENT — binary / censoring indicator columns (e.g. os_event, pfs_status)
    # -----------------------------------------------------------------------
    SurvivalSemanticRule(
        semantic_type=SurvivalDataTypes.EVENT,
        keywords={
            "event", "events",
            "status",
            "censor", "censored",
            "outcome",
            "indicator", "flag",
            "death",
            "progression",
            "relapse", "recurrence",
            "failure",
            "response",
            "alive", "dead", "deceased",
        },
        score=0.30,
    ),

    # -----------------------------------------------------------------------
    # DATETIME_FROM — start / anchor date columns
    # -----------------------------------------------------------------------
    SurvivalSemanticRule(
        semantic_type=SurvivalDataTypes.DATETIME_FROM,
        keywords={
            "start", "begin", "baseline",
            "index",
            "enrol", "enroll", "enrollment",
            "rand", "randomisation", "randomization", "randomised", "randomized",
            "registration",
            "diagnosis", "dx",
            "surgery",
            "treatment", "therapy",
            "date_from", "from_date",
            "start_date", "date_start",
            "date_of",
        },
        score=0.35,
    ),

    # -----------------------------------------------------------------------
    # DATETIME_TO — end / last-contact date columns
    # -----------------------------------------------------------------------
    SurvivalSemanticRule(
        semantic_type=SurvivalDataTypes.DATETIME_TO,
        keywords={
            "end", "last", "final",
            "contact", "last_contact",
            #"death",
            #"event",
            "progression",
            #"relapse",
            "date_to", "to_date",
            "end_date", "date_end",
            "date_last", "last_date",
            "censored", "censor",
            "follow", "followup", "fu",
        },
        score=0.35,
    ),
]


# ---------------------------------------------------------------------------
# Multi-token rules  (left_token × right_token → strong signal)
# ---------------------------------------------------------------------------

SURVIVAL_MULTI_TOKEN_RULES = [
    # endpoint_prefix + time_unit  →  TIME  (e.g. "os_months", "pfs_days")
    (
        {"os", "pfs", "dfs", "rfs", "efs", "ttf", "ttp", "ttr", "ttd", "lfs"},  # Left
        {"months", "month", "days", "day", "weeks", "week", "years", "year",
         "time", "duration"},                                                      # Right
        SurvivalDataTypes.TIME,                                                    # Type
        0.50,                                                                      # Score
    ),

    # endpoint_prefix + event_word  →  EVENT  (e.g. "os_event", "pfs_status")
    (
        {"os", "pfs", "dfs", "rfs", "efs", "ttf", "ttp", "ttr", "ttd", "lfs"},  # Left
        {"event", "events", "status", "censor", "censored", "outcome",
         "indicator", "flag", "death", "progression", "relapse"},                 # Right
        SurvivalDataTypes.EVENT,                                                   # Type
        0.50,                                                                      # Score
    ),

    # "date" + start_anchor  →  DATETIME_FROM  (e.g. "date_diagnosis", "date_start")
    (
        {"date", "dt"},                                                            # Left
        {"start", "begin", "baseline", "diagnosis", "dx", "surgery",
         "treatment", "therapy", "enrol", "enroll", "rand", "registration"},      # Right
        SurvivalDataTypes.DATETIME_FROM,                                           # Type
        0.50,                                                                      # Score
    ),

    # "date" + end_anchor  →  DATETIME_TO  (e.g. "date_death", "date_last_contact")
    (
        {"date", "dt"},                                                            # Left
        {"death", "last", "end", "contact", "progression",
         "relapse", "censor", "censored", "fu", "followup"},                      # Right
        SurvivalDataTypes.DATETIME_TO,                                             # Type
        0.50,                                                                      # Score
    ),
]