"""Dependency-light contracts used throughout Statomix."""

from .contracts import (
    AnalyzerInputPaths,
    ConfigRef,
    CuratedStateLineage,
    GroupBundle,
    GroupInfo,
    ProcedureState,
    ProcedureStatus,
    VersionRef,
)
from .errors import (
    AnalysisError,
    ArtifactNotFoundError,
    ContractError,
    CuratedStateInheritanceError,
    StatomixError,
    VersionSelectionError,
)
from .registry import Analysis, AnalysisRegistry
from .version_selection import ArtifactVersionSelection, resolve_artifact_version

__all__ = [
    "Analysis",
    "AnalysisError",
    "AnalysisRegistry",
    "ArtifactVersionSelection",
    "AnalyzerInputPaths",
    "ArtifactNotFoundError",
    "ConfigRef",
    "ContractError",
    "CuratedStateInheritanceError",
    "CuratedStateLineage",
    "GroupBundle",
    "GroupInfo",
    "ProcedureState",
    "ProcedureStatus",
    "StatomixError",
    "VersionRef",
    "VersionSelectionError",
    "resolve_artifact_version",
]
