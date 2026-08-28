"""Dependency-light contracts used throughout Statomix."""

from .contracts import (
    AnalyzerInputPaths,
    ConfigRef,
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
    "GroupBundle",
    "GroupInfo",
    "ProcedureState",
    "ProcedureStatus",
    "StatomixError",
    "VersionRef",
    "VersionSelectionError",
    "resolve_artifact_version",
]
