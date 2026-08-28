"""Statomix-specific exceptions.

The subclasses retain compatibility with built-in exception handlers while
allowing callers to distinguish malformed data from missing artifacts and
version-selection failures.
"""


class StatomixError(Exception):
    """Base class for package-specific failures."""


class ContractError(StatomixError, ValueError):
    """Raised when an object violates a cross-module contract."""


class CuratedStateInheritanceError(ContractError):
    """Raised when a derived dataset cannot inherit a curated parent state."""


class ArtifactNotFoundError(StatomixError, FileNotFoundError):
    """Raised when a required persisted artifact is absent."""


class VersionSelectionError(StatomixError, ValueError):
    """Raised for ambiguous or invalid version/configuration selection."""


class AnalysisError(StatomixError):
    """Raised when an analysis cannot produce a scientifically valid result."""
