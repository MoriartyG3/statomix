"""Backend-independent registry for analysis implementations."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar, runtime_checkable

ResultT = TypeVar("ResultT", covariant=True)


@runtime_checkable
class Analysis(Protocol[ResultT]):
    """Minimal protocol implemented by executable analyses."""

    def run(self) -> ResultT:
        """Execute the analysis and return its typed result."""


AnalysisFactory = Callable[..., Analysis[Any]]


@dataclass(slots=True)
class AnalysisRegistry:
    """Ordered mapping from stable analysis names to factories."""

    _factories: dict[str, AnalysisFactory] = field(default_factory=dict)

    def register(self, *, name: str, factory: AnalysisFactory) -> None:
        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("analysis name must not be empty")
        if normalized_name in self._factories:
            raise ValueError(f"analysis is already registered: {normalized_name}")
        self._factories[normalized_name] = factory

    def replace(self, *, name: str, factory: AnalysisFactory) -> None:
        normalized_name = name.strip()
        if normalized_name not in self._factories:
            raise KeyError(normalized_name)
        self._factories[normalized_name] = factory

    def create(self, *, name: str, **kwargs: Any) -> Analysis[Any]:
        try:
            factory = self._factories[name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown analysis {name!r}; available: {list(self._factories)}"
            ) from exc
        return factory(**kwargs)

    def names(self) -> tuple[str, ...]:
        return tuple(self._factories)

    def __iter__(self) -> Iterator[str]:
        return iter(self._factories)
