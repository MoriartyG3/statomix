"""Small serialization adapters used by workflow code."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from fileverse.formats.yaml import BaseYAML

from statomix.core.contracts import AnalyzerInputPaths
from statomix.storage.atomic import atomic_output_path


def save_analyzer_input_paths(*, paths: AnalyzerInputPaths, destination: Path) -> None:
    with atomic_output_path(destination=Path(destination)) as temporary_path:
        BaseYAML.save(
            data=paths.as_dict(stringify=True),
            path=temporary_path,
            replace=True,
        )


def load_analyzer_input_paths(*, source: Path) -> AnalyzerInputPaths:
    serialized: Mapping[str, str] = BaseYAML.load(path=Path(source))
    return AnalyzerInputPaths.from_mapping(serialized)
