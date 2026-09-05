"""Atomic creation of external project-history report bundles."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from tempfile import mkdtemp

from statomix.core.artifacts import canonical_json
from statomix.history.discovery import discover_project_history
from statomix.history.model import ProjectHistory
from statomix.history.renderers import (
    render_history_excel,
    render_history_html,
    render_history_svg,
)


def _slug(value: str) -> str:
    characters = [
        character.casefold() if character.isalnum() else "-"
        for character in value.strip()
    ]
    result = "".join(characters)
    while "--" in result:
        result = result.replace("--", "-")
    return result.strip("-") or "statomix-project"


@dataclass(frozen=True, slots=True, kw_only=True)
class HistoryReport:
    """Paths and graph for one content-addressed history report."""

    directory: Path
    html_path: Path
    svg_path: Path
    json_path: Path
    audit_path: Path
    graph: ProjectHistory

    @property
    def history_id(self) -> str:
        return self.graph.history_id


def _report(
    *,
    directory: Path,
    history: ProjectHistory,
) -> HistoryReport:
    return HistoryReport(
        directory=directory,
        html_path=directory / "history.html",
        svg_path=directory / "history.svg",
        json_path=directory / "history.json",
        audit_path=directory / "history_audit.xlsx",
        graph=history,
    )


def _payload(history: ProjectHistory) -> dict:
    payload = history.to_dict()
    payload["history_id"] = history.history_id
    return payload


def create_history_report(
    *,
    project,
    output_dir: str | Path,
    verify_checksums: bool = True,
    include_files: bool = False,
) -> HistoryReport:
    """Discover and render project history outside the project store."""

    project_root = (Path(project.project_dir) / project.project_name).resolve()
    output_root = Path(output_dir).expanduser().resolve()

    if output_root == project_root or output_root.is_relative_to(project_root):
        raise ValueError(
            "History reports must be written outside the Statomix project store."
        )

    history = discover_project_history(
        project=project,
        verify_checksums=verify_checksums,
        include_files=include_files,
    )
    destination = output_root / (
        f"{_slug(project.project_name)}-history-{history.history_id[:12]}"
    )
    report = _report(directory=destination, history=history)
    expected_paths = (
        report.html_path,
        report.svg_path,
        report.json_path,
        report.audit_path,
    )

    if destination.exists():
        if not all(path.is_file() for path in expected_paths):
            raise FileExistsError(
                f"Incomplete history report already exists at {destination}."
            )
        stored = json.loads(report.json_path.read_text(encoding="utf-8"))
        if stored != _payload(history):
            raise ValueError(
                "Existing history report content does not match its history ID."
            )
        return report

    output_root.mkdir(parents=True, exist_ok=True)
    stage = Path(mkdtemp(prefix=".statomix-history-", dir=output_root))
    try:
        staged = _report(directory=stage, history=history)
        staged.json_path.write_text(
            canonical_json(_payload(history)),
            encoding="utf-8",
        )
        render_history_svg(history=history, destination=staged.svg_path)
        render_history_html(history=history, destination=staged.html_path)
        render_history_excel(history=history, destination=staged.audit_path)
        os.rename(stage, destination)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise

    return report
