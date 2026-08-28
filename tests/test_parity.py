from __future__ import annotations

import json
from copy import copy

from openpyxl import Workbook

from statomix.testing.parity import compare_artifact_trees


def _write_workbook(path, *, value: str) -> None:
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "Summary"
    worksheet["A1"] = value
    font = copy(worksheet["A1"].font)
    font.bold = True
    worksheet["A1"].font = font
    workbook.save(path)


def test_identical_semantic_artifacts_match(tmp_path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    reference.mkdir()
    candidate.mkdir()

    for root in (reference, candidate):
        (root / "meta.json").write_text(
            json.dumps({"status": "completed", "count": 3}),
            encoding="utf-8",
        )
        _write_workbook(root / "summary.xlsx", value="same")

    report = compare_artifact_trees(
        reference=reference,
        candidate=candidate,
    )

    assert report.matches
    assert report.compared_files == 2


def test_content_and_inventory_differences_are_reported(tmp_path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    reference.mkdir()
    candidate.mkdir()
    (reference / "value.json").write_text('{"value": 1}', encoding="utf-8")
    (candidate / "value.json").write_text('{"value": 2}', encoding="utf-8")
    (candidate / "extra.txt").write_text("extra", encoding="utf-8")

    report = compare_artifact_trees(
        reference=reference,
        candidate=candidate,
    )

    assert not report.matches
    assert {(item.path.as_posix(), item.kind) for item in report.differences} == {
        ("extra.txt", "extra"),
        ("value.json", "content"),
    }
