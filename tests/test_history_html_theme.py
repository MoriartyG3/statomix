"""Regression checks for the shared Statomix history-report theme."""

from __future__ import annotations

import json

from statomix.history.model import HistoryEdge, HistoryNode, ProjectHistory
from statomix.history.renderers.html import render_history_html


def _history() -> ProjectHistory:
    nodes = [
        HistoryNode(
            node_id="source:cohort",
            node_type="source",
            label="cohort",
            status="completed",
            dataset="cohort",
            display_label="Discovery cohort",
            dataset_role="analysis",
            pipeline="source",
        ),
        HistoryNode(
            node_id="analyzer:cohort:v1:c1",
            node_type="analyzer",
            label="analysis",
            status="completed",
            dataset="cohort",
            display_label="Discovery cohort",
            dataset_role="analysis",
            pipeline="analyzer",
        ),
    ]
    edges = [
        HistoryEdge(
            source="source:cohort",
            target="analyzer:cohort:v1:c1",
            relationship="analyzed_from",
        )
    ]
    return ProjectHistory(
        project_name="Theme test",
        nodes=nodes,
        edges=edges,
        warnings=[],
    )


def test_history_html_is_nonempty_and_uses_shared_theme(tmp_path):
    destination = tmp_path / "history.html"
    render_history_html(history=_history(), destination=destination)

    html = destination.read_text(encoding="utf-8")
    assert html.startswith("<!doctype html>")
    assert html.count("</html>") == 1
    assert "--stx-accent: #d8ff4f" in html
    assert "edge-label-bg" in html
    assert "Discovery cohort" in html
    assert 'id="history-data"' in html
    json.loads(
        html.split('id="history-data" type="application/json">', 1)[1].split(
            "</script>", 1
        )[0]
    )
