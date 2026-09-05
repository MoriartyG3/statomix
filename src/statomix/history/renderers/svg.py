"""Deterministic, dependency-free SVG lineage rendering."""

from __future__ import annotations

from collections import defaultdict
from html import escape
from pathlib import Path

from statomix.history.model import ProjectHistory

STAGES = {
    "source": 0,
    "cleaner": 1,
    "reference": 1,
    "transformer": 2,
    "analyzer": 3,
    "report": 4,
}
COLORS = {
    "source": "#22221e",
    "cleaner": "#3f6f85",
    "reference": "#7c5bb5",
    "transformer": "#3c8066",
    "analyzer": "#a96720",
    "report": "#77766d",
}


def _layout(history: ProjectHistory):
    datasets = sorted({node.dataset or "Project" for node in history.nodes})
    grouped = defaultdict(list)
    for node in history.nodes:
        grouped[node.dataset or "Project"].append(node)

    positions = {}
    lane_bounds = {}
    y_cursor = 70
    for dataset in datasets:
        by_stage = defaultdict(list)
        for node in grouped[dataset]:
            by_stage[STAGES.get(node.node_type, 2)].append(node)
        maximum = max((len(values) for values in by_stage.values()), default=1)
        lane_height = max(150, 55 + maximum * 92)
        lane_bounds[dataset] = (y_cursor, lane_height)
        for stage, nodes in by_stage.items():
            for index, node in enumerate(sorted(nodes, key=lambda item: item.node_id)):
                positions[node.node_id] = (
                    190 + stage * 270,
                    y_cursor + 48 + index * 92,
                )
        y_cursor += lane_height + 18
    display_labels = {}
    for node in history.nodes:
        dataset = node.dataset or "Project"
        display_labels.setdefault(
            dataset,
            getattr(node, "display_label", None)
            or getattr(node, "dataset_label", None)
            or dataset,
        )
    return positions, lane_bounds, datasets, display_labels, y_cursor


def render_history_svg(*, history: ProjectHistory, destination: Path) -> None:
    """Write a static SVG suitable for documentation."""

    positions, lane_bounds, datasets, display_labels, height = _layout(history)
    width = 1450
    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        "<style>",
        "text{font-family:Inter,Arial,sans-serif}",
        ".lane{fill:#fbfaf5;stroke:#d9d5ca}",
        ".edge{stroke:#8c8a81;stroke-width:1.7;fill:none}",
        ".edge-label-bg{fill:#fff;stroke:#d9d5ca;stroke-width:1}",
        ".edge-label{font-size:10px;fill:#6d6a60;font-weight:700}",
        ".node-label{font-size:12px;fill:white;font-weight:600}",
        ".lane-label{font-size:14px;fill:#171714;font-weight:700}",
        ".stage-label{font-size:12px;fill:#6d6a60;font-weight:700}",
        "</style>",
        (
            '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" '
            'refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
            '<path d="M 0 0 L 10 5 L 0 10 z" fill="#8c8a81"/>'
            "</marker></defs>"
        ),
        f'<text x="24" y="30" font-size="20" font-weight="700">{escape(history.project_name)} — artifact lineage</text>',
    ]

    for stage_name, stage in (
        ("Source", 0),
        ("Curation / reference", 1),
        ("Transformation", 2),
        ("Analysis", 3),
        ("Reports", 4),
    ):
        parts.append(
            f'<text class="stage-label" x="{190 + stage * 270}" y="54" '
            f'text-anchor="middle">{escape(stage_name)}</text>'
        )

    for dataset in datasets:
        top, lane_height = lane_bounds[dataset]
        parts.append(
            f'<rect class="lane" x="12" y="{top}" width="{width - 24}" '
            f'height="{lane_height}" rx="10"/>'
        )
        parts.append(
            f'<text class="lane-label" x="26" y="{top + 27}">'
            f"{escape(display_labels[dataset])}</text>"
        )

    for edge in history.edges:
        if edge.source not in positions or edge.target not in positions:
            continue
        source_x, source_y = positions[edge.source]
        target_x, target_y = positions[edge.target]
        start_x = source_x + 105
        end_x = target_x - 105
        middle_x = (start_x + end_x) / 2
        parts.append(
            f'<path class="edge" marker-end="url(#arrow)" '
            f'd="M {start_x} {source_y} C {middle_x} {source_y}, '
            f'{middle_x} {target_y}, {end_x} {target_y}"/>'
        )
    for node in history.nodes:
        x, y = positions[node.node_id]
        color = COLORS.get(node.node_type, "#475569")
        if node.status in {"invalid", "incomplete"}:
            color = "#dc2626"
        title = escape(
            f"{node.node_id}\nStatus: {node.status}\nReason: {node.reason or 'Not recorded'}"
        )
        lines = node.label.splitlines()[:2]
        parts.append(f"<g><title>{title}</title>")
        parts.append(
            f'<rect x="{x - 105}" y="{y - 31}" width="210" height="62" '
            f'rx="8" fill="{color}"/>'
        )
        for index, line in enumerate(lines):
            parts.append(
                f'<text class="node-label" x="{x}" y="{y - 4 + index * 17}" '
                f'text-anchor="middle">{escape(line[:31])}</text>'
            )
        parts.append("</g>")

    # Render labels after nodes and give them an opaque pill background.
    for edge in history.edges:
        if edge.source not in positions or edge.target not in positions:
            continue
        source_x, source_y = positions[edge.source]
        target_x, target_y = positions[edge.target]
        middle_x = (source_x + target_x) / 2
        label_y = (source_y + target_y) / 2 - 4
        label_width = max(48, len(edge.relationship) * 6 + 12)
        parts.append(
            f'<rect class="edge-label-bg" x="{middle_x - label_width / 2}" '
            f'y="{label_y - 11}" width="{label_width}" height="18" rx="8"/>'
        )
        parts.append(
            f'<text class="edge-label" x="{middle_x}" y="{label_y + 2}" '
            f'text-anchor="middle">{escape(edge.relationship)}</text>'
        )

    parts.append("</svg>")
    destination.write_text("\n".join(parts), encoding="utf-8")
