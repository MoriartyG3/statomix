"""Structural validation for project-history graphs."""

from __future__ import annotations

from collections import defaultdict, deque

from statomix.history.model import (
    HistoryEdge,
    HistoryNode,
    HistoryWarning,
)


def validate_graph(
    *,
    nodes: tuple[HistoryNode, ...],
    edges: tuple[HistoryEdge, ...],
) -> tuple[HistoryWarning, ...]:
    """Report dangling relationships and directed cycles."""

    warnings: list[HistoryWarning] = []
    node_ids = {node.node_id for node in nodes}
    valid_edges = []

    for edge in edges:
        missing = [
            node_id for node_id in (edge.source, edge.target) if node_id not in node_ids
        ]
        if missing:
            warnings.append(
                HistoryWarning(
                    code="dangling_edge",
                    severity="error",
                    message=(
                        f"Relationship {edge.relationship!r} references "
                        f"missing nodes: {missing!r}."
                    ),
                    node_id=edge.target,
                )
            )
        else:
            valid_edges.append(edge)

    adjacency: dict[str, list[str]] = defaultdict(list)
    indegree = {node_id: 0 for node_id in node_ids}

    for edge in valid_edges:
        adjacency[edge.source].append(edge.target)
        indegree[edge.target] += 1

    queue = deque(sorted(key for key, value in indegree.items() if value == 0))
    visited = 0

    while queue:
        node_id = queue.popleft()
        visited += 1
        for target in sorted(adjacency[node_id]):
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)

    if visited != len(node_ids):
        cycle_nodes = sorted(key for key, value in indegree.items() if value > 0)
        warnings.append(
            HistoryWarning(
                code="provenance_cycle",
                severity="error",
                message=(
                    "The provenance graph contains a directed cycle involving "
                    f"{cycle_nodes!r}."
                ),
            )
        )

    return tuple(warnings)
