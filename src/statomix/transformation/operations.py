"""Ordered dispatch for row-preserving and row-selecting operations."""

from __future__ import annotations

from .columns import apply_operations as apply_column_operations
from .rows import apply_row_exclusion
from .specifications import Affine, ConvertUnit, ExcludeRows, Ratio

COLUMN_OPERATIONS = (Affine, Ratio, ConvertUnit)


def apply_operations(parent, operations):
    """Apply a mixed operation plan in order without mutating its parent."""

    state = parent.copy()
    audit = []
    exclusions = []
    for step, operation in enumerate(operations, start=1):
        if isinstance(operation, COLUMN_OPERATIONS):
            state, records = apply_column_operations(state, [operation])
            record = records[0]
            record["step"] = step
            record["operation"] = operation.to_dict()["kind"]
            audit.append(record)
        elif isinstance(operation, ExcludeRows):
            state, summary, records = apply_row_exclusion(
                state,
                operation,
                step=step,
            )
            audit.append(summary)
            exclusions.extend(records)
        else:
            raise TypeError(f"Unsupported operation: {type(operation).__name__}.")
    return state, audit, exclusions
