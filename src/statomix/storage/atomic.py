"""Recoverable same-filesystem writes for file-based artifacts."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile


@contextmanager
def atomic_output_path(*, destination: Path) -> Iterator[Path]:
    """Yield a sibling temporary path and atomically replace on success."""

    final_path = Path(destination)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        prefix=f".{final_path.name}.",
        suffix=final_path.suffix,
        dir=final_path.parent,
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)

    try:
        yield temporary_path
        os.replace(temporary_path, final_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
