"""Content hashes for persisted workflow artifacts."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path


def sha256_file(*, path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of one file without loading it into memory."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Cannot hash missing artifact: {source}")
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")

    digest = sha256()
    with source.open(mode="rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
