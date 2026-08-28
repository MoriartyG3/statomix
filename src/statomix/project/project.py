"""Compatibility façade for the Project workflow."""

from pathlib import Path

from statomix.workflows.project import DEFAULT_PROJECT_DIR, Project

ROOT = Path.cwd() / DEFAULT_PROJECT_DIR

__all__ = ["Project", "ROOT"]
