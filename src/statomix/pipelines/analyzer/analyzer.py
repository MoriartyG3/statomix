"""Compatibility façade for the dataset Analyzer workflow."""

from fileverse.formats.zarr import BaseZARR

from statomix.workflows.dataset_analyzer import Analyzer

__all__ = ["Analyzer"]
