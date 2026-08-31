"""Project-level analyzer workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from statomix.core.version_selection import resolve_artifact_version
from statomix.logging import get_logger
from statomix.reporting.excel.project_analysis_config import (
    _create_analysis_config,
)
from statomix.storage.layout import StatomixLayout
from statomix.storage.versioning import BasePipeline

logger = get_logger(name="project_analyzer")


class Analyzer(BasePipeline):
    """Create cross-dataset analysis configuration workbooks."""

    def __init__(self, root_group: Any, dataset_name: str) -> None:
        super().__init__(
            root_group=root_group,
            dataset_name=dataset_name,
            pipeline_name="project_analyzer",
        )

    def _get_default_version_meta(self) -> dict[str, Any]:
        return {}

    def _get_default_config_meta(self) -> dict[str, Any]:
        return {}

    def create_analysis_config(
        self,
        project: Any,
        version: int,
        config_version: int,
        analysis_name: str | None = None,
        create_new: bool = False,
        analysis_config_version: int | None = None,
    ) -> Path:
        group_bundle = self._get_group_bundle(
            version=version,
            config_version=config_version,
        )
        config_group = group_bundle["config"]["group"]
        config_meta = dict(group_bundle["config"]["meta"])
        existing_tracking = config_meta.get("analysis_config")
        selection = resolve_artifact_version(
            tracking=(
                dict(existing_tracking) if isinstance(existing_tracking, dict) else None
            ),
            requested_version=analysis_config_version,
            create_new=create_new,
        )
        existing_name = (
            existing_tracking.get("name")
            if isinstance(existing_tracking, dict)
            else None
        )
        config_meta["analysis_config"] = selection.as_tracking_dict(
            name=analysis_name if analysis_name is not None else existing_name,
        )
        config_group.attrs["meta"] = config_meta

        output_path = StatomixLayout(
            root=group_bundle["config"]["path"]
        ).analysis_config(version=selection.version)
        if output_path.exists():
            logger.info(
                "Analysis configuration version:%s already exists. Set "
                "create_new=True to create the next version.",
                selection.version,
            )
            return output_path

        _create_analysis_config(
            project=project,
            version=version,
            config_version=config_version,
            path=output_path,
        )
        return output_path
