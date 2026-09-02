"""Dataset-level composition of curation and analysis workflows."""

from __future__ import annotations

from typing import Any

import pandas as pd
from fileverse.formats.zarr import BaseZARR

from statomix.core.contracts import AnalyzerInputPaths
from statomix.dataset.base import BaseDataset
from statomix.logging import get_logger
from statomix.pipelines.analyzer.analyzer import Analyzer
from statomix.pipelines.cleaner.cleaner import Cleaner
from statomix.storage.layout import StatomixLayout
from statomix.storage.serializers import (
    load_analyzer_input_paths,
    save_analyzer_input_paths,
)

logger = get_logger(name="dataset")


class Dataset(BaseDataset):
    """A dataset with versioned Cleaner and Analyzer workflows."""

    def __init__(
        self,
        dataset_name: str,
        root_group: Any,
        df: pd.DataFrame | None = None,
        display_label: str | None = None,
    ) -> None:
        super().__init__(
            dataset_name=dataset_name,
            root_group=root_group,
            df=df,
            display_label=display_label,
        )
        self.cleaner = Cleaner(
            df_path=self.paths["df"]["source"],
            root_group=self.groups["cleaner"],
            dataset_name=dataset_name,
        )
        self.analyzer = Analyzer(
            root_group=self.groups["analyzer"],
            dataset_name=dataset_name,
        )

    def configure_analyzer(
        self,
        version: int | None = None,
        config_version: int | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Bind Analyzer inputs to one exact completed Cleaner configuration."""

        cleaner_bundle = self.cleaner._find_group_bundle(
            version=version,
            config_version=config_version,
        )
        cleaner_version_meta = cleaner_bundle["version"]["meta"]
        cleaner_config_meta = cleaner_bundle["config"]["meta"]
        cleaner_version = int(cleaner_version_meta["version"])
        cleaner_config_version = int(cleaner_config_meta["version"])

        if version is not None and cleaner_version != int(version):
            raise RuntimeError(
                f"Requested cleaner version {version}, but version "
                f"{cleaner_version} was resolved."
            )
        if config_version is not None and cleaner_config_version != int(config_version):
            raise RuntimeError(
                f"Requested cleaner config version {config_version}, but "
                f"config version {cleaner_config_version} was resolved."
            )

        curated_data_group = self.cleaner.get_curated_data_group(
            version=cleaner_version,
            config_version=cleaner_config_version,
        )
        if curated_data_group is None:
            raise FileNotFoundError(
                "Curated data is unavailable for "
                f"version:{cleaner_version} and "
                f"config_version:{cleaner_config_version}."
            )

        curated_root = BaseZARR.get_abs_path(curated_data_group)
        analyzer_paths = AnalyzerInputPaths(
            df=curated_root / StatomixLayout.CURATED_DF,
            surv_pairs=curated_root / StatomixLayout.CURATED_SURV_PAIRS,
            col_profiles=curated_root / StatomixLayout.CURATED_COL_PROFILES,
        )

        analyzer_bundle = self.analyzer._require_exact_group_bundle(
            version=cleaner_version,
            config_version=cleaner_config_version,
            version_name=cleaner_version_meta.get("name"),
            config_name=cleaner_config_meta.get("name"),
        )
        analyzer_config_group = analyzer_bundle["config"]["group"]
        analyzer_config_meta = dict(analyzer_bundle["config"]["meta"])
        paths_file = StatomixLayout(
            root=analyzer_bundle["config"]["path"]
        ).group_analyzer_paths()

        if paths_file.exists():
            existing = load_analyzer_input_paths(source=paths_file)
            if existing != analyzer_paths:
                raise RuntimeError(
                    "The existing analyzer path configuration does not match "
                    "the selected cleaner configuration.\n"
                    f"Expected: {analyzer_paths.as_dict(stringify=True)}\n"
                    f"Existing: {existing.as_dict(stringify=True)}"
                )
            reason = "existing_configuration_reused"
        else:
            save_analyzer_input_paths(
                paths=analyzer_paths,
                destination=paths_file,
            )
            reason = "configuration_created"

        analyzer_config_meta["group_analyzer_exists"] = True
        analyzer_config_meta["cleaner_source"] = {
            "version": cleaner_version,
            "config_version": cleaner_config_version,
        }
        analyzer_config_group.attrs["meta"] = analyzer_config_meta

        logger.info(
            "Analyzer configured for version:%s and config_version:%s (%s)",
            cleaner_version,
            cleaner_config_version,
            reason,
        )
        return analyzer_bundle
