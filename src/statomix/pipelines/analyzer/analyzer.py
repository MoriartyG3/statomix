"""Dataset-level Analyzer orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from fileverse.formats.excel import BaseExcel
from fileverse.formats.zarr import BaseZARR

from statomix.analytics.datatypes.survival import SingleClassSurv
from statomix.core.analysis_results import (
    DEFAULT_SURVIVAL_TIME_POINTS,
    empty_survival_descriptives,
)
from statomix.core.version_selection import resolve_artifact_version
from statomix.curation.columns import DataTypes
from statomix.logging import get_logger
from statomix.pipelines.analyzer.group_analyzer import GroupAnalyzer
from statomix.reporting.excel.dataset_analysis_config import AnalysisConfig
from statomix.storage.layout import StatomixLayout
from statomix.storage.serializers import load_analyzer_input_paths
from statomix.storage.versioning import BasePipeline

logger = get_logger(name="dataset_analyzer")


class Analyzer(BasePipeline):
    """Coordinate persisted reports for one dataset."""

    def __init__(self, root_group: Any, dataset_name: str) -> None:
        super().__init__(
            root_group=root_group,
            dataset_name=dataset_name,
            pipeline_name="dataset_analyzer",
        )

    def _get_default_version_meta(self) -> dict[str, Any]:
        return {}

    def _get_default_config_meta(self) -> dict[str, Any]:
        return {"group_analyzer_exists": False}

    def _get_group_analyzer(
        self,
        version: int | None,
        config_version: int | None,
    ) -> GroupAnalyzer:
        group_bundle = self._find_group_bundle(
            version=version,
            config_version=config_version,
        )
        if (group_bundle["config"]["path"] / "input_artifact.json").is_file():
            from statomix.pipelines.analyzer.artifact_inputs import load_binding

            _, _, input_paths, _ = load_binding(self, group_bundle)
            group_analyzer = GroupAnalyzer(paths=input_paths)
            group_analyzer._get_surv_pairs()
            return group_analyzer
        paths_file = StatomixLayout(
            root=group_bundle["config"]["path"]
        ).group_analyzer_paths()
        if not paths_file.exists():
            message = f"Group analyzer paths do not exist at {paths_file}."
            logger.error(message)
            raise FileNotFoundError(message)

        group_analyzer = GroupAnalyzer(
            paths=load_analyzer_input_paths(source=paths_file)
        )
        group_analyzer._get_surv_pairs()
        return group_analyzer
        # return GroupAnalyzer(paths=load_analyzer_input_paths(source=paths_file))

    def create_summary_report(
        self,
        version: int,
        config_version: int,
    ) -> Path:
        group_analyzer = self._get_group_analyzer(
            version=version,
            config_version=config_version,
        )
        group_bundle = self._find_group_bundle(
            version=version,
            config_version=config_version,
        )
        summary_path = StatomixLayout(
            root=group_bundle["config"]["path"]
        ).summary_report()

        datatype_map = group_analyzer._get_datatype_map()
        config_group = group_bundle["config"]["group"]
        config_meta = dict(config_group.attrs.get("meta", {}))
        applicability = dict(config_meta.get("summary_applicability", {}))
        for procedure, datatype in (
            ("categorical_summary", DataTypes.CATEGORICAL),
            ("numerical_summary", DataTypes.NUMERICAL),
            ("normality_diagnostics", DataTypes.NUMERICAL),
        ):
            input_count = len(datatype_map[datatype])
            applicability[procedure] = {
                "status": "applicable" if input_count else "not_applicable",
                "reason": (
                    "applicable_columns_found"
                    if input_count
                    else f"no_curated_{datatype.value.lower()}_columns"
                ),
                "input_count": input_count,
            }
        config_meta["summary_applicability"] = applicability
        config_group.attrs["meta"] = config_meta

        if summary_path.exists():
            logger.info("Summary report already exists at %s.", summary_path)
        else:
            group_analyzer.create_summary_report(path=summary_path)

        self._create_surv_summary_report(
            version=int(group_bundle["version"]["meta"]["version"]),
            config_version=int(group_bundle["config"]["meta"]["version"]),
        )
        return summary_path

    def _create_surv_summary_report(
        self,
        version: int | None = None,
        config_version: int | None = None,
    ) -> Path:
        group_bundle = self._find_group_bundle(
            version=version,
            config_version=config_version,
        )
        config_path = BaseZARR.get_abs_path(group_bundle["config"]["group"])
        if (config_path / "input_artifact.json").is_file():
            from statomix.pipelines.analyzer.artifact_survival import (
                create_artifact_survival_summary,
            )

            return create_artifact_survival_summary(self, group_bundle)
        group_analyzer = self._get_group_analyzer(
            version=version,
            config_version=config_version,
        )
        surv_group = group_bundle["config"]["group"].require_group("surv")
        surv_dir = BaseZARR.get_abs_path(surv_group)
        descriptives_path = surv_dir / "descriptives.xlsx"

        if descriptives_path.exists():
            logger.info(
                "Survival descriptives already exist at %s.",
                descriptives_path,
            )
            return descriptives_path

        surv_pairs = group_analyzer._get_surv_pairs()
        if surv_pairs.is_empty:
            empty_survival_descriptives().to_excel(descriptives_path)
            BaseExcel.format_cell_length(path=descriptives_path)
            surv_group.attrs["meta"] = {
                **dict(surv_group.attrs.get("meta", {})),
                "status": "not_applicable",
                "reason": "no_survival_pairs",
                "input_count": 0,
                "output_count": 0,
            }
            logger.info(
                "No survival pairs found. Created an empty survival "
                "descriptives report."
            )
            return descriptives_path

        plots_dir = BaseZARR.get_abs_path(surv_group.require_group("km_plots"))
        df = group_analyzer._get_df()
        descriptives: list[dict[str, Any]] = []
        for surv_label, surv_pair in surv_pairs.pairs.items():
            surv_df = df[
                [
                    surv_pair.time_profile.col_name,
                    surv_pair.event_profile.col_name,
                ]
            ].rename(
                columns={
                    surv_pair.time_profile.col_name: "time",
                    surv_pair.event_profile.col_name: "event",
                }
            )
            survival = SingleClassSurv(
                surv_label=surv_label,
                surv_df=surv_df,
            )
            survival.plot_km_curve(
                title=surv_label,
                save_path=plots_dir / f"{surv_label}.png",
                plot_grid=False,
                plot=False,
            )
            for time_point in DEFAULT_SURVIVAL_TIME_POINTS:
                survival.get_survival_probability(time_point=time_point)
                survival.get_rmst(restricted_time=time_point)

            row = pd.json_normalize(survival.descriptives).iloc[0].to_dict()
            row["surv_label"] = surv_label
            descriptives.append(row)

        descriptives_df = pd.DataFrame(descriptives).set_index(["surv_label"])
        descriptives_df.to_excel(descriptives_path)
        BaseExcel.format_cell_length(path=descriptives_path)
        surv_group.attrs["meta"] = {
            **dict(surv_group.attrs.get("meta", {})),
            "status": "completed",
            "reason": "descriptives_created",
            "input_count": len(surv_pairs.pairs),
            "output_count": len(descriptives_df),
        }
        return descriptives_path

    def create_analysis_config(
        self,
        version: int,
        config_version: int,
        analysis_config_version: int | None = None,
        analysis_name: str | None = None,
        create_new: bool = False,
    ) -> Path:
        group_bundle = self._find_group_bundle(
            version=version,
            config_version=config_version,
        )
        group_analyzer = self._get_group_analyzer(
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

        AnalysisConfig.create_analysis_config(
            path=output_path,
            datatype_map_df=group_analyzer._get_datatype_map_df(),
        )
        return output_path
