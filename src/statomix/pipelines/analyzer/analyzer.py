from pathlib import Path

import pandas as pd
from fileverse.formats.excel import BaseExcel
from fileverse.formats.yaml import BaseYAML
from fileverse.formats.zarr import BaseZARR
from fileverse.logger import Logger

from statomix.analytics.datatypes.survival import SingleClassSurv
from statomix.pipelines.base import BasePipeline
from statomix.pipelines.cleaner.col.col_semantic_rules import DataTypes

from .analysis_config import AnalysisConfig
from .contracts import (
    DEFAULT_SURVIVAL_TIME_POINTS,
    empty_survival_descriptives,
)
from .group_analyzer import GroupAnalyzer

logger = Logger(name="dataset_analyzer").get_logger()


class Analyzer(BasePipeline):
    def __init__(self, root_group, dataset_name):
        super().__init__(
            root_group=root_group,
            dataset_name=dataset_name,
            pipeline_name="dataset_analyzer",
        )

    def _get_default_version_meta(self):
        return {}

    def _get_default_config_meta(self):
        return {"group_analyzer_exists": False}

    def _get_group_analyzer(self, version, config_version):
        group_bundle = self._get_group_bundle(
            version=version,
            config_version=config_version,
        )
        group_analyzer_paths_path = (
            group_bundle["config"]["path"] / "group_analyzer_path.yaml"
        )

        if not group_analyzer_paths_path.exists():
            error_msg = (
                "Group analyzer paths do not exist at " f"{group_analyzer_paths_path}."
            )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        serialized_paths = BaseYAML.load(path=group_analyzer_paths_path)
        group_analyzer_paths = {
            key: Path(value) for key, value in serialized_paths.items()
        }
        return GroupAnalyzer(paths=group_analyzer_paths)

    def create_summary_report(self, version, config_version):
        group_analyzer = self._get_group_analyzer(
            version=version,
            config_version=config_version,
        )
        group_bundle = self._get_group_bundle(
            version=version,
            config_version=config_version,
        )
        summary_report_path = group_bundle["config"]["path"] / "summary.xlsx"

        datatype_map = group_analyzer._get_datatype_map()
        config_group = group_bundle["config"]["group"]
        config_meta = dict(config_group.attrs.get("meta", {}))
        summary_applicability = dict(config_meta.get("summary_applicability", {}))

        for procedure, datatype in (
            ("categorical_summary", DataTypes.CATEGORICAL),
            ("numerical_summary", DataTypes.NUMERICAL),
            ("normality_diagnostics", DataTypes.NUMERICAL),
        ):
            input_count = len(datatype_map[datatype])
            summary_applicability[procedure] = {
                "status": ("applicable" if input_count > 0 else "not_applicable"),
                "reason": (
                    "applicable_columns_found"
                    if input_count > 0
                    else f"no_curated_{datatype.value.lower()}_columns"
                ),
                "input_count": input_count,
            }

        config_meta["summary_applicability"] = summary_applicability
        config_group.attrs["meta"] = config_meta

        if summary_report_path.exists():
            logger.info("Summary report already exists.")
            return

        group_analyzer.create_summary_report(path=summary_report_path)
        self._create_surv_summary_report(
            version=group_bundle["version"]["meta"]["version"],
            config_version=group_bundle["config"]["meta"]["version"],
        )

    def _create_surv_summary_report(
        self,
        version=None,
        config_version=None,
    ):
        group_bundle = self._get_group_bundle(
            version=version,
            config_version=config_version,
        )
        group_analyzer = self._get_group_analyzer(
            version=version,
            config_version=config_version,
        )

        surv_group = group_bundle["config"]["group"].require_group("surv")
        surv_dir = BaseZARR.get_abs_path(surv_group)
        descriptives_path = surv_dir / "descriptives.xlsx"
        surv_pairs = group_analyzer._get_surv_pairs()

        if surv_pairs.is_empty:
            descriptives_df = empty_survival_descriptives()
            descriptives_df.to_excel(descriptives_path)
            BaseExcel.format_cell_length(path=descriptives_path)

            surv_meta = dict(surv_group.attrs.get("meta", {}))
            surv_meta.update(
                {
                    "status": "not_applicable",
                    "reason": "no_survival_pairs",
                    "input_count": 0,
                    "output_count": 0,
                }
            )
            surv_group.attrs["meta"] = surv_meta
            logger.info(
                "No survival pairs found. Created an empty survival "
                "descriptives report."
            )
            return

        km_plots_group = surv_group.require_group("km_plots")
        plots_dir = BaseZARR.get_abs_path(km_plots_group)
        df = group_analyzer._get_df()

        descriptives = []
        for surv_label, surv_pair in surv_pairs.pairs.items():
            surv_df = df[
                [
                    surv_pair.time_profile.col_name,
                    surv_pair.event_profile.col_name,
                ]
            ]
            surv_df = surv_df.rename(
                columns={
                    surv_pair.time_profile.col_name: "time",
                    surv_pair.event_profile.col_name: "event",
                }
            )
            surv_df["event"] = surv_df["event"].astype(float).astype(int)
            surv_df["event"] = surv_df["event"].astype(str).map({"0": False, "1": True})

            survival = SingleClassSurv(
                surv_label=surv_label,
                surv_df=surv_df,
            )
            save_path = plots_dir / f"{surv_label}.png"
            survival.plot_km_curve(
                title=surv_label,
                save_path=save_path,
                plot_grid=False,
                plot=False,
            )

            for time_point in DEFAULT_SURVIVAL_TIME_POINTS:
                survival.get_survival_probability(time_point=time_point)
                survival.get_rmst(restricted_time=time_point)

            descriptive_dict = (
                pd.json_normalize(survival.descriptives).iloc[0].to_dict()
            )
            descriptive_dict["surv_label"] = surv_label
            descriptives.append(descriptive_dict)

        descriptives_df = pd.DataFrame(descriptives).set_index(["surv_label"])
        descriptives_df.to_excel(descriptives_path)
        BaseExcel.format_cell_length(path=descriptives_path)

        surv_meta = dict(surv_group.attrs.get("meta", {}))
        surv_meta.update(
            {
                "status": "completed",
                "reason": "descriptives_created",
                "input_count": len(surv_pairs.pairs),
                "output_count": len(descriptives_df),
            }
        )
        surv_group.attrs["meta"] = surv_meta

    def create_analysis_config(
        self,
        version,
        config_version,
        analysis_config_version,
        analysis_name=None,
        create_new=False,
    ):
        group_bundle = self._get_group_bundle(
            version=version,
            config_version=config_version,
        )
        group_analyzer = self._get_group_analyzer(
            version=version,
            config_version=config_version,
        )

        config_meta = group_bundle["config"]["meta"]
        if "analysis_config" not in config_meta:
            config_meta["analysis_config"] = {
                "latest_version": 1,
                "version_history": [1],
                "name": analysis_name,
            }
            group_bundle["config"]["group"].attrs["meta"] = config_meta
        elif create_new:
            latest_version = config_meta["analysis_config"]["latest_version"]
            latest_version += 1
            config_meta["analysis_config"]["version_history"].append(latest_version)
            config_meta["analysis_config"]["latest_version"] = latest_version
            group_bundle["config"]["group"].attrs["meta"] = config_meta

        latest_version = config_meta["analysis_config"]["latest_version"]
        analysis_config_path = (
            group_bundle["config"]["path"]
            / f"analysis_config_version{latest_version}.xlsx"
        )

        if analysis_config_path.exists():
            logger.info(
                f"Analysis configuration version:{latest_version} already "
                "exists. Set create_new=True to create a new one."
            )
            return

        AnalysisConfig.create_analysis_config(
            path=analysis_config_path,
            datatype_map_df=group_analyzer._get_datatype_map_df(),
        )
