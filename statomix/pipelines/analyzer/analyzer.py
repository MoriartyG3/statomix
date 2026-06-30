import pandas as pd
from pathlib import Path

from fileverse.logger import Logger
from fileverse.formats.yaml import BaseYAML
from fileverse.formats.zarr import BaseZARR
from fileverse.formats.excel import BaseExcel

from statomix.pipelines.base import BasePipeline
from statomix.analytics.datatypes.survival.single_class_surv import SingleClassSurv

from .group_analyzer import GroupAnalyzer

logger = Logger(name="Analyzer").get_logger()

class Analyzer(BasePipeline):
    def __init__(self, root_group):
        super().__init__(root_group=root_group)

    def _get_default_version_meta(self):
        return {}
        
    def _get_default_config_meta(self):
        return {"group_analyzer_exists":False}

    def _get_group_analyzer(self, version,  config_version):
        # version_group = self.get_version_group(version=version, create_new=False, version_name=None)
    
        # config_group = self.get_config_group(
        #     version=None,
        #     version_group=version_group,
        #     config_name=None,
        #     create_new=False
        # )
        
        # base_path = BaseZARR.get_abs_path(config_group)
        
        # group_analyzer_paths_path = base_path/"group_analyzer_path.yaml"

        group_bundle = self._get_group_bundle(version=version, config_version=config_version)
        group_analyzer_paths_path = group_bundle['config']['path']/"group_analyzer_path.yaml"
        
        if not group_analyzer_paths_path.exists():
            error_msg = f"Group analyzer paths does not exists t \n{group_analyzer_paths_path}\n."
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        group_analyzer_paths = BaseYAML.load(path=group_analyzer_paths_path)
        for k, v in group_analyzer_paths.items():
            group_analyzer_paths[k] = Path(v)
    
        return GroupAnalyzer(paths=group_analyzer_paths)

    def create_summary_report(self, version, config_version):
        group_analyzer = self._get_group_analyzer(version=version, config_version=config_version)
    
        group_bundle = self._get_group_bundle(
            version=version, 
            config_version=config_version,
        )
        summary_report_path = group_bundle['config']['path']/"summary.xlsx"

        if summary_report_path.exists():
            logger.info(f"Summary report already exists.")
            return
            
        group_analyzer.create_summary_report(path=summary_report_path)

        self._create_surv_summary_report()

    def _create_surv_summary_report(self, version=None, config_version=None):
        group_bundle = self._get_group_bundle(version=version, config_version=config_version)
        group_analyzer = self._get_group_analyzer(version=version, config_version=config_version)
        
        surv_group = group_bundle['config']['group'].require_group("surv")
        km_plots_group = surv_group.require_group("km_plots")
        
        plots_dir = BaseZARR.get_abs_path(km_plots_group)
        surv_dir = BaseZARR.get_abs_path(surv_group)
        
        df = group_analyzer._get_df()
        surv_pairs = group_analyzer._get_surv_pairs()
        
        descriptives = []
        for surv_label, surv_pair in surv_pairs.pairs.items():
            surv_df = df[[surv_pair.time_profile.col_name, surv_pair.event_profile.col_name]]
            surv_df = surv_df.rename(columns={
                surv_pair.time_profile.col_name: "time",
                surv_pair.event_profile.col_name: "event"
            })
            
            #surv_df['event'] = surv_df['event'].map({'0.0': False, '1.0': True}).astype(bool)
            surv_df['event'] = surv_df['event'].astype(float).astype(int)
            surv_df['event'] = surv_df['event'].astype(str).map({'0': False, '1': True})
            # if surv_df['event'].isna().any():
            #     raise ValueError(f"Unmapped event values for {surv_label}: {surv_pair.event_profile.col_name}")
            # surv_df['event'] = surv_df['event'].astype(bool)
            scs = SingleClassSurv(surv_label=surv_label, surv_df=surv_df)
            
            savepath =  plots_dir/f"{surv_label}.png"
            scs.plot_km_curve(title=surv_label, savepath=savepath, plot_grid=False, plot=False)
        
            time_points = [12,24,36,48,60]
            for time_point in time_points:
                _ = scs.get_survival_probability(time_point=time_point)
                _ = scs.get_rmst(restricted_time=time_point)
            
            descriptive_dict = pd.json_normalize(scs.descriptives).iloc[0].to_dict()
            descriptive_dict["surv_label"] = surv_label  
        
            descriptives.append(descriptive_dict)
        
        descriptives_df = pd.DataFrame(descriptives).set_index(["surv_label"])
        
        descriptives_df.to_excel(surv_dir/"descriptives.xlsx")
        BaseExcel.format_cell_length(path=surv_dir/"descriptives.xlsx")
        
