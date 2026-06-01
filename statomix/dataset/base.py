import pandas as pd
from pandas.testing import assert_frame_equal

from fileverse.logger import Logger
from fileverse.formats.zarr import BaseZARR
from fileverse.formats.yaml import BaseYAML

from statomix.reports.col_report import ColReport

base_yaml = BaseYAML()
logger = Logger(name="BaseDataset").get_logger()


class BaseDataset:
    def __init__(self, dataset_name: str, root_group, df: pd.DataFrame = None):

        self.dataset_name = dataset_name

        self._create_groups(root_group=root_group)
        self._create_paths()

        self._create_source_df(df=df)
        self._col_report = ColReport()
        # self.create_col_report()

    def _create_groups(self, root_group):
        self.zarr_groups = {}
        self.zarr_groups["root"] = root_group.require_group(self.dataset_name)

        self.zarr_groups["df"] = self.zarr_groups["root"].require_group("df")
        if "source_df_exists" not in self.zarr_groups["df"].attrs:
            self.zarr_groups["df"].attrs["source_df_exists"] = False

        self.zarr_groups["col_report_default"] = self.zarr_groups["root"].require_group(
            "col_report_default"
        )
        self.zarr_groups["col_report_curated"] = self.zarr_groups["root"].require_group(
            "col_report_curated"
        )

    def _create_paths(self):
        self.paths = {}
        self.paths["source_df"] = (
            BaseZARR.get_abs_path(zarr_group=self.zarr_groups["df"])
            / "source_df.parquet"
        )

    def get_source_df(self):
        source_df_path = self.paths["source_df"]

        if source_df_path.exists():
            return pd.read_parquet(path=source_df_path)
        else:
            logger.error(f"source_df does not exist at {source_df_path}")
            return

    def _create_source_df(self, df: pd.DataFrame | None):
        source_df_path = self.paths["source_df"]

        if source_df_path.exists():

            if df is not None:

                existing_df = pd.read_parquet(source_df_path)

                try:
                    assert_frame_equal(left=existing_df, right=df)
                    logger.warning(
                        f"source_df already exists. The provided DataFrame was NOT saved to avoid overwriting."
                    )
                except AssertionError as e:
                    logger.warning(
                        f"source_df already exists. However, the provided DataFrame is NOT identical to the saved DataFrame."
                    )
                    logger.debug(str(e))

            return

        if df is not None:
            df.to_parquet(path=source_df_path, index=False)
            self.zarr_groups["df"].attrs["source_df_exists"] = True
            logger.info(f"Successfully created and saved new source_df.")
            return

        error_msg = (
            f"source_df doesn't exist at {source_df_path} and provided df is None."
        )
        logger.error(msg=error_msg)
        raise ValueError(error_msg)

    def create_col_report(
        self,
        report_type="default",
        create_new=False,
        password="statomix",
        lock=False,
    ):
        # dataset = self.datasets[dataset_name]

        if report_type == "default":
            zarr_group = self.zarr_groups["col_report_default"]
            col_report_default_meta = zarr_group.attrs.get("col_report_default", {})

            if "default" not in col_report_default_meta:
                col_report_default_meta["default"] = {}
                col_report_default_meta["default"]["exists"] = False
                col_report_default_meta["default"]["version"] = 0

            version = col_report_default_meta["default"]["version"]

            # report_path = BaseZARR.get_abs_path(zarr_group=zarr_group)/ f"col_report_version{version}.xlsx"
            # profiles_path = BaseZARR.get_abs_path(zarr_group=zarr_group)/ f"col_profile_version{version}.parquet"

            if not col_report_default_meta["default"]["exists"]:
                version = 1
            elif create_new:
                version += 1
            else:
                logger.info(
                    msg=f"{Logger.Emojis.WARN} Default column report version {version} already exists. Set create_new=True to create a new version."
                )
                return

            report_path = (
                BaseZARR.get_abs_path(zarr_group=zarr_group)
                / f"version{version}_col_report.xlsx"
            )
            profiles_path = (
                BaseZARR.get_abs_path(zarr_group=zarr_group)
                / f"version{version}_col_profile.parquet"
            )

            col_profiles = self._col_report.create_col_profiles(df=self.get_source_df())
            self._col_report.save_col_profiles(profiles_path=profiles_path, col_profiles=col_profiles)

            self._col_report._create_col_report(
                df=self.get_source_df(),
                col_profiles=col_profiles,
                report_path=report_path,
                #profiles_path=profiles_path,
                password=password,
                lock=lock,
            )

            col_report_default_meta["default"]["version"] = version
            col_report_default_meta["default"]["exists"] = True

            self.zarr_groups["col_report_default"].attrs[
                "col_report_default"
            ] = col_report_default_meta

    def create_col_edit_schema(self, version=None):

        default_zarr_group = self.zarr_groups["col_report_default"]
        curated_zarr_group = self.zarr_groups["col_report_curated"]
        # curated_version_meta = curated_zarr_group.attrs.get(f"version{version}", {})
        
        if version is None:
            col_report_default_meta = default_zarr_group.attrs["col_report_default"]

            if col_report_default_meta["default"]["exists"]:
                version = col_report_default_meta["default"]["version"]
            else:
                print("Create a default column report first")

        col_edit_schema_path = (
            BaseZARR.get_abs_path(curated_zarr_group)
            / f"version{version}_col_edit_schema.parquet"
        )
        curated_report_path = (
            BaseZARR.get_abs_path(curated_zarr_group)
            / f"version{version}_col_report.xlsx"
        )
        default_profiles_path = (
            BaseZARR.get_abs_path(zarr_group=default_zarr_group)
            / f"version{version}_col_profile.parquet"
        )

        if not curated_report_path.exists():
            print(f"Curated column report does not exist at {curated_report_path}")
            raise FileNotFoundError

        curated_col_report = pd.ExcelFile(curated_report_path)

        if col_edit_schema_path.exists():
            print(f"Column edit schema version{version} already exists.")
            return

        rename_mapping, col_edit_schema = self._col_report.get_col_edit_schema(curated_col_report)

        rename_mapping_path = (BaseZARR.get_abs_path(curated_zarr_group)/ f"version{version}_rename_mapping.yaml")

        base_yaml.save(data=rename_mapping, path=rename_mapping_path)
        
        # curated_version_meta['rename_mapping'] = rename_mapping
        # curated_zarr_group.attrs[f'version{version}'] = curated_version_meta

        col_edit_schema.save(path=col_edit_schema_path)
