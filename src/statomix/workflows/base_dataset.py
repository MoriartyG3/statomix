"""Dataset storage workflow with immutable source-data semantics."""

from __future__ import annotations

from typing import Any

import pandas as pd
from fileverse.formats.zarr import BaseZARR
from pandas.testing import assert_frame_equal

from statomix.core.errors import ArtifactNotFoundError
from statomix.logging import get_logger
from statomix.storage.layout import StatomixLayout

logger = get_logger(name="BaseDataset")


class BaseDataset:
    """Own the stable dataset groups and source dataframe artifact."""

    def __init__(
        self,
        dataset_name: str,
        root_group: Any,
        df: pd.DataFrame | None = None,
    ) -> None:
        if not dataset_name.strip():
            raise ValueError("dataset_name must not be empty")
        self.dataset_name = dataset_name
        self._create_groups(root_group=root_group)
        self._create_paths()
        self._create_source_df(df=df)

    def _create_groups(self, *, root_group: Any) -> None:
        self.groups: dict[str, Any] = {}
        self.groups["root"] = root_group.require_group(self.dataset_name)
        self.groups["df"] = self.groups["root"].require_group("df")
        self.groups["cleaner"] = self.groups["root"].require_group("cleaner")
        self.groups["analyzer"] = self.groups["root"].require_group("analyzer")

    def _create_paths(self) -> None:
        df_root = BaseZARR.get_abs_path(group=self.groups["df"])
        self.paths = {
            "df": {
                "source": StatomixLayout(root=df_root).source_df(),
            }
        }

    def _get_source_df(self) -> pd.DataFrame:
        source_df_path = self.paths["df"]["source"]
        if source_df_path.exists():
            return pd.read_parquet(path=source_df_path)

        message = f"Source dataframe does not exist at:\n{source_df_path}"
        logger.error(message)
        raise ArtifactNotFoundError(message)

    def _create_source_df(self, *, df: pd.DataFrame | None) -> None:
        source_df_path = self.paths["df"]["source"]

        if source_df_path.exists():
            if df is not None:
                existing_df = pd.read_parquet(path=source_df_path)
                try:
                    assert_frame_equal(left=existing_df, right=df)
                    logger.warning(
                        "Provided data already exists. The provided DataFrame "
                        "was not saved."
                    )
                except AssertionError as exc:
                    logger.warning(
                        "Provided data already exists, but the supplied "
                        "DataFrame is not identical to the saved data."
                    )
                    logger.debug(str(exc))
            return

        if df is None:
            message = (
                f"Source data does not exist at {source_df_path} and provided "
                "data is None."
            )
            logger.error(message)
            raise ValueError(message)

        df.to_parquet(path=source_df_path, index=False)
        self.groups["df"].attrs["source_df_exists"] = True
        logger.info("Successfully created and saved new data.")
