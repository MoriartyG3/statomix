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


def normalize_display_label(display_label: str) -> str:
    """Validate and normalize a dataset presentation label."""

    if not isinstance(display_label, str):
        raise TypeError("display_label must be a string")

    normalized_label = display_label.strip()

    if not normalized_label:
        raise ValueError("display_label must not be empty")

    return normalized_label


class BaseDataset:
    """Own the stable dataset groups and source dataframe artifact."""

    def __init__(
        self,
        dataset_name: str,
        root_group: Any,
        df: pd.DataFrame | None = None,
        display_label: str | None = None,
    ) -> None:
        if not dataset_name.strip():
            raise ValueError("dataset_name must not be empty")

        requested_display_label = (
            normalize_display_label(display_label)
            if display_label is not None
            else None
        )

        self.dataset_name = dataset_name

        self._create_groups(root_group=root_group)
        self._create_paths()
        self._create_source_df(df=df)
        self._initialize_display_label(
            display_label=requested_display_label,
        )

    @property
    def display_label(self) -> str:
        """Current human-readable label used for presentation."""

        return self._display_label

    def _initialize_display_label(
        self,
        *,
        display_label: str | None,
    ) -> None:
        """Load an existing label or initialize it for a new dataset."""

        stored_display_label = self.groups["root"].attrs.get("display_label")

        if stored_display_label is None:
            resolved_display_label = (
                self.dataset_name if display_label is None else display_label
            )

            self.groups["root"].attrs["display_label"] = resolved_display_label
        else:
            resolved_display_label = normalize_display_label(stored_display_label)

            if display_label is not None and display_label != resolved_display_label:
                raise RuntimeError(
                    "The dataset already has display_label="
                    f"{resolved_display_label!r}. Use "
                    "set_display_label() to change it explicitly."
                )

        self._display_label = resolved_display_label

    def set_display_label(
        self,
        *,
        display_label: str,
    ) -> None:
        """Persistently change the dataset's presentation label."""

        normalized_label = normalize_display_label(display_label)
        previous_label = self._display_label

        self.groups["root"].attrs["display_label"] = normalized_label
        self._display_label = normalized_label

        logger.info(
            "Changed dataset '%s' display label from %r to %r.",
            self.dataset_name,
            previous_label,
            normalized_label,
        )

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
