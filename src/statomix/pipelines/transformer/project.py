"""Project-level creation of a new dataset from multiple transformed parents."""

from __future__ import annotations

import shutil

from fileverse.formats.zarr import BaseZARR

from statomix.dataset.dataset import Dataset
from statomix.storage.artifacts import artifact_lock, load_artifact
from statomix.transformation.concatenation import concatenate_states


def combine_datasets(
    project,
    *,
    sources,
    mappings,
    identity_columns,
    dataset_name,
    display_label,
    reason,
    cohort_column="source_cohort",
):
    if (
        not isinstance(dataset_name, str)
        or not dataset_name
        or dataset_name in {".", ".."}
        or "/" in dataset_name
        or "\\" in dataset_name
    ):
        raise ValueError("New dataset name must be one safe path segment.")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("A combination reason is required.")
    sources = tuple(sources)
    mappings = tuple(dict(m) for m in mappings)
    if isinstance(identity_columns, str):
        raise TypeError("identity_columns must be a sequence.")
    identity_columns = tuple(identity_columns)
    root = BaseZARR.get_abs_path(project.groups["root"]).resolve()
    if any(s.project_root != root for s in sources):
        raise ValueError("All parents must belong to this project.")
    if len({s.artifact_id for s in sources}) != len(sources):
        raise ValueError("Duplicate parent artifact.")
    states = [load_artifact(source) for source in sources]
    for state in states:
        state.pairs.require_supported(operation="Dataset combination")
    output, _ = concatenate_states(
        states,
        mappings=mappings,
        identity_columns=identity_columns,
        cohort_column=cohort_column,
    )
    # Validate before reserving the dataset name. Publish the project registry
    # only after the complete transformed artifact exists.
    with artifact_lock(root):
        if (
            dataset_name in project.groups["datasets_root"]
            or dataset_name in project.datasets
        ):
            raise FileExistsError(
                f"Dataset {dataset_name!r} already exists; no overwrite."
            )
        dataset = Dataset(
            dataset_name=dataset_name,
            display_label=display_label,
            root_group=project.groups["datasets_root"],
            df=output.df,
        )
        reference = dataset.transformer.create_concatenated_data(
            sources=sources,
            mappings=mappings,
            identity_columns=identity_columns,
            version=1,
            config_version=1,
            reason=reason,
            cohort_column=cohort_column,
        )
        # Preserve custom footer metadata in the newly created dataset source.
        # This dataset has not yet been registered and no existing source is replaced.
        shutil.copy2(reference.path("df"), dataset.paths["df"]["source"])
        dataset.groups["root"].attrs["derived_source"] = reference.to_dict()
        registry = dict(project.groups["root"].attrs.get("datasets", {}))
        registry[dataset_name] = {"created_successfully": True}
        project.groups["root"].attrs["datasets"] = registry
        project.datasets[dataset_name] = dataset
        return dataset
