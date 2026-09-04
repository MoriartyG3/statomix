"""Bind portable data artifacts to independent Analyzer configurations."""

from __future__ import annotations

import json
import math
from pathlib import Path

from fileverse.formats.zarr import BaseZARR

from statomix.core.artifacts import DatasetArtifactRef, canonical_json, digest_json
from statomix.core.contracts import AnalyzerInputPaths
from statomix.storage.artifacts import (
    artifact_lock,
    load_artifact,
    project_root_for_dataset,
)
from statomix.transformation.specifications import Unit, finite_number


def bind_artifact(
    dataset, *, source, version, config_version, survival_evaluation=None
):
    if any(type(x) is not int or x < 1 for x in (version, config_version)):
        raise ValueError("Explicit positive Analyzer version/configuration required.")
    root = project_root_for_dataset(dataset)
    if (
        source.project_root != root
        or source.manifest["dataset"] != dataset.dataset_name
    ):
        raise ValueError("Bind an artifact owned by this dataset/project.")
    state = load_artifact(source)
    state.pairs.require_supported(operation="Analyzer artifact binding")
    evaluation = survival_evaluation or {}
    if set(evaluation) != set(state.pairs.pairs):
        raise ValueError(
            "Provide explicit survival_evaluation for every endpoint, and no unknown endpoints."
        )
    normalized = {}
    for label, pair in state.pairs.pairs.items():
        record = dict(evaluation[label])
        if set(record) != {"unit", "time_points"}:
            raise ValueError("Each endpoint evaluation requires unit and time_points.")
        unit = record["unit"]
        if not isinstance(unit, Unit):
            raise TypeError("Evaluation unit must be a Unit instance.")
        declared = state.metadata["columns"][pair.time_profile.col_name]["unit"]
        if declared is None or unit.to_dict() != declared or unit.dimension != "time":
            raise ValueError(
                f"Evaluation unit does not match duration unit for {label!r}."
            )
        points = [
            finite_number(x, name="evaluation time") for x in record["time_points"]
        ]
        if (
            any(x <= 0 for x in points)
            or len(set(points)) != len(points)
            or points != sorted(points)
        ):
            raise ValueError(
                "Evaluation times must be positive, unique, and increasing."
            )
        normalized[label] = {"unit": unit.to_dict(), "time_points": points}
    binding = {
        "schema_version": 1,
        "source": source.to_dict(),
        "survival_evaluation": normalized,
    }
    analyzer = dataset.analyzer
    pipeline_root = BaseZARR.get_abs_path(analyzer.root_group)
    config_path = pipeline_root / f"version{version}" / f"config{config_version}"
    binding_path = config_path / "input_artifact.json"
    with artifact_lock(pipeline_root):
        if binding_path.exists():
            if json.loads(binding_path.read_text()) != binding:
                raise ValueError("Analyzer configuration is already bound differently.")
            return analyzer._find_group_bundle(
                version=version, config_version=config_version
            )
        if config_path.exists() and any(
            p.name != "zarr.json" for p in config_path.iterdir()
        ):
            raise FileExistsError(
                "Analyzer configuration contains existing artifacts; select a new version/configuration."
            )
        vg = analyzer.root_group.require_group(f"version{version}")
        cg = vg.require_group(f"config{config_version}")
        root_meta = dict(analyzer.root_group.attrs.get("meta", {}))
        history = sorted(set(root_meta.get("version_history", [])) | {version})
        root_meta.update(version_history=history, latest_version=max(history))
        analyzer.root_group.attrs["meta"] = root_meta
        analyzer.meta = root_meta
        vm = dict(vg.attrs.get("meta", {}))
        configs = sorted(
            set(vm.get("config", {}).get("version_history", [])) | {config_version}
        )
        vm.update(
            version=version,
            name=vm.get("name", "artifact_input"),
            config={"version_history": configs, "latest_version": max(configs)},
        )
        vg.attrs["meta"] = vm
        cg.attrs["meta"] = {
            "version": config_version,
            "name": "artifact_input",
            "group_analyzer_exists": True,
            "input_artifact_id": source.artifact_id,
            "binding_sha256": digest_json(binding),
        }
        from statomix.storage.atomic import atomic_output_path

        with atomic_output_path(destination=binding_path) as temporary:
            temporary.write_text(canonical_json(binding), encoding="utf-8")
        analyzer._group_cache["version"].clear()
        analyzer._group_cache["config"].clear()
        return analyzer._find_group_bundle(
            version=version, config_version=config_version
        )


def load_binding(analyzer, bundle):
    path = bundle["config"]["path"] / "input_artifact.json"
    binding = json.loads(path.read_text())
    if binding.get("schema_version") != 1:
        raise ValueError("Unsupported Analyzer artifact binding.")
    if digest_json(binding) != bundle["config"]["group"].attrs["meta"].get(
        "binding_sha256"
    ):
        raise ValueError("Analyzer binding was modified after configuration.")
    # project/datasets/<dataset>/analyzer
    root = Path(BaseZARR.get_abs_path(analyzer.root_group)).resolve().parents[2]
    reference = DatasetArtifactRef.from_dict(project_root=root, data=binding["source"])
    state = load_artifact(reference)
    state.pairs.require_supported(operation="Analyzer")
    if set(binding["survival_evaluation"]) != set(state.pairs.pairs):
        raise ValueError("Invalid endpoint evaluation binding.")
    for label, pair in state.pairs.pairs.items():
        evaluation = binding["survival_evaluation"][label]
        if (
            evaluation["unit"]
            != state.metadata["columns"][pair.time_profile.col_name]["unit"]
        ):
            raise ValueError("Survival evaluation units no longer agree.")
        if any(not math.isfinite(x) or x <= 0 for x in evaluation["time_points"]):
            raise ValueError("Invalid survival evaluation times.")
    paths = AnalyzerInputPaths(
        df=reference.path("df"),
        surv_pairs=reference.path("surv_pairs"),
        col_profiles=reference.path("col_profiles"),
    )
    return reference, state, paths, binding
