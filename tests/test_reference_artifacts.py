"""Regression tests for explicit reference-only dataset artifacts."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from statomix import Project
from statomix.curation.columns import ColProfiler, DataTypes
from statomix.curation.survival.profiler import (
    SurvivalDataTypes,
    get_survival_semantic_col_profile,
)
from statomix.curation.survival.report import SurvPair, SurvPairs
from statomix.storage.artifacts import load_artifact
from statomix.storage.parquet_metadata import write_dataframe_with_category_ranks
from statomix.transformation import MONTHS

DEFINITIONS = {
    "OS": "Overall survival event as recorded; elapsed time from randomisation.",
    "DFS": (
        "Disease-free survival event as recorded; elapsed time from randomisation."
    ),
    "LRC": ("Locoregional control event as recorded; elapsed time from randomisation."),
}


def _source_frame(*, duplicate=False, unknown_event=False):
    identifiers = ["P2", "P1", "P3"]
    if duplicate:
        identifiers[-1] = "P1"

    os_events = [0, 1, 0]
    if unknown_event:
        os_events[-1] = 2

    return pd.DataFrame(
        {
            "sourceID": identifiers,
            "OS Event": os_events,
            "DFS Event": [1, 0, 1],
            "LRC Event": [0, 0, 1],
            "OS_months": [22.0, 11.0, 33.0],
            "DFS_months": [20.0, 10.0, 30.0],
            "LRC_months": [19.0, 9.0, 29.0],
            "unused_date": ["2020-01-01", "2020-01-02", "2020-01-03"],
        }
    )


def _event_encodings():
    return {
        "OS Event": {0: False, 1: True},
        "DFS Event": {0: False, 1: True},
        "LRC Event": {0: False, 1: True},
    }


def _endpoints(*, renamed=False):
    suffix = " Duration" if renamed else "_months"
    return {
        label: {
            "event": f"{label} Event",
            "duration": f"{label}{suffix}",
            "definition": DEFINITIONS[label],
        }
        for label in ("OS", "DFS", "LRC")
    }


def _duration_units(*, renamed=False):
    suffix = " Duration" if renamed else "_months"
    return {f"{label}{suffix}": MONTHS for label in ("OS", "DFS", "LRC")}


def _mapping(*, renamed=False):
    mapping = {
        "Patient ID": "sourceID",
        "OS Event": "OS Event",
        "DFS Event": "DFS Event",
        "LRC Event": "LRC Event",
    }
    for label in ("OS", "DFS", "LRC"):
        output = f"{label} Duration" if renamed else f"{label}_months"
        mapping[output] = f"{label}_months"
    return mapping


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return Project(
        project_name="reference_test",
        project_dir=tmp_path / "projects",
    )


def _create_reference(
    project,
    *,
    frame=None,
    renamed=False,
    version=1,
    config_version=1,
):
    frame = _source_frame() if frame is None else frame
    dataset = project.add_dataset(
        df=frame,
        dataset_name="ocat_900_survival",
        display_label="OCAT 900 authoritative survival",
        dataset_role="reference",
    )
    return dataset, dataset.create_reference_artifact(
        version=version,
        config_version=config_version,
        identifier="Patient ID",
        column_mapping=_mapping(renamed=renamed),
        event_columns=_event_encodings(),
        duration_units=_duration_units(renamed=renamed),
        endpoints=_endpoints(renamed=renamed),
        reason="Use the reviewed survival source.",
        name="authoritative_survival",
    )


def _survival_profile(name, role):
    return replace(
        get_survival_semantic_col_profile(name),
        col_type=role,
    )


def _base_artifact(project):
    frame = pd.DataFrame(
        {
            "Patient ID": ["P1", "P2"],
            "OS Event": pd.array([False, True], dtype="boolean"),
            "OS Duration": pd.array([1.0, 2.0], dtype="Float64"),
            "DFS Event": pd.array([False, True], dtype="boolean"),
            "DFS Duration": pd.array([1.0, 2.0], dtype="Float64"),
            "LRC Event": pd.array([False, False], dtype="boolean"),
            "LRC Duration": pd.array([1.0, 2.0], dtype="Float64"),
            "feature": [10.0, 20.0],
        }
    )
    dataset = project.add_dataset(
        df=frame,
        dataset_name="ocat_raw",
        dataset_role="analysis",
    )
    bundle = dataset.cleaner._get_group_bundle(version=1, config_version=1)
    config_path = bundle["config"]["path"]
    curated_path = config_path / "curated_data"
    bundle["config"]["group"].require_group("curated_data")

    roles = {
        "Patient ID": DataTypes.IDENTIFIER,
        "OS Event": DataTypes.SURVIVAL,
        "OS Duration": DataTypes.SURVIVAL,
        "DFS Event": DataTypes.SURVIVAL,
        "DFS Duration": DataTypes.SURVIVAL,
        "LRC Event": DataTypes.SURVIVAL,
        "LRC Duration": DataTypes.SURVIVAL,
        "feature": DataTypes.NUMERICAL,
    }
    profiler = ColProfiler()
    profiles = {
        name: replace(
            profiler.get_col_profile(col_name=name, col_series=frame[name]),
            col_type=role,
        )
        for name, role in roles.items()
    }
    dataset.cleaner.col_report.save_col_profiles(
        col_profiles=profiles,
        path=curated_path / "col_profiles.parquet",
    )
    pairs = SurvPairs(
        pairs={
            label: SurvPair(
                surv_label=label,
                event_profile=_survival_profile(
                    f"{label} Event", SurvivalDataTypes.EVENT
                ),
                time_profile=_survival_profile(
                    f"{label} Duration", SurvivalDataTypes.TIME
                ),
            )
            for label in ("OS", "DFS", "LRC")
        }
    )
    pairs.save(curated_path / "surv_pairs.parquet")
    write_dataframe_with_category_ranks(
        df=frame,
        path=curated_path / "df.parquet",
        category_ranks={},
    )
    reference = dataset.curated_artifact(
        version=1,
        config_version=1,
        endpoint_definitions=DEFINITIONS,
        reason="Declare reviewed endpoint definitions.",
    )
    return dataset, reference


def test_reference_role_defaults_and_persists(project):
    analysis = project.add_dataset(df=pd.DataFrame({"x": [1]}), dataset_name="a")
    reference = project.add_dataset(
        df=pd.DataFrame({"x": [1]}),
        dataset_name="r",
        dataset_role="reference",
    )

    assert analysis.dataset_role == "analysis"
    assert reference.dataset_role == "reference"

    reopened = Project(
        project_name=project.project_name,
        project_dir=project.project_dir,
    )
    assert reopened.datasets["a"].dataset_role == "analysis"
    assert reopened.datasets["r"].dataset_role == "reference"


def test_reference_dataset_is_blocked_from_analyzer(project):
    dataset = project.add_dataset(
        df=pd.DataFrame({"x": [1]}),
        dataset_name="reference",
        dataset_role="reference",
    )

    with pytest.raises(PermissionError, match="not eligible for direct analysis"):
        dataset.configure_analyzer(version=1, config_version=1)

    with pytest.raises(PermissionError, match="not eligible for direct analysis"):
        dataset.analyzer._get_group_analyzer(version=1, config_version=1)


def test_reference_artifact_selects_renames_and_types(project):
    dataset, reference = _create_reference(project, renamed=False)
    state = load_artifact(reference)

    assert reference.manifest["pipeline"] == "reference"
    assert reference.path("source_df") == dataset.paths["df"]["source"]
    assert state.df.columns.tolist() == list(_mapping())
    assert "unused_date" not in state.df
    assert state.df["Patient ID"].tolist() == ["P2", "P1", "P3"]
    assert str(state.df["OS Event"].dtype) == "boolean"
    assert state.df["OS Event"].tolist() == [False, True, False]
    assert state.metadata["columns"]["OS_months"]["unit"] == MONTHS.to_dict()
    assert state.metadata["endpoint_definitions"] == DEFINITIONS
    assert state.pairs.pairs["OS"].time_profile.col_name == "OS_months"
    assert reference.path("audit").is_file()
    assert reference.path("lineage").is_file()


def test_reference_artifact_can_rename_columns(project):
    _, reference = _create_reference(project, renamed=True)
    state = load_artifact(reference)

    assert "OS Duration" in state.df
    assert "OS_months" not in state.df
    assert state.pairs.pairs["OS"].time_profile.col_name == "OS Duration"
    assert state.metadata["columns"]["OS Duration"]["unit"] == MONTHS.to_dict()


def test_reference_artifact_rejects_undeclared_event_value(project):
    frame = _source_frame(unknown_event=True)
    dataset = project.add_dataset(
        df=frame,
        dataset_name="ocat_900_survival",
        dataset_role="reference",
    )

    with pytest.raises(ValueError, match="undeclared value"):
        dataset.create_reference_artifact(
            version=1,
            config_version=1,
            identifier="Patient ID",
            column_mapping=_mapping(),
            event_columns=_event_encodings(),
            duration_units=_duration_units(),
            endpoints=_endpoints(),
            reason="Reviewed source.",
        )


def test_reference_artifact_rejects_duplicate_identifier(project):
    frame = _source_frame(duplicate=True)
    dataset = project.add_dataset(
        df=frame,
        dataset_name="ocat_900_survival",
        dataset_role="reference",
    )

    with pytest.raises(ValueError, match="is not unique"):
        dataset.create_reference_artifact(
            version=1,
            config_version=1,
            identifier="Patient ID",
            column_mapping=_mapping(),
            event_columns=_event_encodings(),
            duration_units=_duration_units(),
            endpoints=_endpoints(),
            reason="Reviewed source.",
        )


def test_analysis_dataset_cannot_create_reference_artifact(project):
    dataset = project.add_dataset(
        df=_source_frame(),
        dataset_name="analysis",
        dataset_role="analysis",
    )

    with pytest.raises(PermissionError, match="require dataset_role='reference'"):
        dataset.create_reference_artifact(
            version=1,
            config_version=1,
            identifier="Patient ID",
            column_mapping=_mapping(),
            event_columns=_event_encodings(),
            duration_units=_duration_units(),
            endpoints=_endpoints(),
            reason="Reviewed source.",
        )


def test_reference_reexecution_is_immutable(project):
    dataset, first = _create_reference(project)
    second = dataset.create_reference_artifact(
        version=1,
        config_version=1,
        identifier="Patient ID",
        column_mapping=_mapping(),
        event_columns=_event_encodings(),
        duration_units=_duration_units(),
        endpoints=_endpoints(),
        reason="Use the reviewed survival source.",
        name="authoritative_survival",
    )
    assert second.artifact_id == first.artifact_id

    with pytest.raises(ValueError, match="configuration differs"):
        dataset.create_reference_artifact(
            version=1,
            config_version=1,
            identifier="Patient ID",
            column_mapping=_mapping(),
            event_columns=_event_encodings(),
            duration_units=_duration_units(),
            endpoints=_endpoints(),
            reason="A different reason changes the specification.",
            name="authoritative_survival",
        )


def test_keyed_update_consumes_reference_artifact(project):
    base_dataset, base = _base_artifact(project)
    _, updates = _create_reference(project)

    corrected = base_dataset.transformer.create_keyed_update_data(
        base=base,
        updates=updates,
        base_key="Patient ID",
        update_key="Patient ID",
        column_mapping={
            "OS Event": "OS Event",
            "OS Duration": "OS_months",
            "DFS Event": "DFS Event",
            "DFS Duration": "DFS_months",
            "LRC Event": "LRC Event",
            "LRC Duration": "LRC_months",
        },
        endpoint_mapping={"OS": "OS", "DFS": "DFS", "LRC": "LRC"},
        version=1,
        config_version=1,
        reason="Replace survival fields from the reviewed reference source.",
    )
    state = load_artifact(corrected)

    assert state.df["Patient ID"].tolist() == ["P1", "P2"]
    assert state.df["feature"].tolist() == [10.0, 20.0]
    assert state.df["OS Event"].tolist() == [True, False]
    assert state.df["OS Duration"].tolist() == [11.0, 22.0]
    assert state.metadata["columns"]["OS Duration"]["unit"] == MONTHS.to_dict()
    assert corrected.path("column_updates").is_file()
    assert corrected.path("unused_updates").is_file()
