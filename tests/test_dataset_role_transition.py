"""Regression tests for audited dataset-role transitions."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from statomix import Project


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return Project(
        project_name="role_transition_test",
        project_dir=tmp_path / "projects",
    )


def _source_frame():
    return pd.DataFrame(
        {
            "Patient ID": ["P1", "P2", "P3"],
            "OS Event": [0, 1, 0],
            "OS_months": [10.0, 20.0, 30.0],
        }
    )


def test_analysis_dataset_can_become_reference_with_audit(project):
    source = _source_frame()
    dataset = project.add_dataset(
        df=source,
        dataset_name="ocat_900_survival",
        display_label="OCAT 900 authoritative survival",
    )
    source_path = dataset.paths["df"]["source"]

    returned = project.set_dataset_role(
        dataset_name="ocat_900_survival",
        dataset_role="reference",
        reason=(
            "Authoritative update source for the OCAT subset; not an "
            "independent analysis cohort."
        ),
    )

    assert returned is dataset
    assert dataset.dataset_role == "reference"
    assert dataset.analyzer.dataset_role == "reference"
    assert dataset.reference.dataset_role == "reference"
    assert dataset.groups["root"].attrs["dataset_role"] == "reference"
    assert (
        project.groups["root"].attrs["datasets"]["ocat_900_survival"]["dataset_role"]
        == "reference"
    )
    assert_frame_equal(pd.read_parquet(source_path), source)

    history = list(dataset.groups["root"].attrs["dataset_role_history"])
    assert len(history) == 1
    assert history[0]["sequence"] == 1
    assert history[0]["previous_role"] == "analysis"
    assert history[0]["dataset_role"] == "reference"
    assert history[0]["reason"] == (
        "Authoritative update source for the OCAT subset; not an "
        "independent analysis cohort."
    )
    assert datetime.fromisoformat(history[0]["changed_at_utc"]).tzinfo is not None


def test_role_transition_persists_after_project_reopen(project):
    project.add_dataset(
        df=_source_frame(),
        dataset_name="ocat_900_survival",
    )
    project.set_dataset_role(
        dataset_name="ocat_900_survival",
        dataset_role="reference",
        reason="Use only as an authoritative update source.",
    )

    reopened = Project(
        project_name=project.project_name,
        project_dir=project.project_dir,
    )
    dataset = reopened.datasets["ocat_900_survival"]

    assert dataset.dataset_role == "reference"
    assert dataset.analyzer.dataset_role == "reference"
    assert dataset.reference.dataset_role == "reference"
    assert len(dataset.groups["root"].attrs["dataset_role_history"]) == 1

    with pytest.raises(
        PermissionError,
        match="not eligible for direct analysis",
    ):
        dataset.configure_analyzer(version=1, config_version=1)


def test_same_role_request_is_idempotent(project):
    dataset = project.add_dataset(
        df=_source_frame(),
        dataset_name="ocat_900_survival",
    )
    project.set_dataset_role(
        dataset_name="ocat_900_survival",
        dataset_role="reference",
        reason="Correct the original administrative classification.",
    )
    original_history = list(dataset.groups["root"].attrs["dataset_role_history"])

    returned = project.set_dataset_role(
        dataset_name="ocat_900_survival",
        dataset_role=" Reference ",
        reason="Repeated notebook cell.",
    )

    assert returned is dataset
    assert list(dataset.groups["root"].attrs["dataset_role_history"]) == (
        original_history
    )


def test_multiple_transitions_append_ordered_history(project):
    dataset = project.add_dataset(
        df=_source_frame(),
        dataset_name="ocat_900_survival",
    )
    project.set_dataset_role(
        dataset_name="ocat_900_survival",
        dataset_role="reference",
        reason="Use as a reference source.",
    )
    project.set_dataset_role(
        dataset_name="ocat_900_survival",
        dataset_role="analysis",
        reason="Approve this dataset as an independent analysis cohort.",
    )

    history = list(dataset.groups["root"].attrs["dataset_role_history"])
    assert [record["sequence"] for record in history] == [1, 2]
    assert [record["previous_role"] for record in history] == [
        "analysis",
        "reference",
    ]
    assert [record["dataset_role"] for record in history] == [
        "reference",
        "analysis",
    ]
    assert dataset.dataset_role == "analysis"
    assert dataset.analyzer.dataset_role == "analysis"
    assert dataset.reference.dataset_role == "analysis"


@pytest.mark.parametrize("reason", ["", "   "])
def test_role_transition_requires_nonempty_reason(project, reason):
    project.add_dataset(
        df=_source_frame(),
        dataset_name="ocat_900_survival",
    )

    with pytest.raises(
        ValueError,
        match="transition reason is required",
    ):
        project.set_dataset_role(
            dataset_name="ocat_900_survival",
            dataset_role="reference",
            reason=reason,
        )


def test_role_transition_rejects_invalid_role(project):
    dataset = project.add_dataset(
        df=_source_frame(),
        dataset_name="ocat_900_survival",
    )

    with pytest.raises(ValueError, match="dataset_role must be one of"):
        project.set_dataset_role(
            dataset_name="ocat_900_survival",
            dataset_role="supplementary",
            reason="Invalid test role.",
        )

    assert dataset.dataset_role == "analysis"
    assert "dataset_role_history" not in dataset.groups["root"].attrs


def test_role_transition_rejects_unknown_dataset(project):
    with pytest.raises(KeyError, match="not registered"):
        project.set_dataset_role(
            dataset_name="missing_dataset",
            dataset_role="reference",
            reason="This dataset does not exist.",
        )
