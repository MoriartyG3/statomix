from __future__ import annotations

import pandas as pd
import pytest

from statomix import Project


def make_source_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": ["P1", "P2", "P3"],
            "age": [41, 52, 63],
        }
    )


def test_custom_display_label_persists_and_can_change(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    project_dir = tmp_path / "projects"

    project = Project(
        project_name="display_label_test",
        project_dir=project_dir,
    )

    dataset = project.add_dataset(
        df=make_source_df(),
        dataset_name="discovery",
        display_label="Discovery cohort",
    )

    assert dataset is not None
    assert dataset.dataset_name == "discovery"
    assert dataset.display_label == "Discovery cohort"

    dataset.set_display_label(
        display_label="Discovery cohort (OCAT)",
    )

    assert dataset.dataset_name == "discovery"
    assert dataset.display_label == "Discovery cohort (OCAT)"

    reopened_project = Project(
        project_name="display_label_test",
        project_dir=project_dir,
    )

    reopened_dataset = reopened_project.datasets["discovery"]

    assert reopened_dataset.dataset_name == "discovery"
    assert reopened_dataset.display_label == ("Discovery cohort (OCAT)")


def test_display_label_defaults_to_dataset_name(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    project = Project(
        project_name="default_label_test",
        project_dir=tmp_path / "projects",
    )

    dataset = project.add_dataset(
        df=make_source_df(),
        dataset_name="validation_1",
    )

    assert dataset is not None
    assert dataset.display_label == "validation_1"


def test_blank_display_label_is_rejected(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    project = Project(
        project_name="invalid_label_test",
        project_dir=tmp_path / "projects",
    )

    with pytest.raises(
        ValueError,
        match="display_label must not be empty",
    ):
        project.add_dataset(
            df=make_source_df(),
            dataset_name="discovery",
            display_label="   ",
        )

    assert "discovery" not in project.datasets
