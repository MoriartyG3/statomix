from __future__ import annotations

from pathlib import Path

import pytest

from statomix.core.contracts import (
    AnalyzerInputPaths,
    ConfigRef,
    ProcedureState,
    ProcedureStatus,
    VersionRef,
)
from statomix.storage.serializers import (
    load_analyzer_input_paths,
    save_analyzer_input_paths,
)


def test_version_refs_are_positive_and_immutable() -> None:
    assert VersionRef(version=1).version == 1
    assert ConfigRef(version=2).version == 2

    with pytest.raises(ValueError, match="at least 1"):
        VersionRef(version=0)


def test_analyzer_paths_validate_required_keys() -> None:
    paths = AnalyzerInputPaths.from_mapping(
        {
            "df": "df.parquet",
            "surv_pairs": "pairs.parquet",
            "col_profiles": "profiles.parquet",
        }
    )

    assert paths.df == Path("df.parquet")
    assert paths.as_dict(stringify=True)["surv_pairs"] == "pairs.parquet"

    with pytest.raises(ValueError, match="missing keys"):
        AnalyzerInputPaths.from_mapping({"df": "df.parquet"})


def test_analyzer_paths_round_trip_through_atomic_yaml(tmp_path: Path) -> None:
    paths = AnalyzerInputPaths(
        df=tmp_path / "df.parquet",
        surv_pairs=tmp_path / "surv_pairs.parquet",
        col_profiles=tmp_path / "col_profiles.parquet",
    )
    destination = tmp_path / "group_analyzer_path.yaml"

    save_analyzer_input_paths(paths=paths, destination=destination)

    assert load_analyzer_input_paths(source=destination) == paths


def test_procedure_status_serializes_existing_metadata_shape() -> None:
    status = ProcedureStatus(
        status=ProcedureState.NOT_APPLICABLE,
        reason="no_survival_pairs",
        input_count=0,
        output_count=0,
    )

    assert status.to_dict() == {
        "status": "not_applicable",
        "reason": "no_survival_pairs",
        "input_count": 0,
        "output_count": 0,
    }
