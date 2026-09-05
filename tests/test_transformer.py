"""Regression tests for reusable Transformer artifacts and strict integration."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from statomix import Project
from statomix.core.artifacts import DatasetArtifactRef, canonical_json
from statomix.curation.columns import ColProfiler, DataTypes
from statomix.curation.survival.profiler import (
    SurvivalDataTypes,
    SurvivalSemanticProfile,
)
from statomix.curation.survival.report import SurvPair, SurvPairs
from statomix.storage.artifacts import (
    artifact_lock,
    load_artifact,
    read_published,
    verify_artifact,
)
from statomix.storage.parquet_metadata import (
    load_category_rank_metadata,
    write_dataframe_with_category_ranks,
)
from statomix.transformation import DAYS, MONTHS, Affine, ConvertUnit, Ratio, Unit
from statomix.transformation.columns import apply_operations
from statomix.transformation.metadata import ArtifactData, initial_metadata
from statomix.transformation.specifications import operation_from_dict


def profile(name, role):
    return SurvivalSemanticProfile(
        col_name=name,
        col_type=role,
        score=1.0,
        tokens=(),
        normalized_name=name.casefold(),
        matched_rules=(),
        all_scores={role: 1.0},
    )


def make_state():
    frame = pd.DataFrame(
        {"a": [1.0, 2.0, np.nan], "b": [2.0, 4.0, 1.0]},
        index=pd.Index([8, 3, 8], name="original_index"),
    )
    profiler = ColProfiler()
    profiles = {
        n: replace(
            profiler.get_col_profile(col_name=n, col_series=frame[n]),
            col_type=DataTypes.NUMERICAL,
        )
        for n in frame
    }
    pairs = SurvPairs.empty()
    return ArtifactData(
        frame,
        profiles,
        pairs,
        initial_metadata(frame, profiles, pairs),
        {"schema_version": 1, "columns": {}},
        pd.DataFrame(
            {
                "output_row": range(3),
                "parent_row": range(3),
                "parent_artifact": ["parent"] * 3,
                "source_dataset": ["a"] * 3,
            }
        ),
    )


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return Project(project_name="transformer_test", project_dir=tmp_path / "projects")


def materialize(project, name="a", offset=0, ranks=None, event_values=None):
    source = pd.DataFrame(
        {
            "patient": [f"p{offset+i}" for i in range(3)],
            "duration": pd.Series([30.4375, 60.875, 91.3125], dtype="Float64"),
            "event": pd.Series(event_values or [True, False, True], dtype="boolean"),
            "feature": [1.0, 2.0, 4.0],
            "grade": ["low", "high", "low"],
        }
    )
    dataset = project.add_dataset(df=source, dataset_name=name)
    bundle = dataset.cleaner._get_group_bundle(version=1, config_version=1)
    config = bundle["config"]["path"]
    curated = config / "curated_data"
    bundle["config"]["group"].require_group("curated_data")
    profiler = ColProfiler()
    roles = {
        "patient": DataTypes.IDENTIFIER,
        "duration": DataTypes.SURVIVAL,
        "event": DataTypes.SURVIVAL,
        "feature": DataTypes.NUMERICAL,
        "grade": DataTypes.CATEGORICAL,
    }
    profiles = {
        n: replace(
            profiler.get_col_profile(col_name=n, col_series=source[n]),
            col_type=roles[n],
        )
        for n in source
    }
    dataset.cleaner.col_report.save_col_profiles(
        col_profiles=profiles, path=curated / "col_profiles.parquet"
    )
    pairs = SurvPairs(
        pairs={
            "OS": SurvPair(
                surv_label="OS",
                event_profile=profile("event", SurvivalDataTypes.EVENT),
                time_profile=profile("duration", SurvivalDataTypes.TIME),
            )
        }
    )
    pairs.save(curated / "surv_pairs.parquet")
    pairs.save(config / "surv_pairs.parquet")
    write_dataframe_with_category_ranks(
        df=source,
        path=curated / "df.parquet",
        category_ranks={"grade": ranks or {"low": 0, "high": 1}},
    )
    reference = dataset.curated_artifact(
        version=1,
        config_version=1,
        units={"duration": DAYS},
        endpoint_definitions={"OS": "All-cause death; time origin: surgery"},
        reason="Protocol review",
    )
    return dataset, reference


def convert(dataset, source, config=1):
    return dataset.transformer.create_data(
        source=source,
        operations=[
            ConvertUnit(
                source="duration",
                output="duration",
                source_unit=DAYS,
                target_unit=MONTHS,
                mode="replace",
                reason="Elapsed months convention",
            )
        ],
        version=1,
        config_version=config,
    )


def test_affine_create_replace_and_parent_immutability():
    original = make_state()
    before = original.df.copy(deep=True)
    output, audit = apply_operations(
        original,
        [
            Affine(
                output="score", terms=(("a", 2), ("b", -0.5)), offset=3, reason="Score"
            ),
            Ratio(
                output="a",
                numerator="score",
                denominator=2,
                mode="replace",
                reason="Rescale",
            ),
        ],
    )
    assert output.df["score"].iloc[:2].tolist() == [4.0, 5.0]
    assert output.df["a"].iloc[:2].tolist() == [2.0, 2.5]
    assert pd.isna(output.df["a"].iloc[2])
    assert output.df.index.equals(before.index)
    assert_frame_equal(original.df, before)
    assert audit[1]["changed_rows"] == 2


@pytest.mark.parametrize(
    "operation",
    [
        Affine(output="a", terms=(), reason="x"),
        Ratio(
            output="missing", numerator="a", denominator=2, mode="replace", reason="x"
        ),
    ],
)
def test_explicit_create_replace_contract(operation):
    with pytest.raises((ValueError, KeyError)):
        apply_operations(make_state(), [operation])


def test_zero_policy():
    with pytest.raises(ZeroDivisionError):
        apply_operations(
            make_state(), [Ratio(output="x", numerator="a", denominator=0, reason="x")]
        )
    state, audit = apply_operations(
        make_state(),
        [Ratio(output="x", numerator="a", denominator=0, zero="missing", reason="x")],
    )
    assert state.df["x"].isna().all()
    assert audit[0]["zero_denominators"] == 3


@pytest.mark.parametrize("value", [float("inf"), float("nan"), True, "two"])
def test_invalid_coefficients(value):
    with pytest.raises((ValueError, TypeError)):
        Affine(output="x", terms=(("a", value),), reason="x")


def test_overflow_is_not_silent():
    state = make_state()
    state.df["a"] = [1e308, 1e308, 1e308]
    with pytest.raises((ValueError, FloatingPointError)):
        apply_operations(
            state, [Affine(output="x", terms=(("a", 1e308),), reason="overflow")]
        )


def test_scalar_column_ratio():
    output, _ = apply_operations(
        make_state(),
        [Ratio(output="x", numerator=8, denominator="b", reason="inverse")],
    )
    assert output.df["x"].tolist() == [4, 2, 8]


def test_specification_round_trip():
    operations = [
        Affine(output="x", terms=(("a", 2),), reason="x"),
        Ratio(output="x", numerator="a", denominator=2, reason="x"),
        ConvertUnit(
            source="a", output="x", source_unit=DAYS, target_unit=MONTHS, reason="x"
        ),
    ]
    for operation in operations:
        serialized = json.loads(canonical_json(operation.to_dict()))
        assert operation_from_dict(serialized).to_dict() == operation.to_dict()


def test_units_are_not_inferred_from_column_names():
    assert make_state().metadata["columns"]["a"]["unit"] is None
    with pytest.raises(ValueError):
        ConvertUnit(
            source="a",
            output="x",
            source_unit=DAYS,
            target_unit=Unit("kg", "mass"),
            reason="x",
        )


def test_conversion_persists_ranks_and_reopens(project):
    dataset, parent = materialize(project)
    before = parent.path("df").read_bytes()
    child = convert(dataset, parent)
    state = load_artifact(child)
    assert state.df["duration"].tolist() == pytest.approx([1, 2, 3])
    assert state.metadata["columns"]["duration"]["unit"] == MONTHS.to_dict()
    assert load_category_rank_metadata(child.path("df")) == load_category_rank_metadata(
        parent.path("df")
    )
    assert parent.path("df").read_bytes() == before
    reopened = Project(
        project_name=project.project_name, project_dir=project.project_dir
    )
    recovered = reopened.datasets["a"].transformer.artifact(version=1, config_version=1)
    assert recovered.artifact_id == child.artifact_id


def test_reexecution_and_configuration_immutability(project):
    dataset, parent = materialize(project)
    first = convert(dataset, parent)
    assert convert(dataset, parent).artifact_id == first.artifact_id
    with pytest.raises(ValueError, match="configuration differs"):
        dataset.transformer.create_data(
            source=parent, operations=[], version=1, config_version=1
        )
    with pytest.raises(ValueError, match="different parents"):
        dataset.transformer.create_data(
            source=first, operations=[], version=1, config_version=2
        )


def test_double_conversion_rejected(project):
    dataset, parent = materialize(project)
    child = convert(dataset, parent)
    with pytest.raises(ValueError, match="Source unit"):
        dataset.transformer.create_data(
            source=child,
            operations=[
                ConvertUnit(
                    source="duration",
                    output="duration",
                    source_unit=DAYS,
                    target_unit=MONTHS,
                    mode="replace",
                    reason="accidental repeat",
                )
            ],
            version=2,
            config_version=1,
        )


def test_new_duration_requires_explicit_binding(project):
    dataset, parent = materialize(project)
    child = dataset.transformer.create_data(
        source=parent,
        operations=[
            ConvertUnit(
                source="duration",
                output="months",
                source_unit=DAYS,
                target_unit=MONTHS,
                reason="new column",
            )
        ],
        version=1,
        config_version=1,
    )
    assert load_artifact(child).pairs.pairs["OS"].time_profile.col_name == "duration"
    bound = dataset.transformer.create_data(
        source=parent,
        operations=[
            ConvertUnit(
                source="duration",
                output="months",
                source_unit=DAYS,
                target_unit=MONTHS,
                bind_endpoints=("OS",),
                reason="new endpoint duration",
            )
        ],
        version=1,
        config_version=2,
    )
    state = load_artifact(bound)
    assert state.pairs.pairs["OS"].time_profile.col_name == "months"
    assert state.profiles["duration"].col_type == DataTypes.NUMERICAL


@pytest.mark.parametrize("name", ["patient", "event", "grade"])
def test_protected_columns_reject_arithmetic(project, name):
    dataset, parent = materialize(project)
    with pytest.raises(TypeError):
        dataset.transformer.create_data(
            source=parent,
            operations=[
                Affine(output=name, terms=(), mode="replace", reason="invalid")
            ],
            version=1,
            config_version=1,
        )


def test_parent_tampering_is_detected(project):
    dataset, parent = materialize(project)
    child = convert(dataset, parent)
    parent.path("df").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="integrity"):
        verify_artifact(child)


def test_incomplete_output_is_unreadable(tmp_path):
    with pytest.raises(FileNotFoundError, match="completed artifact"):
        read_published(project_root=tmp_path, directory=tmp_path)


def test_writer_lock(tmp_path):
    with artifact_lock(tmp_path):
        with pytest.raises(RuntimeError, match="writer lock"):
            with artifact_lock(tmp_path):
                pytest.fail("Concurrent writer entered")
    assert not (tmp_path / ".transformer.lock").exists()


def test_failed_publication_never_completes(project, monkeypatch):
    dataset, parent = materialize(project)
    from statomix.storage import artifacts

    def fail(*args, **kwargs):
        raise OSError("simulated write failure")

    monkeypatch.setattr(artifacts, "write_state", fail)
    with pytest.raises(OSError, match="simulated"):
        convert(dataset, parent)
    assert not (
        dataset.transformer.root / "version1/config1/data/manifest.json"
    ).exists()
    assert not list(dataset.transformer.root.rglob(".transformer-stage-*"))


def test_concatenation_uses_month_artifacts_and_lineage(project):
    first, a = materialize(project, "n0", 0)
    second, b = materialize(project, "priyanka", 10)
    am, bm = convert(first, a), convert(second, b)
    mapping = {name: name for name in load_artifact(am).df.columns}
    combined = project.combine_datasets(
        sources=[am, bm],
        mappings=[mapping, mapping],
        identity_columns=["patient"],
        dataset_name="combined",
        display_label="Combined validation cohorts",
        reason="Pooled analysis",
    )
    ref = combined.transformer.artifact(version=1, config_version=1)
    state = load_artifact(ref)
    assert state.df["duration"].tolist() == pytest.approx([1, 2, 3, 1, 2, 3])
    assert state.df["source_cohort"].tolist() == ["n0"] * 3 + ["priyanka"] * 3
    lineage = pd.read_parquet(ref.path("lineage"))
    assert (
        lineage["parent_artifact"].tolist()
        == [am.artifact_id] * 3 + [bm.artifact_id] * 3
    )
    assert lineage["parent_row"].tolist() == [0, 1, 2, 0, 1, 2]
    assert load_category_rank_metadata(combined.paths["df"]["source"]) == state.ranks


def test_conversion_commutes_with_concatenation(project):
    first, a = materialize(project, "first", 0)
    second, b = materialize(project, "second", 10)
    mapping = {name: name for name in load_artifact(a).df}
    combined_days = project.combine_datasets(
        sources=[a, b],
        mappings=[mapping, mapping],
        identity_columns=["patient"],
        dataset_name="combined_days",
        display_label="Combined days",
        reason="Commutation test",
    )
    day_ref = combined_days.transformer.artifact(version=1, config_version=1)
    converted_after = combined_days.transformer.create_data(
        source=day_ref,
        version=2,
        config_version=1,
        operations=[
            ConvertUnit(
                source="duration",
                output="duration",
                source_unit=DAYS,
                target_unit=MONTHS,
                mode="replace",
                reason="Convert combined",
            )
        ],
    )
    am, bm = convert(first, a), convert(second, b)
    combined_months = project.combine_datasets(
        sources=[am, bm],
        mappings=[mapping, mapping],
        identity_columns=["patient"],
        dataset_name="combined_months",
        display_label="Combined months",
        reason="Commutation test",
    )
    converted_before = combined_months.transformer.artifact(version=1, config_version=1)
    assert_frame_equal(
        load_artifact(converted_after).df, load_artifact(converted_before).df
    )


@pytest.mark.parametrize("conflict", ["units", "overlap", "ranks", "definitions"])
def test_concatenation_conflicts_fail_before_dataset_registration(project, conflict):
    first, a = materialize(project, "a", 0)
    second, b = materialize(
        project,
        "b",
        0 if conflict == "overlap" else 10,
        ranks={"low": 1, "high": 0} if conflict == "ranks" else None,
    )
    am = convert(first, a)
    bm = b if conflict == "units" else convert(second, b)
    if conflict == "definitions":
        b = second.curated_artifact(
            version=1,
            config_version=1,
            units={"duration": DAYS},
            endpoint_definitions={"OS": "Other endpoint origin"},
            reason="Different protocol",
        )
        bm = second.transformer.create_data(
            source=b,
            operations=[
                ConvertUnit(
                    source="duration",
                    output="duration",
                    source_unit=DAYS,
                    target_unit=MONTHS,
                    mode="replace",
                    reason="conversion",
                )
            ],
            version=2,
            config_version=1,
        )
    mapping = {n: n for n in load_artifact(am).df}
    with pytest.raises(ValueError):
        project.combine_datasets(
            sources=[am, bm],
            mappings=[mapping, mapping],
            identity_columns=["patient"],
            dataset_name="bad",
            display_label="Bad",
            reason="test conflict",
        )
    assert "bad" not in project.datasets
    assert "bad" not in project.groups["datasets_root"]


def test_analyzer_binding_rejects_wrong_units_and_rebinding(project):
    dataset, parent = materialize(project)
    child = convert(dataset, parent)
    with pytest.raises(ValueError, match="unit"):
        dataset.configure_analyzer_from_artifact(
            source=child,
            version=2,
            config_version=1,
            survival_evaluation={"OS": {"unit": DAYS, "time_points": [1, 2]}},
        )
    dataset.configure_analyzer_from_artifact(
        source=child,
        version=2,
        config_version=1,
        survival_evaluation={"OS": {"unit": MONTHS, "time_points": [1, 2]}},
    )
    group = dataset.analyzer._get_group_analyzer(version=2, config_version=1)
    assert group._get_df()["duration"].tolist() == pytest.approx([1, 2, 3])
    with pytest.raises(ValueError, match="bound differently"):
        dataset.configure_analyzer_from_artifact(
            source=parent,
            version=2,
            config_version=1,
            survival_evaluation={"OS": {"unit": DAYS, "time_points": [1, 2]}},
        )


def test_legacy_analyzer_configuration_still_works(project):
    dataset, parent = materialize(project)
    dataset.configure_analyzer(version=1, config_version=1)
    group = dataset.analyzer._get_group_analyzer(version=1, config_version=1)
    assert_frame_equal(group._get_df(), load_artifact(parent).df)


def test_legacy_route_cannot_replace_artifact_binding(project):
    dataset, parent = materialize(project)
    child = convert(dataset, parent)
    dataset.configure_analyzer_from_artifact(
        source=child,
        version=1,
        config_version=1,
        survival_evaluation={"OS": {"unit": MONTHS, "time_points": []}},
    )
    with pytest.raises(ValueError, match="artifact-bound"):
        dataset.configure_analyzer(version=1, config_version=1)
    group = dataset.analyzer._get_group_analyzer(version=1, config_version=1)
    assert group._get_df()["duration"].tolist() == pytest.approx([1, 2, 3])


def test_affine_incompatible_units_rejected():
    state = make_state()
    state.metadata["columns"]["a"]["unit"] = DAYS.to_dict()
    state.metadata["columns"]["b"]["unit"] = MONTHS.to_dict()
    with pytest.raises(ValueError, match="matching declared units"):
        apply_operations(
            state,
            [
                Affine(
                    output="x", terms=(("a", 1), ("b", 1)), reason="invalid dimensions"
                )
            ],
        )


def test_large_integer_precision_rejected():
    state = make_state()
    state.df["a"] = pd.Series([-(2**63), 1, 2], index=state.df.index, dtype="int64")
    with pytest.raises(ValueError, match="exact integer"):
        apply_operations(
            state,
            [Ratio(output="x", numerator="a", denominator=2, reason="test precision")],
        )


def test_artifact_relocation(project, tmp_path):
    dataset, parent = materialize(project)
    child = convert(dataset, parent)
    moved = tmp_path / "moved_project"
    shutil.copytree(child.project_root, moved)
    relocated = DatasetArtifactRef.from_dict(project_root=moved, data=child.to_dict())
    assert relocated.artifact_id == child.artifact_id
    assert load_artifact(relocated).df["duration"].tolist() == pytest.approx([1, 2, 3])


@pytest.mark.parametrize("time_points", [[], [1, 2]])
def test_survival_summary_passes_units_and_exact_times(
    project,
    monkeypatch,
    time_points,
):
    from statomix.pipelines.analyzer import artifact_survival

    dataset, parent = materialize(project)
    child = convert(dataset, parent)

    dataset.configure_analyzer_from_artifact(
        source=child,
        version=2,
        config_version=1,
        survival_evaluation={
            "OS": {
                "unit": MONTHS,
                "time_points": time_points,
            }
        },
    )

    calls = []
    captured_ticks = []

    class FakeSurvival:
        def __init__(self, *, surv_label, surv_df):
            self.descriptives = {"test": 1}

            # The plotting correction must not rescale input durations.
            assert surv_df["time"].tolist() == pytest.approx([1, 2, 3])

        def plot_km_curve(self, **kwargs):
            calls.append(kwargs["xlabel"])
            captured_ticks.append(list(kwargs["x_axis_range"]))
            kwargs["save_path"].write_bytes(b"fake image")

        def get_survival_probability(self, *, time_point):
            calls.append(("probability", time_point))

        def get_rmst(self, *, restricted_time):
            calls.append(("rmst", restricted_time))

    monkeypatch.setattr(
        artifact_survival,
        "SingleClassSurv",
        FakeSurvival,
    )

    report = dataset.analyzer._create_surv_summary_report(
        version=2,
        config_version=1,
    )

    expected_calls = ["Time (months)"]

    for point in time_points:
        expected_calls.append(("probability", point))
        expected_calls.append(("rmst", point))

    assert calls == expected_calls

    # This fixture ends at 3 months: no 12-month tick lies in its range.
    # Display ticks must not depend on statistical evaluation times.
    assert captured_ticks == [[0]]
    assert report.exists()

    manifest_path = report.parent / "report_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    plot_record = manifest["plots"][0]

    assert plot_record["endpoint"] == "OS"
    assert plot_record["x_axis_unit"] == "months"
    assert plot_record["x_axis_ticks"] == [0]

    # Reading a completed report must not refit or regenerate it.
    dataset.analyzer._create_surv_summary_report(
        version=2,
        config_version=1,
    )

    assert calls == expected_calls
    assert captured_ticks == [[0]]

    # A missing recorded plot must still be detected.
    (report.parent / plot_record["path"]).unlink()

    with pytest.raises(FileNotFoundError):
        dataset.analyzer._create_surv_summary_report(
            version=2,
            config_version=1,
        )
