"""Strict artifacts created directly from reference-only source datasets."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_complex_dtype,
    is_numeric_dtype,
)

from statomix.core.artifacts import canonical_json
from statomix.curation.columns import ColProfiler, DataTypes
from statomix.curation.survival.declarations import (
    DEFAULT_EVENT_STRUCTURE,
    DEFAULT_OBSERVATION_SCHEME,
)
from statomix.curation.survival.profiler import (
    SurvivalDataTypes,
    get_survival_semantic_col_profile,
)
from statomix.curation.survival.report import SurvPair, SurvPairs
from statomix.storage.artifacts import (
    artifact_lock,
    execution_fingerprint,
    publish_artifact,
    read_published,
)
from statomix.storage.hashing import sha256_file
from statomix.transformation.metadata import ArtifactData, initial_metadata
from statomix.transformation.specifications import Unit


def _scalar_identity(value) -> tuple[str, object]:
    """Return a type-sensitive, JSON-safe identity for a source scalar."""

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, bool):
        return "boolean", value

    if isinstance(value, Integral):
        return "integer", int(value)

    if isinstance(value, Real):
        result = float(value)
        if not np.isfinite(result):
            raise ValueError("Declared event values must be finite.")
        return "real", result

    if isinstance(value, str):
        return "string", value

    raise TypeError(
        "Reference event encodings support strings, Booleans, integers, "
        f"and finite real numbers; received {type(value).__name__}."
    )


def _encoded_identity(identity: tuple[str, object]) -> dict[str, object]:
    kind, value = identity
    return {"type": kind, "value": value}


def _normalize_column_mapping(column_mapping) -> tuple[tuple[str, str], ...]:
    if not isinstance(column_mapping, Mapping):
        raise TypeError("column_mapping must map output names to source names.")

    pairs = tuple(column_mapping.items())
    if not pairs:
        raise ValueError("At least one reference column must be selected.")

    for output_name, source_name in pairs:
        if (
            not isinstance(output_name, str)
            or not output_name.strip()
            or not isinstance(source_name, str)
            or not source_name.strip()
        ):
            raise ValueError("Mapped column names must be nonempty strings.")

    outputs = [output for output, _ in pairs]
    sources = [source for _, source in pairs]

    if len(set(outputs)) != len(outputs):
        raise ValueError("Reference output column names must be unique.")

    if len(set(sources)) != len(sources):
        raise ValueError("A source column may be selected only once.")

    return pairs


def _normalize_event_encodings(event_columns) -> dict[str, dict]:
    if not isinstance(event_columns, Mapping):
        raise TypeError("event_columns must map output columns to encodings.")

    normalized = {}
    for column, encoding in event_columns.items():
        if not isinstance(column, str) or not column.strip():
            raise ValueError("Event column names must be nonempty strings.")
        if not isinstance(encoding, Mapping) or not encoding:
            raise ValueError(f"Event column {column!r} requires an encoding.")

        identities = {}
        records = []
        for source_value, event_observed in encoding.items():
            if not isinstance(event_observed, (bool, np.bool_)):
                raise TypeError(f"Encoding results for {column!r} must be Boolean.")
            identity = _scalar_identity(source_value)
            if identity in identities:
                raise ValueError(
                    f"Event encoding for {column!r} contains duplicate "
                    "type-sensitive source values."
                )
            identities[identity] = bool(event_observed)
            records.append(
                {
                    "source": _encoded_identity(identity),
                    "event_observed": bool(event_observed),
                }
            )

        records.sort(key=canonical_json)
        normalized[column] = {
            "lookup": identities,
            "records": records,
        }

    return normalized


def _normalize_column_types(column_types) -> dict[str, DataTypes]:
    if column_types is None:
        return {}
    if not isinstance(column_types, Mapping):
        raise TypeError("column_types must map output columns to DataTypes.")

    normalized = {}
    for column, datatype in column_types.items():
        if not isinstance(column, str) or not column.strip():
            raise ValueError("Column type names must be nonempty strings.")
        normalized[column] = (
            datatype if isinstance(datatype, DataTypes) else DataTypes(datatype)
        )
        if normalized[column] in {DataTypes.IDENTIFIER, DataTypes.SURVIVAL}:
            raise ValueError(
                "Declare the primary identifier with identifier= and "
                "survival columns with endpoints/event_columns, not "
                f"column_types[{column!r}]."
            )
    return normalized


def _normalize_units(duration_units) -> dict[str, Unit]:
    if not isinstance(duration_units, Mapping):
        raise TypeError("duration_units must map duration columns to Unit objects.")

    units = {}
    for column, unit in duration_units.items():
        if not isinstance(unit, Unit):
            raise TypeError(f"Unit for {column!r} must be a Unit instance.")
        if unit.dimension != "time":
            raise ValueError(f"Duration unit for {column!r} must describe time.")
        units[column] = unit
    return units


def _normalize_endpoints(endpoints) -> dict[str, dict]:
    if not isinstance(endpoints, Mapping):
        raise TypeError("endpoints must be a mapping.")

    normalized = {}
    for label, supplied in endpoints.items():
        if not isinstance(label, str) or not label.strip():
            raise ValueError("Endpoint labels must be nonempty strings.")
        if not isinstance(supplied, Mapping):
            raise TypeError(f"Endpoint {label!r} must be a mapping.")

        allowed = {
            "event",
            "duration",
            "definition",
            "event_structure",
            "observation_scheme",
        }
        unknown = set(supplied) - allowed
        if unknown:
            raise ValueError(
                f"Endpoint {label!r} contains unknown fields: {sorted(unknown)!r}."
            )

        record = {
            "event": supplied.get("event"),
            "duration": supplied.get("duration"),
            "definition": supplied.get("definition"),
            "event_structure": supplied.get("event_structure", DEFAULT_EVENT_STRUCTURE),
            "observation_scheme": supplied.get(
                "observation_scheme", DEFAULT_OBSERVATION_SCHEME
            ),
        }

        for field in ("event", "duration", "definition"):
            value = record[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Endpoint {label!r} requires nonempty {field!r}.")

        if record["event"] == record["duration"]:
            raise ValueError(
                f"Endpoint {label!r} cannot use one column as both event and duration."
            )

        normalized[label] = record

    return normalized


def _validate_identifier(frame: pd.DataFrame, identifier: str) -> None:
    if identifier not in frame.columns:
        raise KeyError(f"Reference identifier {identifier!r} is not selected.")

    series = frame[identifier]
    if series.isna().any():
        raise ValueError(
            f"Reference identifier {identifier!r} contains missing values."
        )

    identities = [_scalar_identity(value) for value in series.tolist()]
    duplicates = [
        identity for identity, count in Counter(identities).items() if count > 1
    ]
    if duplicates:
        raise ValueError(
            f"Reference identifier {identifier!r} is not unique: "
            f"{[_encoded_identity(value) for value in duplicates]!r}."
        )


def _encode_events(
    frame: pd.DataFrame,
    event_encodings: dict[str, dict],
) -> list[dict]:
    audit = []
    for column, encoding in event_encodings.items():
        lookup = encoding["lookup"]
        converted = []
        observed_counts = Counter()

        for value in frame[column].tolist():
            if pd.isna(value):
                converted.append(pd.NA)
                continue

            identity = _scalar_identity(value)
            if identity not in lookup:
                raise ValueError(
                    f"Event column {column!r} contains an undeclared value: "
                    f"{_encoded_identity(identity)!r}."
                )

            event_observed = lookup[identity]
            converted.append(event_observed)
            observed_counts[event_observed] += 1

        frame[column] = pd.array(converted, dtype="boolean")
        audit.append(
            {
                "operation": "encode_reference_event",
                "column": column,
                "false_count": observed_counts[False],
                "true_count": observed_counts[True],
                "missing_count": int(frame[column].isna().sum()),
            }
        )

    return audit


class ReferenceArtifactBuilder:
    """Create immutable, explicitly typed artifacts without Cleaner."""

    def __init__(
        self,
        *,
        dataset_group,
        dataset_name,
        project_root,
        project_name,
        source_path,
        dataset_role,
    ):
        self.dataset_group = dataset_group
        self.dataset_name = dataset_name
        self.project_root = Path(project_root).resolve()
        self.project_name = project_name
        self.source_path = Path(source_path).resolve()
        self.dataset_role = dataset_role
        self.root = self.source_path.parents[1] / "reference"

    @staticmethod
    def _numbers(version, config_version):
        if any(
            type(value) is not int or value < 1 for value in (version, config_version)
        ):
            raise ValueError(
                "Explicit positive reference version/config_version required."
            )

    def artifact(self, *, version: int, config_version: int):
        self._numbers(version, config_version)
        return read_published(
            project_root=self.project_root,
            directory=(
                self.root / f"version{version}" / f"config{config_version}" / "data"
            ),
        )

    def create_artifact(
        self,
        *,
        version: int,
        config_version: int,
        identifier: str,
        column_mapping,
        event_columns,
        duration_units,
        endpoints,
        reason: str,
        column_types=None,
        name: str = "",
    ):
        """Select, rename, type, validate, and publish reference columns."""

        self._numbers(version, config_version)
        if self.dataset_role != "reference":
            raise PermissionError(
                f"Dataset {self.dataset_name!r} has dataset_role="
                f"{self.dataset_role!r}; direct reference artifacts require "
                "dataset_role='reference'."
            )
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("A reference-artifact reason is required.")
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError("identifier must be a nonempty output column name.")
        if not isinstance(name, str):
            raise TypeError("name must be a string.")

        mapping = _normalize_column_mapping(column_mapping)
        event_encodings = _normalize_event_encodings(event_columns)
        units = _normalize_units(duration_units)
        endpoint_records = _normalize_endpoints(endpoints)
        explicit_types = _normalize_column_types(column_types)

        output_names = {output for output, _ in mapping}
        event_names = set(event_encodings)
        duration_names = {record["duration"] for record in endpoint_records.values()}
        endpoint_event_names = {record["event"] for record in endpoint_records.values()}

        if event_names != endpoint_event_names:
            raise ValueError(
                "event_columns must cover exactly the event columns declared "
                "by endpoints."
            )
        if set(units) != duration_names:
            raise ValueError(
                "duration_units must cover exactly the duration columns "
                "declared by endpoints."
            )

        role_names = {identifier} | event_names | duration_names | set(explicit_types)
        if role_names != output_names:
            missing_roles = sorted(output_names - role_names)
            unknown_roles = sorted(role_names - output_names)
            raise ValueError(
                "Every selected output requires exactly one semantic role. "
                f"Missing={missing_roles!r}; unknown={unknown_roles!r}."
            )

        protected = {identifier} | event_names | duration_names
        conflicts = protected.intersection(explicit_types)
        if conflicts:
            raise ValueError(
                "column_types must not redeclare identifier or survival columns: "
                f"{sorted(conflicts)!r}."
            )

        if not self.source_path.is_file():
            raise FileNotFoundError(self.source_path)
        source_df = pd.read_parquet(self.source_path)
        if not source_df.columns.is_unique:
            raise ValueError("Reference source has duplicate column names.")

        source_names = [source for _, source in mapping]
        missing_sources = sorted(set(source_names) - set(source_df.columns))
        if missing_sources:
            raise KeyError(
                f"Reference source columns are missing: {missing_sources!r}."
            )

        frame = pd.DataFrame(
            {output: source_df[source].copy() for output, source in mapping},
            index=source_df.index.copy(),
        )
        _validate_identifier(frame, identifier)
        event_audit = _encode_events(frame, event_encodings)

        for duration in duration_names:
            series = frame[duration]
            if (
                not is_numeric_dtype(series.dtype)
                or is_bool_dtype(series.dtype)
                or is_complex_dtype(series.dtype)
            ):
                raise TypeError(
                    f"Reference duration {duration!r} must be real numeric data."
                )
            values = series.dropna().to_numpy(dtype=float)
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(
                    f"Reference duration {duration!r} contains invalid times."
                )

        roles = {
            identifier: DataTypes.IDENTIFIER,
            **{name: DataTypes.SURVIVAL for name in event_names | duration_names},
            **explicit_types,
        }
        profiler = ColProfiler()
        profiles = {
            column: replace(
                profiler.get_col_profile(
                    col_name=column,
                    col_series=frame[column],
                ),
                col_type=roles[column],
            )
            for column in frame.columns
        }

        pairs = {}
        for label, record in endpoint_records.items():
            event_profile = replace(
                get_survival_semantic_col_profile(record["event"]),
                col_type=SurvivalDataTypes.EVENT,
            )
            duration_profile = replace(
                get_survival_semantic_col_profile(record["duration"]),
                col_type=SurvivalDataTypes.TIME,
            )
            pairs[label] = SurvPair(
                surv_label=label,
                event_profile=event_profile,
                time_profile=duration_profile,
                event_structure=record["event_structure"],
                observation_scheme=record["observation_scheme"],
            )
        surv_pairs = SurvPairs(pairs=pairs)
        surv_pairs.require_supported(operation="Reference artifact creation")

        metadata = initial_metadata(
            frame,
            profiles,
            surv_pairs,
            units=units,
            endpoint_definitions={
                label: record["definition"]
                for label, record in endpoint_records.items()
            },
        )
        source_checksum = sha256_file(path=self.source_path)
        lineage = pd.DataFrame(
            {
                "output_row": range(len(frame)),
                "parent_artifact": [f"source_df:{source_checksum}"] * len(frame),
                "parent_row": range(len(frame)),
                "source_dataset": [self.dataset_name] * len(frame),
            }
        )
        state = ArtifactData(
            df=frame,
            profiles=profiles,
            pairs=surv_pairs,
            metadata=metadata,
            ranks={"schema_version": 1, "columns": {}},
            lineage=lineage,
        )

        omitted_columns = [
            column for column in source_df.columns if column not in set(source_names)
        ]
        specification = {
            "schema_version": 1,
            "kind": "reference_import",
            "name": name,
            "dataset_role": self.dataset_role,
            "identifier": identifier,
            "column_mapping": [
                {"output": output, "source": source} for output, source in mapping
            ],
            "column_types": {
                column: datatype.value
                for column, datatype in sorted(explicit_types.items())
            },
            "event_columns": {
                column: event_encodings[column]["records"]
                for column in sorted(event_encodings)
            },
            "duration_units": {
                column: units[column].to_dict() for column in sorted(units)
            },
            "endpoints": endpoint_records,
            "omitted_source_columns": omitted_columns,
            "source_df_sha256": source_checksum,
            "reason": reason,
            "execution": execution_fingerprint(),
        }
        audit = [
            {
                "operation": "select_reference_column",
                "source_column": source,
                "output_column": output,
                "renamed": output != source,
                "semantic_type": roles[output].value,
                "row_count": len(frame),
                "missing_count": int(frame[output].isna().sum()),
            }
            for output, source in mapping
        ]
        audit.extend(event_audit)
        audit.append(
            {
                "operation": "reference_import_summary",
                "source_columns": len(source_df.columns),
                "selected_columns": len(frame.columns),
                "omitted_columns": len(omitted_columns),
                "rows": len(frame),
            }
        )

        version_path = self.root / f"version{version}"
        config_path = version_path / f"config{config_version}"
        destination = config_path / "data"
        with artifact_lock(self.root):
            if destination.exists():
                existing = self.artifact(
                    version=version,
                    config_version=config_version,
                )
                if existing.manifest["specification"] != specification:
                    raise ValueError(
                        "Completed reference configuration differs; choose a "
                        "new configuration."
                    )
                return existing

            reference_group = self.dataset_group.require_group("reference")
            version_group = reference_group.require_group(f"version{version}")
            config_group = version_group.require_group(f"config{config_version}")
            reference = publish_artifact(
                project_root=self.project_root,
                destination=destination,
                state=state,
                identity={
                    "project": self.project_name,
                    "dataset": self.dataset_name,
                    "pipeline": "reference",
                    "version": version,
                    "config_version": config_version,
                },
                parents=(),
                specification=specification,
                audit=audit,
                linked_files={"source_df": self.source_path},
            )
            config_group.attrs["meta"] = {
                "status": "completed",
                "artifact_id": reference.artifact_id,
                "manifest": "data/manifest.json",
            }
            return reference
