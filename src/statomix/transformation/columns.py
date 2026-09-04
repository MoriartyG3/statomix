"""Row-preserving arithmetic with explicit metadata effects."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from statomix.curation.columns import DataTypes
from statomix.curation.survival.report import SurvPairs

from .metadata import refresh_profiles, validate_state
from .specifications import Affine, ConvertUnit, Ratio, Unit


def _numeric(state, name):
    if name not in state.df:
        raise KeyError(f"Unknown input column: {name!r}")
    event_columns = {p.event_profile.col_name for p in state.pairs.pairs.values()}
    durations = {p.time_profile.col_name for p in state.pairs.pairs.values()}
    role = state.profiles[name].col_type
    allowed = role == DataTypes.NUMERICAL or (
        role == DataTypes.SURVIVAL and name in durations
    )
    series = state.df[name]
    if not allowed or name in event_columns or pd.api.types.is_bool_dtype(series.dtype):
        raise TypeError(f"Arithmetic is not permitted for {name!r} ({role}).")
    if not pd.api.types.is_numeric_dtype(series.dtype) or pd.api.types.is_complex_dtype(
        series.dtype
    ):
        raise TypeError(f"{name!r} is not a real numerical column.")
    if (
        pd.api.types.is_integer_dtype(series.dtype)
        and series.dropna().map(lambda value: abs(int(value)) > 2**53).any()
    ):
        raise ValueError(f"{name!r} exceeds exact integer-to-float range.")
    values = series.to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(values).any():
        raise ValueError(f"Infinite input in {name!r}.")
    return values


def _unit(state, operand):
    if not isinstance(operand, str):
        return Unit("dimensionless", "dimensionless")
    data = state.metadata["columns"][operand]["unit"]
    return Unit(**data) if data else None


def apply_operations(parent, operations):
    state = parent.copy()
    audit = []
    for position, operation in enumerate(operations, start=1):
        output = operation.output
        exists = output in state.df
        if operation.mode == "create" and exists:
            raise ValueError(f"Column {output!r} exists; use explicit replace mode.")
        if operation.mode == "replace" and not exists:
            raise KeyError(f"Cannot replace missing column {output!r}.")
        if exists:
            _numeric(state, output)
        duration_labels = [
            label
            for label, p in state.pairs.pairs.items()
            if p.time_profile.col_name == output
        ]
        if duration_labels and not isinstance(operation, ConvertUnit):
            raise ValueError(
                "Replace a bound survival duration only through ConvertUnit."
            )
        previous_missing = int(state.df[output].isna().sum()) if exists else None
        previous = (
            state.df[output].to_numpy(dtype=float, na_value=np.nan).copy()
            if exists
            else None
        )
        unit = None
        zero_count = 0
        if isinstance(operation, Affine):
            values = np.full(len(state.df), operation.offset, dtype=float)
            known_units = []
            for name, coefficient in operation.terms:
                with np.errstate(over="raise", invalid="raise"):
                    values = values + coefficient * _numeric(state, name)
                known_units.append(_unit(state, name))
            if any(u is not None for u in known_units):
                if any(u is None for u in known_units) or len(set(known_units)) != 1:
                    raise ValueError(
                        "Affine inputs must have matching declared units, or all unknown units."
                    )
                unit = known_units[0]
            elif not operation.terms:
                unit = Unit("dimensionless", "dimensionless")
        elif isinstance(operation, Ratio):
            numerator = (
                _numeric(state, operation.numerator)
                if isinstance(operation.numerator, str)
                else np.full(len(state.df), operation.numerator, dtype=float)
            )
            denominator = (
                _numeric(state, operation.denominator)
                if isinstance(operation.denominator, str)
                else np.full(len(state.df), operation.denominator, dtype=float)
            )
            zero_mask = denominator == 0
            zero_count = int(zero_mask.sum())
            if zero_count and operation.zero == "raise":
                raise ZeroDivisionError(f"{output!r}: {zero_count} zero denominators.")
            with np.errstate(divide="ignore", invalid="ignore", over="raise"):
                values = (
                    operation.scale
                    * numerator
                    / np.where(zero_mask, np.nan, denominator)
                )
            top, bottom = _unit(state, operation.numerator), _unit(
                state, operation.denominator
            )
            if top is not None and bottom is not None:
                if bottom.dimension == "dimensionless":
                    unit = (
                        top
                        if bottom.scale == 1
                        else Unit(
                            f"{top.name}/{bottom.name}",
                            top.dimension,
                            top.scale / bottom.scale,
                        )
                    )
                elif top.dimension == bottom.dimension:
                    unit = Unit(
                        f"{top.name}/{bottom.name}",
                        "dimensionless",
                        top.scale / bottom.scale,
                    )
                else:
                    unit = Unit(
                        f"{top.name}/{bottom.name}",
                        f"({top.dimension})/({bottom.dimension})",
                        top.scale / bottom.scale,
                    )
        elif isinstance(operation, ConvertUnit):
            input_values = _numeric(state, operation.source)
            is_duration = any(
                p.time_profile.col_name == operation.source
                for p in state.pairs.pairs.values()
            )
            if is_duration and operation.target_unit.dimension != "time":
                raise ValueError("A survival duration requires time units.")
            declared = _unit(state, operation.source)
            if declared is not None and declared != operation.source_unit:
                raise ValueError(
                    "Source unit differs from recorded unit; possible double conversion."
                )
            if duration_labels and operation.source != output:
                raise ValueError(
                    "Cannot replace an endpoint duration from a different source column."
                )
            state.metadata["columns"][operation.source][
                "unit"
            ] = operation.source_unit.to_dict()
            with np.errstate(over="raise", invalid="raise"):
                values = input_values * (
                    operation.source_unit.scale / operation.target_unit.scale
                )
            unit = operation.target_unit
            pairs = dict(state.pairs.pairs)
            for label in operation.bind_endpoints:
                if (
                    label not in pairs
                    or pairs[label].time_profile.col_name != operation.source
                ):
                    raise ValueError(
                        f"Endpoint {label!r} does not use {operation.source!r}."
                    )
                pairs[label] = replace(
                    pairs[label],
                    time_profile=replace(
                        pairs[label].time_profile,
                        col_name=output,
                        normalized_name=output.casefold(),
                    ),
                )
            state.pairs = SurvPairs(pairs=pairs)
        else:
            raise TypeError(f"Unsupported operation: {type(operation).__name__}.")
        if np.isinf(values).any():
            raise ValueError(f"Nonfinite result for {output!r}.")
        state.df[output] = pd.Series(values, index=state.df.index, dtype="Float64")
        state.metadata["columns"][output] = {
            "unit": unit.to_dict() if unit else None,
            "category_domain": [],
        }
        bound = any(
            p.time_profile.col_name == output for p in state.pairs.pairs.values()
        )
        output_types = {output: DataTypes.SURVIVAL if bound else DataTypes.NUMERICAL}
        if (
            isinstance(operation, ConvertUnit)
            and operation.source != output
            and operation.bind_endpoints
        ):
            if not any(
                p.time_profile.col_name == operation.source
                for p in state.pairs.pairs.values()
            ):
                output_types[operation.source] = DataTypes.NUMERICAL
        refresh_profiles(state, output_types)
        changed = (
            len(values)
            if previous is None
            else int(
                (
                    ~((previous == values) | (np.isnan(previous) & np.isnan(values)))
                ).sum()
            )
        )
        audit.append(
            {
                "step": position,
                "output": output,
                "mode": operation.mode,
                "rows": len(values),
                "changed_rows": changed,
                "missing_before": previous_missing,
                "missing_after": int(np.isnan(values).sum()),
                "zero_denominators": zero_count,
                "reason": operation.reason,
            }
        )
        validate_state(state)
    return state, audit
