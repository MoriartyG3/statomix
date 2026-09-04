"""Closed, serializable v1 operation specifications; no eval or callbacks."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


def finite_number(value, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"{name} must be a real numeric scalar, not {type(value).__name__}."
        )
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


@dataclass(frozen=True, slots=True)
class Unit:
    """Multiplicative unit: base_value = value * scale.

    Months are an elapsed-time convention, not calendar arithmetic.
    Offset units such as Celsius are intentionally not supported.
    """

    name: str
    dimension: str
    scale: float = 1.0

    def __post_init__(self):
        if (
            not isinstance(self.name, str)
            or not isinstance(self.dimension, str)
            or not self.name.strip()
            or not self.dimension.strip()
        ):
            raise ValueError("Unit name and dimension must be nonempty.")
        if finite_number(self.scale, name="unit scale") <= 0:
            raise ValueError("Unit scale must be positive.")
        object.__setattr__(self, "scale", float(self.scale))

    def to_dict(self):
        return asdict(self)


DAYS = Unit("days", "time", 1.0)
MONTHS = Unit("months", "time", 365.25 / 12)
DIMENSIONLESS = Unit("dimensionless", "dimensionless", 1.0)


def _common(output, mode, reason):
    if not isinstance(output, str) or not output.strip():
        raise ValueError("Output column must be nonempty text.")
    if mode not in {"create", "replace"}:
        raise ValueError("mode must be create or replace.")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("A transformation reason is required.")


@dataclass(frozen=True, slots=True, kw_only=True)
class Affine:
    output: str
    terms: tuple[tuple[str, float], ...]
    reason: str
    offset: float = 0.0
    mode: str = "create"

    def __post_init__(self):
        _common(self.output, self.mode, self.reason)
        terms = tuple(
            (name, finite_number(weight, name="coefficient"))
            for name, weight in self.terms
        )
        if any(not isinstance(name, str) or not name.strip() for name, _ in terms):
            raise ValueError("Input columns must be nonempty text.")
        object.__setattr__(self, "terms", terms)
        object.__setattr__(self, "offset", finite_number(self.offset, name="offset"))

    def to_dict(self):
        return {"kind": "affine", **asdict(self)}


@dataclass(frozen=True, slots=True, kw_only=True)
class Ratio:
    output: str
    numerator: str | float
    denominator: str | float
    reason: str
    scale: float = 1.0
    mode: str = "create"
    zero: str = "raise"

    def __post_init__(self):
        _common(self.output, self.mode, self.reason)
        for value in (self.numerator, self.denominator):
            if isinstance(value, str):
                if not value.strip():
                    raise ValueError("Input column cannot be empty.")
            else:
                finite_number(value, name="ratio operand")
        if self.zero not in {"raise", "missing"}:
            raise ValueError("zero must be raise or missing.")
        object.__setattr__(self, "scale", finite_number(self.scale, name="scale"))

    def to_dict(self):
        return {"kind": "ratio", **asdict(self)}


@dataclass(frozen=True, slots=True, kw_only=True)
class ConvertUnit:
    source: str
    output: str
    source_unit: Unit
    target_unit: Unit
    reason: str
    mode: str = "create"
    bind_endpoints: tuple[str, ...] = ()

    def __post_init__(self):
        _common(self.output, self.mode, self.reason)
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source must be a column name.")
        if self.source_unit.dimension != self.target_unit.dimension:
            raise ValueError("Cannot convert between different dimensions.")
        bindings = tuple(self.bind_endpoints)
        if len(set(bindings)) != len(bindings) or any(
            not isinstance(x, str) or not x for x in bindings
        ):
            raise ValueError("Endpoint bindings must be unique nonempty labels.")
        object.__setattr__(self, "bind_endpoints", bindings)

    def to_dict(self):
        return {"kind": "convert_unit", **asdict(self)}


def operation_from_dict(data):
    payload = dict(data)
    kind = payload.pop("kind")
    if kind == "convert_unit":
        payload["source_unit"] = Unit(**payload["source_unit"])
        payload["target_unit"] = Unit(**payload["target_unit"])
        return ConvertUnit(**payload)
    if kind == "affine":
        return Affine(**payload)
    if kind == "ratio":
        return Ratio(**payload)
    raise ValueError(f"Unknown transformation: {kind!r}")
