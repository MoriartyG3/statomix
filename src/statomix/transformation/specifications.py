"""Closed, serializable v1 operation specifications; no eval or callbacks."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from numbers import Integral, Real


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


def _identifier_value(value):
    """Normalize a JSON-safe identifier scalar without changing its kind."""

    if isinstance(value, str):
        if not value:
            raise ValueError("Excluded identifier values must not be empty strings.")
        return value
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("Excluded identifier values must be finite.")
        return result
    raise TypeError(
        "Excluded identifier values must be strings, booleans, integers, "
        f"or real numbers, not {type(value).__name__}."
    )


def encode_identifier_value(value):
    """Serialize an identifier while retaining its scalar type."""

    normalized = _identifier_value(value)
    if isinstance(normalized, str):
        kind = "string"
    elif isinstance(normalized, bool):
        kind = "boolean"
    elif isinstance(normalized, int):
        kind = "integer"
    else:
        kind = "real"
    return {"type": kind, "value": normalized}


def decode_identifier_value(record):
    """Decode a typed identifier emitted by :func:`encode_identifier_value`."""

    if not isinstance(record, dict) or set(record) != {"type", "value"}:
        raise ValueError("Invalid typed identifier record.")
    kind = record["type"]
    value = _identifier_value(record["value"])
    expected = {
        "string": str,
        "boolean": bool,
        "integer": int,
        "real": float,
    }.get(kind)
    if expected is None or type(value) is not expected:
        raise ValueError("Identifier type tag does not match its value.")
    return value


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


@dataclass(frozen=True, slots=True, kw_only=True)
class ExcludeRows:
    """Remove rows selected by complete values in one Identifier column."""

    identifier: str
    values: tuple[object, ...]
    reason: str

    def __post_init__(self):
        if not isinstance(self.identifier, str) or not self.identifier.strip():
            raise ValueError("identifier must be a nonempty column name.")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("A row-exclusion reason is required.")
        if isinstance(self.values, (str, bytes)):
            raise TypeError("values must be a sequence, not one string.")
        values = tuple(_identifier_value(value) for value in self.values)
        if not values:
            raise ValueError("At least one identifier value must be supplied.")
        identities = {
            (record["type"], record["value"])
            for record in map(encode_identifier_value, values)
        }
        if len(identities) != len(values):
            raise ValueError("Excluded identifier values must be unique.")
        object.__setattr__(self, "values", values)

    def to_dict(self):
        return {
            "kind": "exclude_rows",
            "identifier": self.identifier,
            "values": [encode_identifier_value(value) for value in self.values],
            "reason": self.reason,
        }


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
    if kind == "exclude_rows":
        payload["values"] = tuple(
            decode_identifier_value(record) for record in payload["values"]
        )
        return ExcludeRows(**payload)
    raise ValueError(f"Unknown transformation: {kind!r}")
