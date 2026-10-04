"""Reject malformed numerical settings before constructing a simulator."""
from dataclasses import fields
import math
from numbers import Integral, Real


def validate_numbers(config, *, integers=("version",), optional=(), exclude=()):
    for field in fields(config):
        name = field.name
        if name in exclude:
            continue
        value = getattr(config, name)
        if value is None and name in optional:
            continue
        kind = Integral if name in integers else Real
        if isinstance(value, bool) or not isinstance(value, kind):
            raise ValueError(f"{name} must be a {'whole' if name in integers else 'real'} number")
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError(f"{name} must be finite")


def require_time_multiple(value, unit):
    ratio = value / unit
    if (not math.isfinite(ratio) or ratio < 1
            or not math.isclose(ratio, round(ratio), rel_tol=0, abs_tol=1e-8)):
        raise ValueError("Durations must be positive integer multiples of the timestep")
