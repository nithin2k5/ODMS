"""Length units for display and entry.

All calculations, calibrations and stored limits use millimetres.  The
operator's chosen unit is applied only when values are shown or typed in.
"""
from __future__ import annotations

# unit -> millimetres per unit, decimals shown
UNITS: dict[str, tuple[float, int]] = {
    "mm": (1.0, 3),
    "cm": (10.0, 4),
    "m": (1000.0, 6),
    "um": (0.001, 1),
    "in": (25.4, 5),
    "mil": (0.0254, 2),     # thousandth of an inch ("thou")
}
DEFAULT_UNIT = "mm"
UNIT_NAMES = tuple(UNITS)


def check_unit(unit: str | None) -> str:
    return unit if unit in UNITS else DEFAULT_UNIT


def from_mm(value_mm: float | None, unit: str) -> float | None:
    if value_mm is None:
        return None
    return value_mm / UNITS[check_unit(unit)][0]


def to_mm(value: float | None, unit: str) -> float | None:
    if value is None:
        return None
    return value * UNITS[check_unit(unit)][0]


def decimals(unit: str) -> int:
    return UNITS[check_unit(unit)][1]


def format_length(value_mm: float | None, unit: str, with_unit: bool = True) -> str:
    """Format a millimetre value in ``unit``, e.g. ``9.0012 cm``."""
    if value_mm is None:
        return "-"
    text = f"{from_mm(value_mm, unit):.{decimals(unit)}f}"
    return f"{text} {check_unit(unit)}" if with_unit else text


def format_plain(value_mm: float | None, unit: str) -> str:
    """Compact form for entry fields (no trailing zeros)."""
    if value_mm is None:
        return ""
    return f"{from_mm(value_mm, unit):.{decimals(unit) + 2}g}"
