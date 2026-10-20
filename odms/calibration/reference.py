"""Scale reference kept in view and measured in every frame.

A circle of known diameter (precision ring, disc or hole) lies on a
calibrated plane.  Each frame its diameter is measured with the same robust
pipeline as the part:

* ``check`` mode - the live scale must agree with the plane calibration
  within ``tolerance``; otherwise millimetres on that plane are blocked
  (camera height, zoom or focus changed).  A passing check counts as
  continuous revalidation.
* ``live`` mode (straight-on planes only) - the plane's scale is taken from
  the reference every frame, compensating for camera height changes, as long
  as it stays within ``max_drift`` of the original calibration.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any

MODES = ("check", "live")


@dataclass
class ScaleReference:
    spec: dict[str, Any]            # MeasurementSpec.to_dict() of the reference circle
    diameter_mm: float
    mode: str = "check"
    tolerance: float = 0.005        # check mode: allowed relative scale deviation
    max_drift: float = 0.10         # live mode: allowed change from the calibrated scale
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "ScaleReference | None":
        if not d:
            return None
        known = {f.name for f in fields(cls)}
        ref = cls(**{k: v for k, v in d.items() if k in known})
        if ref.mode not in MODES:
            ref.mode = "check"
        return ref
