"""Part profiles: a named set of measurements that can be saved and reloaded."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ..measurement.models import MeasurementSpec
from ..units import DEFAULT_UNIT, check_unit


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class PartProfile:
    name: str = "Untitled part"
    description: str = ""
    specs: list[MeasurementSpec] = field(default_factory=list)
    # Calibration the geometry was set up with; ring positions are in that
    # calibration's view image, so a different lens calibration or camera
    # usually means the rings must be re-selected.
    calibration_id: str | None = None
    camera_id: str | None = None
    lens_id: str | None = None
    # Display/entry unit; limits and results are stored in millimetres.
    unit: str = DEFAULT_UNIT
    created: str = field(default_factory=_now)
    updated: str = field(default_factory=_now)

    def spec_by_uid(self, uid: str) -> MeasurementSpec | None:
        return next((s for s in self.specs if s.uid == uid), None)

    def unique_name(self, base: str) -> str:
        names = {s.name for s in self.specs}
        if base not in names:
            return base
        i = 2
        while f"{base} ({i})" in names:
            i += 1
        return f"{base} ({i})"

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "odms-profile",
            "version": 1,
            "name": self.name,
            "description": self.description,
            "calibration_id": self.calibration_id,
            "camera_id": self.camera_id,
            "lens_id": self.lens_id,
            "unit": self.unit,
            "created": self.created,
            "updated": self.updated,
            "measurements": [s.to_dict() for s in self.specs],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PartProfile":
        if d.get("format") != "odms-profile":
            raise ValueError("not an ODMS part profile")
        return cls(name=d.get("name", "Untitled part"), description=d.get("description", ""),
                   specs=[MeasurementSpec.from_dict(s) for s in d.get("measurements", [])],
                   calibration_id=d.get("calibration_id"), camera_id=d.get("camera_id"),
                   lens_id=d.get("lens_id"), unit=check_unit(d.get("unit")),
                   created=d.get("created", _now()), updated=d.get("updated", _now()))


def save_profile(path: str | Path, profile: PartProfile) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile.updated = _now()
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(profile.to_dict(), indent=2), encoding="utf-8")
    tmp.replace(path)


def load_profile(path: str | Path) -> PartProfile:
    return PartProfile.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
