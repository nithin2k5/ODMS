"""Calibration data and the rules that decide whether it may be used.

A :class:`CalibrationSet` belongs to one camera at one resolution.  It may
hold one optional lens (distortion) calibration and any number of *plane*
calibrations.  Each plane calibration is valid only for surfaces at the
plane's height/distance; measurements reference the plane they lie on, so a
scale measured on a flange face is never reused for a recess floor.
"""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np

from ..geometry import apply_homography, distance


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def new_id() -> str:
    return uuid.uuid4().hex[:10]


class CalState(str, Enum):
    VALID = "valid"
    REVALIDATE = "revalidation required"
    INVALID = "invalid"
    MISSING = "missing"
    REFERENCE_FAILED = "reference check failed"


@dataclass
class SetupInfo:
    """Operator-recorded description of the physical setup."""

    camera_height_mm: str = ""
    zoom: str = ""
    focus: str = ""
    part_height_mm: str = ""
    lighting: str = ""
    notes: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SetupInfo":
        known = {f.name for f in fields(cls)}
        return cls(**{k: str(v) for k, v in (data or {}).items() if k in known})


@dataclass
class LensCalibration:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    image_size: tuple[int, int]
    rms_px: float
    n_views: int
    pattern_size: tuple[int, int]
    square_mm: float
    lens_id: str = field(default_factory=new_id)
    created: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_matrix": np.asarray(self.camera_matrix).tolist(),
            "dist_coeffs": np.asarray(self.dist_coeffs).ravel().tolist(),
            "image_size": list(self.image_size),
            "rms_px": self.rms_px,
            "n_views": self.n_views,
            "pattern_size": list(self.pattern_size),
            "square_mm": self.square_mm,
            "lens_id": self.lens_id,
            "created": self.created,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LensCalibration":
        return cls(camera_matrix=np.asarray(d["camera_matrix"], dtype=np.float64),
                   dist_coeffs=np.asarray(d["dist_coeffs"], dtype=np.float64),
                   image_size=tuple(d["image_size"]), rms_px=float(d["rms_px"]),
                   n_views=int(d["n_views"]), pattern_size=tuple(d["pattern_size"]),
                   square_mm=float(d["square_mm"]), lens_id=d["lens_id"],
                   created=d.get("created", ""))


@dataclass
class PlaneCalibration:
    """Pixel-to-millimetre mapping for one physical plane.

    ``mode == "scale"``: the camera faces the plane straight on and a single
    ``px_per_mm`` applies in the view image.

    ``mode == "perspective"``: ``homography`` maps view pixels to plane
    millimetres; ``output_transform`` maps view pixels to a rectified image
    with ``px_per_mm`` pixels per millimetre in which circles are circles.
    """

    name: str
    mode: str
    px_per_mm: float
    lens_id: str | None = None
    plane_id: str = field(default_factory=new_id)
    created: str = field(default_factory=now_iso)
    setup: SetupInfo = field(default_factory=SetupInfo)
    # Scale mode
    ref_points: list[list[float]] = field(default_factory=list)
    ref_length_mm: float | None = None
    # Perspective mode
    image_points: list[list[float]] = field(default_factory=list)
    world_points: list[list[float]] = field(default_factory=list)
    homography: list[list[float]] | None = None
    output_transform: list[list[float]] | None = None
    output_size: tuple[int, int] | None = None
    reprojection_rms_mm: float | None = None
    # Optional scale reference measured in every frame (ScaleReference dict)
    reference: dict[str, Any] | None = None
    # Validity tracking
    requires_revalidation: bool = False
    last_validated: str | None = None
    validation_log: list[dict[str, Any]] = field(default_factory=list)

    @property
    def space_id(self) -> str:
        return "view" if self.mode == "scale" else f"rect:{self.plane_id}"

    @property
    def H(self) -> np.ndarray:
        return np.asarray(self.homography, dtype=np.float64)

    @property
    def T(self) -> np.ndarray:
        return np.asarray(self.output_transform, dtype=np.float64)

    def view_distance_mm(self, p1, p2) -> float:
        """Real-world distance between two view-image points on this plane."""
        if self.mode == "scale":
            return distance(p1, p2) / self.px_per_mm
        w = apply_homography(self.H, np.array([p1, p2], dtype=np.float64))
        return distance(w[0], w[1])

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["setup"] = asdict(self.setup)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PlaneCalibration":
        d = dict(d)
        setup = SetupInfo.from_dict(d.pop("setup", None))
        known = {f.name for f in fields(cls)}
        plane = cls(**{k: v for k, v in d.items() if k in known}, setup=setup)
        if plane.output_size is not None:
            plane.output_size = tuple(plane.output_size)
        return plane


@dataclass
class CalibrationSet:
    camera_id: str
    resolution: tuple[int, int]
    lens: LensCalibration | None = None
    planes: dict[str, PlaneCalibration] = field(default_factory=dict)
    calibration_id: str = field(default_factory=new_id)
    created: str = field(default_factory=now_iso)
    updated: str = field(default_factory=now_iso)
    notes: str = ""

    def touch(self) -> None:
        """Record a content change; a new identifier keeps exported results traceable."""
        self.calibration_id = new_id()
        self.updated = now_iso()

    def matches_source(self, camera_id: str, resolution: tuple[int, int]) -> tuple[bool, str]:
        if camera_id != self.camera_id:
            return False, f"camera changed (calibrated for {self.camera_id})"
        if tuple(resolution) != tuple(self.resolution):
            r = self.resolution
            return False, f"resolution changed (calibrated at {r[0]}x{r[1]})"
        return True, ""

    def lens_active(self, camera_id: str, resolution: tuple[int, int]) -> bool:
        return (self.lens is not None and self.matches_source(camera_id, resolution)[0]
                and tuple(self.lens.image_size) == tuple(resolution))

    def plane_state(self, name: str, camera_id: str,
                    resolution: tuple[int, int]) -> tuple[CalState, str]:
        plane = self.planes.get(name)
        if plane is None:
            return CalState.MISSING, f"no calibration for plane '{name}'"
        ok, why = self.matches_source(camera_id, resolution)
        if not ok:
            return CalState.INVALID, why
        current_lens = self.lens.lens_id if self.lens else None
        if plane.lens_id != current_lens:
            return CalState.INVALID, "lens calibration changed since this plane was calibrated"
        if plane.requires_revalidation:
            return CalState.REVALIDATE, "setup must be revalidated with a known reference"
        return CalState.VALID, ""

    def mark_setup_changed(self, plane_name: str | None = None) -> None:
        targets = [self.planes[plane_name]] if plane_name else list(self.planes.values())
        for plane in targets:
            plane.requires_revalidation = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "odms-calibration",
            "version": 1,
            "calibration_id": self.calibration_id,
            "camera_id": self.camera_id,
            "resolution": list(self.resolution),
            "created": self.created,
            "updated": self.updated,
            "notes": self.notes,
            "lens": self.lens.to_dict() if self.lens else None,
            "planes": {k: p.to_dict() for k, p in self.planes.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CalibrationSet":
        if d.get("format") != "odms-calibration":
            raise ValueError("not an ODMS calibration file")
        return cls(camera_id=d["camera_id"], resolution=tuple(d["resolution"]),
                   lens=LensCalibration.from_dict(d["lens"]) if d.get("lens") else None,
                   planes={k: PlaneCalibration.from_dict(p)
                           for k, p in d.get("planes", {}).items()},
                   calibration_id=d["calibration_id"], created=d.get("created", ""),
                   updated=d.get("updated", ""), notes=d.get("notes", ""))
