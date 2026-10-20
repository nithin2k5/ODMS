"""Data classes describing what to measure and what was measured."""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from typing import Any

import numpy as np

POLARITIES = ("auto", "any", "rising", "falling")
POLARITY_HELP = {
    "auto": "any edge while setting up; locked to the observed direction on confirm",
    "any": "accept dark-to-bright and bright-to-dark edges",
    "rising": "dark inside, bright outside (intensity rises moving outward)",
    "falling": "bright inside, dark outside (intensity falls moving outward)",
}


@dataclass
class ProcessingParams:
    """Tunable parameters of the per-measurement processing pipeline."""

    # Pre-processing
    clahe: bool = True
    clahe_clip: float = 2.0
    blur_sigma: float = 1.2
    # Edge extraction
    polarity: str = "auto"
    n_rays: int = 360
    candidates_per_ray: int = 2
    min_edge_strength: float = 6.0      # grey levels per pixel
    # Circle fit
    ransac_threshold_px: float = 1.5
    # Acceptance criteria
    min_coverage: float = 0.60          # fraction of 5-degree bins with inliers
    max_gap_deg: float = 90.0
    min_inlier_ratio: float = 0.50      # rays with an inlier / rays with an edge
    max_rms_px: float = 1.0
    max_edge_width_px: float = 7.0      # median edge-profile FWHM (blur check)
    max_saturation: float = 0.20        # fraction of saturated band pixels (glare)
    max_ellipticity: float = 0.015      # allowed (major/minor - 1)
    lock_tolerance: float = 0.05        # allowed relative radius change vs confirmed ring
    # Temporal smoothing
    smoothing_window: int = 5
    move_tolerance_px: float = 3.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ProcessingParams":
        if not data:
            return cls()
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class MeasurementSpec:
    """One named circular boundary chosen by the operator.

    Geometry (``roi``, ``center``, ``r_inner``, ``r_outer``) is stored in the
    *view* image: the camera frame after lens undistortion (if a lens
    calibration is active).  ``locked_radius`` is the radius confirmed by the
    operator, in the measurement space identified by ``locked_space``.
    """

    name: str
    plane: str = "default"
    roi: tuple[int, int, int, int] | None = None
    center: tuple[float, float] | None = None
    r_inner: float = 0.0
    r_outer: float = 0.0
    min_mm: float | None = None
    max_mm: float | None = None
    locked_radius: float | None = None
    locked_space: str | None = None
    params: ProcessingParams = field(default_factory=ProcessingParams)
    uid: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    @property
    def is_complete(self) -> bool:
        return (self.center is not None and self.r_outer > self.r_inner >= 0
                and self.r_outer > 2)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["params"] = self.params.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MeasurementSpec":
        data = dict(data)
        params = ProcessingParams.from_dict(data.pop("params", None))
        roi = data.pop("roi", None)
        center = data.pop("center", None)
        known = {f.name for f in fields(cls)}
        spec = cls(**{k: v for k, v in data.items() if k in known}, params=params)
        spec.roi = tuple(int(v) for v in roi) if roi else None
        spec.center = tuple(float(v) for v in center) if center else None
        return spec

    def fingerprint(self) -> str:
        """Changes whenever anything affecting the measurement changes."""
        return json.dumps(self.to_dict(), sort_keys=True)

    def copy(self) -> "MeasurementSpec":
        return MeasurementSpec.from_dict(self.to_dict())


class Status(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    MEASURED = "MEASURED"
    CALIBRATION_REQUIRED = "Calibration required"
    REVALIDATION_REQUIRED = "Revalidation required"
    CALIBRATION_INVALID = "Calibration invalid"
    REFERENCE_FAILED = "Reference check failed"
    NOT_DETECTED = "Not detected"
    REJECTED = "Rejected"
    INCOMPLETE = "Setup incomplete"


@dataclass
class QualityReport:
    """Detection-quality metrics.  These describe how well the boundary was
    found and fitted; they are *not* an estimate of dimensional accuracy."""

    score: float = 0.0
    n_rays: int = 0
    rays_with_edge: int = 0
    rays_with_inlier: int = 0
    inlier_ratio: float = 0.0
    coverage: float = 0.0
    max_gap_deg: float = 360.0
    rms_px: float = float("nan")
    edge_width_px: float = float("nan")
    saturation: float = 0.0
    axis_ratio: float | None = None
    edge_sign: int = 0                  # +1 rising, -1 falling, 0 unknown


@dataclass
class SpecResult:
    """Outcome of measuring one spec on one frame."""

    uid: str
    name: str
    plane: str
    status: Status
    reason: str = ""
    detected: bool = False
    accepted: bool = False
    space: str = "view"                 # "view" or "rect:<plane id>"
    radius_px: float | None = None      # raw, measurement-space pixels
    diameter_px: float | None = None    # smoothed, measurement-space pixels
    diameter_mm: float | None = None    # smoothed, only with a valid calibration
    raw_diameter_mm: float | None = None
    calibration_state: str = ""
    quality: QualityReport = field(default_factory=QualityReport)
    min_mm: float | None = None
    max_mm: float | None = None
    is_draft: bool = False
    is_reference: bool = False
    px_per_mm: float | None = None      # scale actually used for diameter_mm
    reference_note: str = ""            # live scale-reference status for this plane
    # Geometry in view coordinates for drawing.
    center_view: tuple[float, float] | None = None
    outline_view: np.ndarray | None = None
    diameter_line_view: tuple[tuple[float, float], tuple[float, float]] | None = None
    band_inner_view: np.ndarray | None = None
    band_outer_view: np.ndarray | None = None
    edge_points_view: np.ndarray | None = None
    inlier_mask: np.ndarray | None = None
    smoothing_count: int = 0

    @property
    def display_value(self) -> str:
        if self.diameter_mm is not None:
            return f"{self.diameter_mm:.3f} mm"
        if self.diameter_px is not None:
            return f"{self.diameter_px:.2f} px"
        return "-"
