"""Straight-on scale calibration.

Two ways to obtain pixels per millimetre for a plane the camera faces
squarely:

* from a reference of known length lying on the plane (most accurate);
* from the pinhole model, ``px_per_mm = focal_px / distance_mm``, using the
  camera's focal length in pixels and the measured distance from the lens's
  optical centre to the plane.  No reference object is needed, but every
  1 % of distance error is a 1 % diameter error.
"""
from __future__ import annotations

import math

from ..geometry import distance
from .models import PlaneCalibration, SetupInfo, now_iso

# Shorter references make a one-pixel click error a large fraction of the scale.
MIN_REFERENCE_PX = 50.0
DEFAULT_REVALIDATION_TOLERANCE = 0.01


def scale_from_reference(p1, p2, length_mm: float) -> float:
    """pixels per millimetre from two points ``length_mm`` apart."""
    if length_mm is None or length_mm <= 0:
        raise ValueError("Reference length must be a positive number of millimetres")
    px = distance(p1, p2)
    if px < MIN_REFERENCE_PX:
        raise ValueError(f"Reference is only {px:.1f} px long; use a reference of at "
                         f"least {MIN_REFERENCE_PX:.0f} px (ideally most of the field of view)")
    return px / float(length_mm)


def click_sensitivity(p1, p2) -> float:
    """Relative scale error caused by a one-pixel error at one end point."""
    px = distance(p1, p2)
    return 1.0 / px if px > 0 else float("inf")


def make_scale_plane(name: str, p1, p2, length_mm: float, lens_id: str | None,
                     setup: SetupInfo | None = None) -> PlaneCalibration:
    ppm = scale_from_reference(p1, p2, length_mm)
    return PlaneCalibration(name=name, mode="scale", px_per_mm=ppm, lens_id=lens_id,
                            setup=setup or SetupInfo(),
                            ref_points=[list(map(float, p1)), list(map(float, p2))],
                            ref_length_mm=float(length_mm), last_validated=now_iso())


def revalidate_plane(plane: PlaneCalibration, p1, p2, length_mm: float,
                     tolerance: float = DEFAULT_REVALIDATION_TOLERANCE
                     ) -> tuple[bool, float, float]:
    """Check a plane calibration against a known reference on that plane.

    Returns ``(passed, measured_mm, relative_error)`` and records the check in
    the plane's validation log.  A passing check clears the revalidation flag;
    a failing one leaves (or sets) it.
    """
    if length_mm <= 0:
        raise ValueError("Reference length must be positive")
    measured = plane.view_distance_mm(p1, p2)
    rel = (measured - length_mm) / length_mm
    passed = abs(rel) <= tolerance
    plane.validation_log.append({
        "time": now_iso(), "reference_mm": float(length_mm),
        "measured_mm": float(measured), "relative_error": float(rel),
        "tolerance": float(tolerance), "passed": bool(passed),
    })
    if passed:
        plane.requires_revalidation = False
        plane.last_validated = now_iso()
    else:
        plane.requires_revalidation = True
    return passed, measured, rel


# Diagonal of a 36 x 24 mm "full-frame" sensor, which defines the
# "35 mm-equivalent" focal length reported by phones and cameras.
FULL_FRAME_DIAGONAL_MM = math.hypot(36.0, 24.0)


def focal_px_from_35mm(f35_mm: float, width_px: int, height_px: int) -> float:
    """Focal length in pixels from a 35 mm-equivalent focal length.

    Valid for the camera's full, uncropped field of view (resizing the image
    is fine; cropping or digital zoom is not).
    """
    if f35_mm <= 0:
        raise ValueError("35 mm-equivalent focal length must be positive")
    return f35_mm * math.hypot(width_px, height_px) / FULL_FRAME_DIAGONAL_MM


def focal_px_from_sensor(f_mm: float, sensor_width_mm: float, width_px: int) -> float:
    if f_mm <= 0 or sensor_width_mm <= 0:
        raise ValueError("Focal length and sensor width must be positive")
    return f_mm * width_px / sensor_width_mm


def make_distance_plane(name: str, distance_mm: float, focal_px: float,
                        lens_id: str | None, setup: SetupInfo | None = None,
                        focal_source: str = "") -> PlaneCalibration:
    """Scale from the pinhole model: ``px_per_mm = focal_px / distance_mm``."""
    if distance_mm is None or distance_mm <= 0:
        raise ValueError("Camera-to-surface distance must be positive")
    if focal_px is None or focal_px <= 0:
        raise ValueError("Focal length in pixels must be positive")
    return PlaneCalibration(name=name, mode="scale", px_per_mm=focal_px / distance_mm,
                            lens_id=lens_id, setup=setup or SetupInfo(), method="distance",
                            distance_mm=float(distance_mm), focal_px=float(focal_px),
                            focal_source=focal_source, last_validated=now_iso())


def exif_focal_35mm(path) -> float | None:
    """35 mm-equivalent focal length stored in a photo's EXIF data, if any."""
    try:
        from PIL import Image

        with Image.open(path) as im:
            exif = im.getexif()
            value = exif.get_ifd(0x8769).get(0xA405) or exif.get(0xA405)
        return float(value) if value else None
    except Exception:  # missing file, no EXIF, unsupported format
        return None
