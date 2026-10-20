"""Straight-on scale calibration from a reference of known length."""
from __future__ import annotations

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
