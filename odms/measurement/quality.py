"""Detection-quality assessment and rejection rules for a fitted boundary.

The score summarises how completely and consistently the selected boundary
was detected.  It says nothing about dimensional accuracy, which can only be
established by validation against independently measured parts.
"""
from __future__ import annotations

import numpy as np

from .circle_fit import CircleFit, angular_coverage, fit_ellipse_axis_ratio
from .edges import EdgeSamples
from .models import ProcessingParams, QualityReport


def detection_score(q: QualityReport, params: ProcessingParams) -> float:
    fit_term = 0.0
    if np.isfinite(q.rms_px) and params.max_rms_px > 0:
        fit_term = max(0.0, 1.0 - q.rms_px / (2.0 * params.max_rms_px))
    score = 100.0 * (0.45 * q.coverage + 0.35 * q.inlier_ratio + 0.20 * fit_term)
    return float(np.clip(score, 0.0, 100.0))


def assess_fit(fit: CircleFit, edges: EdgeSamples, params: ProcessingParams,
               q: QualityReport) -> QualityReport:
    """Fill fit-dependent fields of ``q``."""
    inl = fit.inliers
    inlier_points = edges.points[inl]
    q.rays_with_inlier = int(np.unique(edges.ray_index[inl]).size)
    q.inlier_ratio = q.rays_with_inlier / max(q.rays_with_edge, 1)
    q.coverage, q.max_gap_deg = angular_coverage(inlier_points, fit.cx, fit.cy)
    q.rms_px = fit.rms
    q.edge_width_px = float(np.median(edges.width[inl])) if inl.any() else float("nan")
    signs = edges.sign[inl]
    if signs.size:
        mean_sign = float(np.mean(signs))
        q.edge_sign = 1 if mean_sign > 0.8 else (-1 if mean_sign < -0.8 else 0)
    q.axis_ratio = fit_ellipse_axis_ratio(inlier_points)
    q.score = detection_score(q, params)
    return q


def rejection_reason(q: QualityReport, params: ProcessingParams) -> str | None:
    """Return a human-readable reason if the fit must not be reported."""
    if q.saturation > params.max_saturation:
        return (f"Glare: {q.saturation:.0%} of the search band is saturated - "
                "reduce exposure or use diffuse lighting")
    if q.coverage < params.min_coverage:
        return (f"Boundary incomplete: only {q.coverage:.0%} angular coverage "
                f"(need {params.min_coverage:.0%})")
    if q.max_gap_deg > params.max_gap_deg:
        return f"Boundary incomplete: {q.max_gap_deg:.0f} deg gap in detected edge"
    if q.inlier_ratio < params.min_inlier_ratio:
        return (f"Edge points inconsistent ({q.inlier_ratio:.0%} inliers): "
                "clutter, reflections or another ring in the band")
    if q.rms_px > params.max_rms_px:
        return f"Poor circle fit: RMS residual {q.rms_px:.2f} px"
    if np.isfinite(q.edge_width_px) and q.edge_width_px > params.max_edge_width_px:
        return f"Edge blurred: profile width {q.edge_width_px:.1f} px - check focus"
    if q.axis_ratio is not None and q.axis_ratio - 1.0 > params.max_ellipticity:
        return (f"Boundary is elliptical (axis ratio {q.axis_ratio:.3f}): camera not "
                "square to this plane - use perspective calibration")
    return None
