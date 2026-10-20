"""Measure one selected circular boundary inside an image window."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .circle_fit import CircleFit, ransac_circle
from .edges import EdgeSamples, extract_radial_edges
from .models import ProcessingParams, QualityReport
from .preprocess import annulus_mask, enhance, saturation_fraction
from .quality import assess_fit, rejection_reason

# Fewer rays than this fraction with any edge means nothing is there.
MIN_EDGE_RAY_FRACTION = 0.20


@dataclass
class MeasureOutcome:
    detected: bool
    accepted: bool
    reason: str
    fit: CircleFit | None
    edges: EdgeSamples
    quality: QualityReport = field(default_factory=QualityReport)


def effective_polarity(params: ProcessingParams) -> str:
    return "any" if params.polarity == "auto" else params.polarity


def measure_circle(gray: np.ndarray, center: tuple[float, float],
                   r_inner: float, r_outer: float, params: ProcessingParams, *,
                   valid_mask: np.ndarray | None = None,
                   locked_radius: float | None = None,
                   point_filter: Callable[[np.ndarray], np.ndarray] | None = None
                   ) -> MeasureOutcome:
    """Find and fit the boundary between ``r_inner`` and ``r_outer``.

    ``gray`` is a uint8 window in measurement-space pixels; ``center`` and
    radii are expressed in the same window.  ``point_filter`` may veto edge
    points (e.g. those outside the operator's ROI).
    """
    q = QualityReport(n_rays=params.n_rays)
    band_mask = annulus_mask(gray.shape, center, r_inner, r_outer)
    if valid_mask is not None:
        band_mask &= valid_mask > 0
    q.saturation = saturation_fraction(gray, band_mask)

    processed = enhance(gray, params)
    edges = extract_radial_edges(
        processed, center, r_inner, r_outer, n_rays=params.n_rays,
        polarity=effective_polarity(params), min_strength=params.min_edge_strength,
        candidates_per_ray=params.candidates_per_ray, valid_mask=valid_mask)
    if point_filter is not None and len(edges.points):
        edges = edges.subset(point_filter(edges.points))
    q.rays_with_edge = edges.rays_with_edge

    if q.rays_with_edge < MIN_EDGE_RAY_FRACTION * params.n_rays:
        if q.saturation > params.max_saturation:
            reason = f"Glare: {q.saturation:.0%} of the search band is saturated"
        else:
            reason = "No clear edge in search band (part absent, low contrast or blurred)"
        return MeasureOutcome(False, False, reason, None, edges, q)

    band = max(r_outer - r_inner, 1.0)
    fit = ransac_circle(edges.points, threshold=params.ransac_threshold_px,
                        radius_range=(r_inner - 0.25 * band, r_outer + 0.25 * band),
                        expected_center=center, max_center_offset=max(band, 3.0))
    if fit is None:
        return MeasureOutcome(False, False, "No consistent circular boundary in search band",
                              None, edges, q)

    assess_fit(fit, edges, params, q)

    if locked_radius:
        change = abs(fit.r - locked_radius) / locked_radius
        if change > params.lock_tolerance:
            reason = (f"Confirmed ring not found: boundary in band is {change:.1%} "
                      f"from the confirmed radius")
            return MeasureOutcome(False, False, reason, fit, edges, q)

    reason = rejection_reason(q, params)
    if reason:
        return MeasureOutcome(True, False, reason, fit, edges, q)
    return MeasureOutcome(True, True, "", fit, edges, q)
