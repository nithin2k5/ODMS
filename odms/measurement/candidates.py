"""Circle candidates to help the operator pick rings.

Hough circles only *propose* rings: the ring the operator clicked on, or the
list offered by automatic detection.  Every proposal is confirmed by the
robust edge fit, which also supplies the diameter.  Nothing is measured
until the operator chooses the rings.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .measure import MeasureOutcome, measure_circle
from .models import ProcessingParams
from .preprocess import to_gray


def hough_candidates(image: np.ndarray, roi: tuple[int, int, int, int] | None = None,
                     max_size: int = 800) -> list[tuple[float, float, float]]:
    gray = to_gray(image)
    x0 = y0 = 0
    if roi is not None:
        x, y, w, h = roi
        x0, y0 = max(int(x), 0), max(int(y), 0)
        gray = gray[y0:y0 + int(h), x0:x0 + int(w)]
    if gray.size == 0 or min(gray.shape[:2]) < 16:
        return []
    scale = min(1.0, max_size / max(gray.shape[:2]))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) \
        if scale < 1.0 else gray
    small = cv2.GaussianBlur(small, (0, 0), 1.5)
    min_dim = min(small.shape[:2])
    max_r = int(0.75 * max(small.shape[:2]))
    if hasattr(cv2, "HOUGH_GRADIENT_ALT"):
        # Small minDist keeps concentric rings, which share a centre.
        circles = cv2.HoughCircles(small, cv2.HOUGH_GRADIENT_ALT, dp=1.5, minDist=3,
                                   param1=150, param2=0.8, minRadius=5, maxRadius=max_r)
    else:  # pragma: no cover - very old OpenCV
        circles = cv2.HoughCircles(small, cv2.HOUGH_GRADIENT, dp=1.2, minDist=4,
                                   param1=100, param2=30, minRadius=6, maxRadius=max_r)
    if circles is None:
        return []
    out = []
    for cx, cy, r in circles[0][:60]:
        if r < 0.02 * min_dim:
            continue
        out.append((cx / scale + x0, cy / scale + y0, r / scale))
    return out


def candidates_near(candidates, point, tolerance: float | None = None):
    """Candidates whose circumference passes near ``point``, in detector rank order."""
    out = []
    for cx, cy, r in candidates:
        d = abs(math.hypot(point[0] - cx, point[1] - cy) - r)
        tol = tolerance if tolerance is not None else max(6.0, 0.03 * r)
        if d <= tol:
            out.append((float(cx), float(cy), float(r)))
    return out


def nearest_candidate(candidates, point, tolerance: float | None = None):
    """Best-ranked candidate whose circumference passes near ``point``."""
    near = candidates_near(candidates, point, tolerance)
    return near[0] if near else None


def fit_ring(gray: np.ndarray, cx: float, cy: float, r_in: float, r_out: float,
             params: ProcessingParams, roi: tuple[int, int, int, int] | None = None
             ) -> MeasureOutcome:
    """Measure one band on a cropped window (much faster than the full image)."""
    h, w = gray.shape[:2]
    m = r_out + 6
    x0, y0 = max(int(cx - m), 0), max(int(cy - m), 0)
    x1, y1 = min(int(cx + m) + 2, w), min(int(cy + m) + 2, h)
    window = gray[y0:y1, x0:x1]

    def inside_roi(points):
        x, y, rw, rh = roi
        px, py = points[:, 0] + x0, points[:, 1] + y0
        return (px >= x) & (px <= x + rw) & (py >= y) & (py <= y + rh)

    out = measure_circle(window, (cx - x0, cy - y0), r_in, r_out, params,
                         point_filter=inside_roi if roi else None)
    if out.fit is not None:
        out.fit.cx += x0
        out.fit.cy += y0
    return out


def _refine(gray, cx, cy, r, params, roi, passes: int = 2):
    """Fit around a seed; the second pass re-centres the band on the first fit."""
    out = None
    for _ in range(passes):
        half = max(10.0, 0.06 * r)
        out = fit_ring(gray, cx, cy, max(r - half, 0.0), r + half, params, roi)
        if out.fit is None:
            return out, (cx, cy, r)
        cx, cy, r = out.fit.cx, out.fit.cy, out.fit.r
    return out, (cx, cy, r)


def select_ring(image: np.ndarray, click: tuple[float, float],
                roi: tuple[int, int, int, int] | None = None,
                center_hint: tuple[float, float] | None = None,
                params: ProcessingParams | None = None,
                max_seeds: int = 6) -> tuple[float, float, float] | None:
    """Find the ring through ``click``: Hough (or the operator's centre)
    proposes seeds, and the robust edge fit confirms one whose fitted
    circumference passes through the clicked point."""
    gray = to_gray(image)
    params = params or ProcessingParams(polarity="any", min_coverage=0.3)
    seeds = []
    if center_hint is not None:
        seeds.append((center_hint[0], center_hint[1],
                      math.hypot(click[0] - center_hint[0], click[1] - center_hint[1])))
    seeds += candidates_near(hough_candidates(gray, roi), click)[:max_seeds]

    best, best_score = None, -1.0
    for seed in seeds:
        out, (cx, cy, r) = _refine(gray, *seed, params, roi)
        if out is None or out.fit is None or not out.detected:
            continue
        q = out.quality
        if q.coverage < 0.5 or q.rms_px > 1.5:
            continue
        if abs(math.hypot(click[0] - cx, click[1] - cy) - r) > max(6.0, 0.03 * r):
            continue
        if q.score > best_score:
            best, best_score = (cx, cy, r), q.score
    return best


@dataclass
class DetectedRing:
    cx: float
    cy: float
    r: float
    score: float
    coverage: float
    rms: float
    edge_sign: int          # +1 rising (dark inside), -1 falling, 0 mixed
    group: int = 0          # rings sharing a centre get the same group number

    @property
    def diameter(self) -> float:
        return 2.0 * self.r

    @property
    def polarity(self) -> str:
        return {1: "rising", -1: "falling"}.get(self.edge_sign, "any")


def _same_circle(a, b) -> bool:
    (ax, ay, ar), (bx, by, br) = a, b
    return (abs(ar - br) <= max(3.0, 0.02 * max(ar, br))
            and math.hypot(ax - bx, ay - by) <= max(3.0, 0.03 * max(ar, br)))


def detect_rings(image: np.ndarray, roi: tuple[int, int, int, int] | None = None,
                 max_rings: int = 20, max_seeds: int = 60,
                 params: ProcessingParams | None = None) -> list[DetectedRing]:
    """Propose every well-defined circular boundary in the image (or ROI).

    Each Hough proposal is refined and must pass the same acceptance rules as
    a normal measurement (coverage, inliers, fit residual, blur, glare,
    ellipticity).  Rings are returned largest first and grouped by centre so
    concentric rings of one part are easy to recognise.
    """
    gray = to_gray(image)
    params = params or ProcessingParams(polarity="any")
    seeds: list[tuple[float, float, float]] = []
    for c in hough_candidates(gray, roi, max_size=900):
        if not any(_same_circle(c, s) for s in seeds):
            seeds.append(c)
    found: list[DetectedRing] = []
    for seed in seeds[:max_seeds]:
        out, (cx, cy, r) = _refine(gray, *seed, params, roi)
        if out is None or not out.accepted:
            continue
        if any(_same_circle((cx, cy, r), (f.cx, f.cy, f.r)) for f in found):
            continue
        q = out.quality
        found.append(DetectedRing(cx, cy, r, q.score, q.coverage, q.rms_px, q.edge_sign))
    found.sort(key=lambda f: -f.r)
    found = found[:max_rings]

    # Group concentric rings (largest ring of each group first).
    groups: list[tuple[float, float, float]] = []
    for ring in found:
        for gi, (gx, gy, gr) in enumerate(groups):
            if math.hypot(ring.cx - gx, ring.cy - gy) <= max(4.0, 0.04 * gr):
                ring.group = gi + 1
                break
        else:
            groups.append((ring.cx, ring.cy, ring.r))
            ring.group = len(groups)
    return found
