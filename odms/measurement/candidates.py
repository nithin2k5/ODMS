"""Circle candidates to help the operator pick a ring.

Hough circles are used only to suggest an approximate centre for the ring
the operator clicked on.  The diameter itself always comes from the robust
edge fit.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .measure import measure_circle
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

    def inside_roi(points):
        if roi is None:
            return np.ones(len(points), bool)
        x, y, w, h = roi
        return ((points[:, 0] >= x) & (points[:, 0] <= x + w)
                & (points[:, 1] >= y) & (points[:, 1] <= y + h))

    best, best_score = None, -1.0
    for cx, cy, r in seeds:
        out = None
        for _ in range(2):  # second pass re-centres the band on the first fit
            half = max(10.0, 0.06 * r)
            out = measure_circle(gray, (cx, cy), max(r - half, 0.0), r + half, params,
                                 point_filter=inside_roi if roi else None)
            if out.fit is None:
                break
            cx, cy, r = out.fit.cx, out.fit.cy, out.fit.r
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
