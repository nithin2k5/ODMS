"""Circle candidates to help the operator pick a ring.

Hough circles are used only to suggest an approximate centre for the ring
the operator clicked on.  The diameter itself always comes from the robust
edge fit.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

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


def nearest_candidate(candidates, point, tolerance: float | None = None):
    """Best-ranked candidate whose circumference passes near ``point``.

    Candidates are in detector rank order, so the first one within tolerance
    is preferred over a weaker one that happens to pass slightly closer.
    """
    for cx, cy, r in candidates:
        d = abs(math.hypot(point[0] - cx, point[1] - cy) - r)
        tol = tolerance if tolerance is not None else max(6.0, 0.03 * r)
        if d <= tol:
            return cx, cy, r
    return None
