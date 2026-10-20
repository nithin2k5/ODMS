"""Robust circle fitting: RANSAC hypothesis search + geometric least squares.

The fitter only ever sees edge points that were extracted inside the
operator-selected search band, and the RANSAC hypotheses are constrained to a
radius range and a maximum centre offset.  Small circles formed by bolt holes
or unrelated features therefore cannot win even if they contribute many
points.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class CircleFit:
    cx: float
    cy: float
    r: float
    inliers: np.ndarray        # bool mask over the input points
    rms: float                 # RMS geometric residual of the inliers (px)
    max_abs_residual: float    # largest inlier residual (px)

    @property
    def n_points(self) -> int:
        return int(self.inliers.size)

    @property
    def n_inliers(self) -> int:
        return int(self.inliers.sum())

    @property
    def diameter(self) -> float:
        return 2.0 * self.r


def residuals(points: np.ndarray, cx: float, cy: float, r: float) -> np.ndarray:
    """Signed geometric distance of each point from the circle."""
    return np.hypot(points[:, 0] - cx, points[:, 1] - cy) - r


def fit_circle_algebraic(points: np.ndarray) -> tuple[float, float, float]:
    """Kasa algebraic fit; used as a starting point for geometric refinement."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        raise ValueError("at least 3 points are required")
    mean = pts.mean(axis=0)
    p = pts - mean
    x, y = p[:, 0], p[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0] / 2.0, sol[1] / 2.0
    r = math.sqrt(max(sol[2] + cx * cx + cy * cy, 0.0))
    return float(cx + mean[0]), float(cy + mean[1]), float(r)


def refine_circle_geometric(points: np.ndarray, cx: float, cy: float, r: float,
                            iterations: int = 30) -> tuple[float, float, float]:
    """Minimise the sum of squared geometric residuals (Gauss-Newton)."""
    pts = np.asarray(points, dtype=np.float64)
    params = np.array([cx, cy, r], dtype=np.float64)
    for _ in range(iterations):
        dx = pts[:, 0] - params[0]
        dy = pts[:, 1] - params[1]
        dist = np.hypot(dx, dy)
        dist = np.where(dist < 1e-9, 1e-9, dist)
        res = dist - params[2]
        J = np.column_stack([-dx / dist, -dy / dist, -np.ones_like(dist)])
        step, *_ = np.linalg.lstsq(J, -res, rcond=None)
        params += step
        if np.max(np.abs(step)) < 1e-7:
            break
    return float(params[0]), float(params[1]), float(abs(params[2]))


def _circles_through_triples(a: np.ndarray, b: np.ndarray, c: np.ndarray):
    """Vectorised circumcircles of point triples (arrays of shape (K, 2))."""
    ax, ay = a[:, 0], a[:, 1]
    bx, by = b[:, 0], b[:, 1]
    cx, cy = c[:, 0], c[:, 1]
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    valid = np.abs(d) > 1e-9
    d = np.where(valid, d, 1.0)
    a2 = ax * ax + ay * ay
    b2 = bx * bx + by * by
    c2 = cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    r = np.hypot(ax - ux, ay - uy)
    return ux, uy, r, valid


def ransac_circle(points: np.ndarray, threshold: float = 1.5,
                  max_iterations: int = 600,
                  radius_range: tuple[float, float] | None = None,
                  expected_center: tuple[float, float] | None = None,
                  max_center_offset: float | None = None,
                  refine_passes: int = 3,
                  rng: np.random.Generator | None = None) -> CircleFit | None:
    """Fit one circle robustly.

    Hypotheses whose radius falls outside ``radius_range`` or whose centre lies
    further than ``max_center_offset`` from ``expected_center`` are discarded.
    The best hypothesis (most inliers, ties broken by lower residual) is then
    refined by geometric least squares on its inliers, re-selecting inliers
    after each pass.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    n = len(pts)
    if n < 5:
        return None
    rng = rng if rng is not None else np.random.default_rng(12345)

    # Work in a centred frame for numerical stability.
    origin = pts.mean(axis=0)
    local = pts - origin
    exp_c = None if expected_center is None else np.asarray(expected_center) - origin

    best_count, best_cost, best = -1, math.inf, None
    chunk = 64
    needed = max_iterations
    done = 0
    while done < min(needed, max_iterations):
        k = min(chunk, max_iterations - done)
        idx = rng.integers(0, n, size=(k, 3))
        done += k
        ux, uy, rr, valid = _circles_through_triples(local[idx[:, 0]], local[idx[:, 1]],
                                                     local[idx[:, 2]])
        valid &= (idx[:, 0] != idx[:, 1]) & (idx[:, 1] != idx[:, 2]) & (idx[:, 0] != idx[:, 2])
        if radius_range is not None:
            valid &= (rr >= radius_range[0]) & (rr <= radius_range[1])
        if exp_c is not None and max_center_offset is not None:
            valid &= np.hypot(ux - exp_c[0], uy - exp_c[1]) <= max_center_offset
        if not np.any(valid):
            continue
        hx, hy, hr = ux[valid, None], uy[valid, None], rr[valid, None]
        res = np.abs(np.hypot(local[None, :, 0] - hx, local[None, :, 1] - hy) - hr)
        counts = (res < threshold).sum(axis=1)
        # MSAC-style cost breaks ties between equal inlier counts.
        costs = np.minimum(res, threshold).sum(axis=1)
        j = int(np.argmax(counts - costs / (threshold * n + 1.0)))
        if counts[j] > best_count or (counts[j] == best_count and costs[j] < best_cost):
            best_count, best_cost = int(counts[j]), float(costs[j])
            best = (float(hx[j, 0]), float(hy[j, 0]), float(hr[j, 0]))
            # Adaptive termination: enough samples for 99.9 % confidence.
            w = best_count / n
            if w > 0:
                p_good = max(w ** 3, 1e-12)
                needed = int(math.ceil(math.log(1 - 0.999) / math.log(max(1 - p_good, 1e-12))))
                needed = max(needed, 2 * chunk)
    if best is None or best_count < 5:
        return None

    cx, cy, r = best
    inliers = np.abs(residuals(local, cx, cy, r)) < threshold
    for _ in range(max(refine_passes, 1)):
        if inliers.sum() < 5:
            return None
        cx, cy, r = refine_circle_geometric(local[inliers], cx, cy, r)
        new_inliers = np.abs(residuals(local, cx, cy, r)) < threshold
        if np.array_equal(new_inliers, inliers):
            break
        inliers = new_inliers
    if inliers.sum() < 5:
        return None

    res = residuals(local[inliers], cx, cy, r)
    return CircleFit(cx=cx + float(origin[0]), cy=cy + float(origin[1]), r=r,
                     inliers=inliers, rms=float(np.sqrt(np.mean(res ** 2))),
                     max_abs_residual=float(np.max(np.abs(res))))


def angular_coverage(points: np.ndarray, cx: float, cy: float,
                     bins: int = 72) -> tuple[float, float]:
    """Fraction of angular bins containing points, and the largest gap (deg)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0:
        return 0.0, 360.0
    ang = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    b = np.floor((ang + math.pi) / (2 * math.pi) * bins).astype(int) % bins
    occupied = np.zeros(bins, dtype=bool)
    occupied[b] = True
    coverage = float(occupied.mean())
    if occupied.all():
        return coverage, 0.0
    # Longest circular run of empty bins.
    empty = ~occupied
    doubled = np.concatenate([empty, empty])
    longest = run = 0
    for e in doubled:
        run = run + 1 if e else 0
        longest = max(longest, run)
    longest = min(longest, bins)
    return coverage, longest * 360.0 / bins


def fit_ellipse_axis_ratio(points: np.ndarray) -> float | None:
    """Major/minor axis ratio of a direct ellipse fit, or None if not possible.

    Used only as a diagnostic: an elliptical boundary indicates the camera is
    not square to the measured plane.  The ellipse axes are never reported as
    a diameter.
    """
    import cv2

    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 6:
        return None
    try:
        (_, _), (a, b), _ = cv2.fitEllipse(pts)
    except cv2.error:
        return None
    if min(a, b) <= 0:
        return None
    return float(max(a, b) / min(a, b))
