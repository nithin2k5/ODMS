import math

import numpy as np
import pytest

from odms.geometry import circle_points
from odms.measurement.circle_fit import (angular_coverage, fit_circle_algebraic,
                                         fit_ellipse_axis_ratio, ransac_circle,
                                         refine_circle_geometric)


def noisy_circle(cx, cy, r, n=360, sigma=0.2, seed=0):
    rng = np.random.default_rng(seed)
    pts = circle_points(cx, cy, r, n)
    return pts + rng.normal(0.0, sigma, pts.shape)


def test_algebraic_and_geometric_fit_recover_circle():
    pts = noisy_circle(320.0, 240.0, 150.0)
    cx, cy, r = fit_circle_algebraic(pts)
    cx, cy, r = refine_circle_geometric(pts, cx, cy, r)
    assert cx == pytest.approx(320.0, abs=0.05)
    assert cy == pytest.approx(240.0, abs=0.05)
    assert r == pytest.approx(150.0, abs=0.05)


def test_ransac_rejects_outliers_and_bolt_hole():
    good = noisy_circle(500.0, 400.0, 200.0, n=300, sigma=0.3)
    rng = np.random.default_rng(1)
    clutter = rng.uniform(250, 750, size=(120, 2))
    hole = noisy_circle(500.0, 230.0, 15.0, n=80, sigma=0.1)
    pts = np.vstack([good, clutter, hole])
    fit = ransac_circle(pts, threshold=1.5, radius_range=(150, 250),
                        expected_center=(505, 395), max_center_offset=30)
    assert fit is not None
    assert fit.r == pytest.approx(200.0, abs=0.1)
    assert fit.cx == pytest.approx(500.0, abs=0.1)
    assert fit.n_inliers >= 290
    assert fit.rms < 0.5


def test_ransac_radius_constraint_prevents_wrong_ring():
    # Only a small circle is present; the constrained search must fail rather
    # than return the wrong feature.
    pts = noisy_circle(100.0, 100.0, 20.0, n=200)
    assert ransac_circle(pts, radius_range=(80, 120)) is None


def test_angular_coverage_partial_arc():
    t = np.linspace(0, math.pi, 100)  # half circle
    pts = np.column_stack([np.cos(t) * 50, np.sin(t) * 50])
    coverage, gap = angular_coverage(pts, 0.0, 0.0, bins=72)
    assert coverage == pytest.approx(0.5, abs=0.03)
    assert gap == pytest.approx(180.0, abs=10.0)

    full = circle_points(0, 0, 50, 360)
    coverage, gap = angular_coverage(full, 0.0, 0.0)
    assert coverage == 1.0 and gap == 0.0


def test_ellipse_ratio_detects_tilt():
    circle = circle_points(0, 0, 100, 200)
    assert fit_ellipse_axis_ratio(circle) == pytest.approx(1.0, abs=1e-3)
    ellipse = circle * np.array([1.0, math.cos(math.radians(20))])
    assert fit_ellipse_axis_ratio(ellipse) == pytest.approx(1 / math.cos(math.radians(20)),
                                                            abs=1e-3)
