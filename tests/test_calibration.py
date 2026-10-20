import math

import cv2
import numpy as np
import pytest

from odms.calibration.lens import calibrate_lens, find_checkerboard
from odms.calibration.models import CalibrationSet, CalState
from odms.calibration.perspective import (checkerboard_world_points, compute_homography,
                                          make_perspective_plane)
from odms.calibration.scale import make_scale_plane, revalidate_plane, scale_from_reference
from odms.calibration.store import load_calibration, save_calibration
from odms.geometry import apply_homography


def test_scale_from_reference():
    assert scale_from_reference((0, 0), (300, 400), 100.0) == pytest.approx(5.0)
    with pytest.raises(ValueError):
        scale_from_reference((0, 0), (10, 0), 1.0)
    with pytest.raises(ValueError):
        scale_from_reference((0, 0), (100, 0), 0.0)


def test_plane_state_rules():
    cal = CalibrationSet(camera_id="usb:0", resolution=(1280, 720))
    assert cal.plane_state("top", "usb:0", (1280, 720))[0] is CalState.MISSING
    cal.planes["top"] = make_scale_plane("top", (0, 0), (500, 0), 100.0, None)
    assert cal.plane_state("top", "usb:0", (1280, 720))[0] is CalState.VALID
    assert cal.plane_state("top", "usb:1", (1280, 720))[0] is CalState.INVALID
    assert cal.plane_state("top", "usb:0", (640, 480))[0] is CalState.INVALID
    # A different plane (e.g. a recess floor) does not inherit the top scale.
    assert cal.plane_state("recess", "usb:0", (1280, 720))[0] is CalState.MISSING
    cal.mark_setup_changed()
    assert cal.plane_state("top", "usb:0", (1280, 720))[0] is CalState.REVALIDATE


def test_revalidation_pass_and_fail():
    plane = make_scale_plane("top", (0, 0), (500, 0), 100.0, None)
    plane.requires_revalidation = True
    ok, measured, rel = revalidate_plane(plane, (10, 10), (260, 10), 50.0)
    assert ok and measured == pytest.approx(50.0) and not plane.requires_revalidation
    ok, _, rel = revalidate_plane(plane, (10, 10), (260, 10), 52.0)
    assert not ok and plane.requires_revalidation and rel < 0
    assert len(plane.validation_log) == 2


def test_save_load_forces_revalidation(tmp_path):
    cal = CalibrationSet(camera_id="usb:0", resolution=(640, 480))
    cal.planes["top"] = make_scale_plane("top", (0, 0), (400, 0), 80.0, None)
    path = tmp_path / "cal.json"
    save_calibration(path, cal)
    loaded = load_calibration(path)
    assert loaded.calibration_id == cal.calibration_id
    assert loaded.planes["top"].px_per_mm == pytest.approx(5.0)
    assert loaded.plane_state("top", "usb:0", (640, 480))[0] is CalState.REVALIDATE


def test_perspective_plane_recovers_world_distances():
    # Simulated tilted view of a plane: world mm -> image px.
    G = np.array([[3.2, 0.4, 150.0], [0.1, 2.6, 90.0], [0.0004, 0.0009, 1.0]])
    world = np.array([[0, 0], [120, 0], [120, 100], [0, 100], [60, 50]], dtype=float)
    image = apply_homography(G, world)
    plane = make_perspective_plane("tilted", image, world, (1280, 960), None)
    assert plane.reprojection_rms_mm < 1e-4
    assert plane.view_distance_mm(image[0], image[1]) == pytest.approx(120.0, rel=1e-6)
    assert plane.view_distance_mm(image[0], image[2]) == pytest.approx(math.hypot(120, 100),
                                                                       rel=1e-6)
    # Rectified space is uniformly scaled world space.
    rect = apply_homography(plane.T, image)
    d = np.linalg.norm(rect[1] - rect[0]) / plane.px_per_mm
    assert d == pytest.approx(120.0, rel=1e-6)


def test_perspective_rejects_collinear_points():
    pts = np.array([[0, 0], [10, 0], [20, 0], [30, 0.01]], dtype=float)
    with pytest.raises(ValueError):
        compute_homography(pts * 10, pts)


def _render_checkerboard(pattern=(7, 5), square_px=40):
    cols, rows = pattern[0] + 1, pattern[1] + 1
    board = np.zeros((rows * square_px, cols * square_px), np.uint8)
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                board[r * square_px:(r + 1) * square_px, c * square_px:(c + 1) * square_px] = 255
    return np.pad(board, 60, constant_values=255)


def test_lens_calibration_on_synthetic_views():
    pattern = (7, 5)
    square_px = 40
    board = _render_checkerboard(pattern, square_px)
    # Board pixel -> board mm: the first inner corner sits at (60 + 40, 60 + 40) px.
    to_mm = np.array([[0.25, 0, -25.0], [0, 0.25, -25.0], [0, 0, 1.0]])
    K = np.array([[900.0, 0, 400], [0, 900.0, 300], [0, 0, 1]])
    rng = np.random.default_rng(3)
    corner_sets = []
    for _ in range(10):
        rvec = rng.uniform(-0.35, 0.35, 3)
        rvec[2] = rng.uniform(-0.2, 0.2)
        R, _ = cv2.Rodrigues(rvec)
        t = np.array([rng.uniform(-40, 0), rng.uniform(-30, 0), rng.uniform(260, 320)])
        H_mm_to_img = K @ np.column_stack([R[:, 0], R[:, 1], t])
        M = H_mm_to_img @ to_mm
        view = cv2.warpPerspective(board, M / M[2, 2], (800, 600), flags=cv2.INTER_AREA,
                                   borderValue=255)
        corners = find_checkerboard(view, pattern)
        assert corners is not None
        corner_sets.append(corners)
    world = checkerboard_world_points(pattern, 10.0)
    assert len(world) == len(corner_sets[0])
    lens = calibrate_lens(corner_sets, pattern, 10.0, (800, 600))
    assert lens.rms_px < 0.3
    assert lens.camera_matrix[0, 0] == pytest.approx(900.0, rel=0.03)
