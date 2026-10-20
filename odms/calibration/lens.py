"""Checkerboard lens calibration and distortion correction."""
from __future__ import annotations

import cv2
import numpy as np

from .models import LensCalibration
from .perspective import checkerboard_world_points

MIN_VIEWS = 5
RECOMMENDED_VIEWS = 12


def find_checkerboard(gray: np.ndarray, pattern_size: tuple[int, int]) -> np.ndarray | None:
    """Sub-pixel inner-corner positions (N, 2) or None if not found."""
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    if hasattr(cv2, "findChessboardCornersSB"):
        found, corners = cv2.findChessboardCornersSB(
            gray, pattern_size, flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_ACCURACY)
        if found:
            return corners.reshape(-1, 2).astype(np.float64)
    found, corners = cv2.findChessboardCorners(
        gray, pattern_size, flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not found:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return corners.reshape(-1, 2).astype(np.float64)


def calibrate_lens(corner_sets: list[np.ndarray], pattern_size: tuple[int, int],
                   square_mm: float, image_size: tuple[int, int]) -> LensCalibration:
    if len(corner_sets) < MIN_VIEWS:
        raise ValueError(f"At least {MIN_VIEWS} checkerboard views are needed "
                         f"({RECOMMENDED_VIEWS}+ recommended, covering the whole field "
                         "of view and several tilts)")
    world = checkerboard_world_points(pattern_size, square_mm)
    obj = np.column_stack([world, np.zeros(len(world))]).astype(np.float32)
    object_points = [obj for _ in corner_sets]
    image_points = [c.reshape(-1, 1, 2).astype(np.float32) for c in corner_sets]
    rms, K, D, _, _ = cv2.calibrateCamera(object_points, image_points, tuple(image_size),
                                          None, None)
    return LensCalibration(camera_matrix=K, dist_coeffs=D.ravel(),
                           image_size=tuple(image_size), rms_px=float(rms),
                           n_views=len(corner_sets), pattern_size=tuple(pattern_size),
                           square_mm=float(square_mm))


class Undistorter:
    """Caches remap tables for one lens calibration."""

    def __init__(self) -> None:
        self._lens_id: str | None = None
        self._maps = None

    def apply(self, frame: np.ndarray, lens: LensCalibration) -> np.ndarray:
        if self._lens_id != lens.lens_id or self._maps is None:
            size = tuple(int(v) for v in lens.image_size)
            # Keep the original camera matrix so the image scale near the
            # optical centre is unchanged.
            self._maps = cv2.initUndistortRectifyMap(
                lens.camera_matrix, lens.dist_coeffs, None, lens.camera_matrix,
                size, cv2.CV_16SC2)
            self._lens_id = lens.lens_id
        return cv2.remap(frame, self._maps[0], self._maps[1], cv2.INTER_LINEAR)
