"""Perspective (homography) calibration for a tilted planar surface."""
from __future__ import annotations

import itertools
import math

import cv2
import numpy as np

from ..geometry import apply_homography, as_points, translation, triangle_area
from .models import PlaneCalibration, SetupInfo, now_iso

MIN_POINTS = 4


def check_point_geometry(points: np.ndarray, label: str, min_area: float) -> None:
    pts = as_points(points)
    if len(pts) < MIN_POINTS:
        raise ValueError(f"At least {MIN_POINTS} {label} points are required")
    if len(np.unique(np.round(pts, 6), axis=0)) != len(pts):
        raise ValueError(f"Duplicate {label} points")
    centred = pts - pts.mean(axis=0)
    sv = np.linalg.svd(centred, compute_uv=False)
    if sv[0] <= 0 or sv[1] / sv[0] < 0.05:
        raise ValueError(f"The {label} points are (nearly) collinear; spread them over "
                         "the plane, e.g. near the corners of the measured area")
    if len(pts) == MIN_POINTS:
        for a, b, c in itertools.combinations(range(4), 3):
            if triangle_area(pts[a], pts[b], pts[c]) < min_area:
                raise ValueError(f"Three of the four {label} points are collinear")


def local_px_per_mm(H_view_to_mm: np.ndarray, point) -> float:
    """Pixels per millimetre of the view image at ``point`` (geometric mean)."""
    eps = 0.5
    p = np.array([point, (point[0] + eps, point[1]), (point[0], point[1] + eps)])
    w = apply_homography(H_view_to_mm, p)
    J = np.column_stack([(w[1] - w[0]) / eps, (w[2] - w[0]) / eps])  # mm per px
    det = abs(np.linalg.det(J))
    return 1.0 / math.sqrt(det) if det > 0 else float("nan")


def compute_homography(image_points, world_points) -> tuple[np.ndarray, float]:
    """Least-squares homography view px -> plane mm and its RMS error in mm."""
    img = as_points(image_points)
    wld = as_points(world_points)
    if len(img) != len(wld):
        raise ValueError("Each image point needs one real-world coordinate")
    check_point_geometry(img, "image", min_area=25.0)
    check_point_geometry(wld, "world", min_area=1.0)
    H, _ = cv2.findHomography(img, wld, 0)
    if H is None:
        raise ValueError("Homography could not be computed from these points")
    err = apply_homography(H, img) - wld
    rms = float(np.sqrt(np.mean(np.sum(err ** 2, axis=1))))
    return H, rms


def make_perspective_plane(name: str, image_points, world_points,
                           view_size: tuple[int, int], lens_id: str | None,
                           setup: SetupInfo | None = None) -> PlaneCalibration:
    img = as_points(image_points)
    wld = as_points(world_points)
    H, rms = compute_homography(img, wld)

    # Keep the native resolution where the reference points are.
    ppm = local_px_per_mm(H, img.mean(axis=0))
    if not np.isfinite(ppm) or ppm <= 0:
        raise ValueError("Degenerate homography")

    # Rectified frame covers the reference area plus a generous margin,
    # clipped to what the camera can actually see.
    w, h = view_size
    corners = apply_homography(H, np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]))
    lo, hi = wld.min(axis=0), wld.max(axis=0)
    extent = hi - lo
    lo_m, hi_m = lo - extent, hi + extent
    lo = np.maximum(lo_m, corners.min(axis=0))
    hi = np.minimum(hi_m, corners.max(axis=0))
    if np.any(hi <= lo):
        lo, hi = wld.min(axis=0), wld.max(axis=0)
    S = np.diag([ppm, ppm, 1.0])
    T = S @ translation(-lo[0], -lo[1]) @ H
    out_size = (int(math.ceil((hi[0] - lo[0]) * ppm)), int(math.ceil((hi[1] - lo[1]) * ppm)))

    return PlaneCalibration(name=name, mode="perspective", px_per_mm=float(ppm),
                            lens_id=lens_id, setup=setup or SetupInfo(),
                            image_points=img.tolist(), world_points=wld.tolist(),
                            homography=H.tolist(), output_transform=T.tolist(),
                            output_size=out_size, reprojection_rms_mm=rms,
                            last_validated=now_iso())


def checkerboard_world_points(pattern_size: tuple[int, int], square_mm: float) -> np.ndarray:
    """Planar coordinates (mm) of inner corners, in OpenCV's corner order."""
    cols, rows = pattern_size
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2).astype(np.float64)
    return grid * float(square_mm)


def rectify_window(view_gray: np.ndarray, T: np.ndarray, x0: int, y0: int,
                   w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Warp the rectified-space window [x0, x0+w) x [y0, y0+h).

    Returns the window image and a mask of pixels that came from the camera.
    """
    M = translation(-x0, -y0) @ T
    win = cv2.warpPerspective(view_gray, M, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    ones = np.full(view_gray.shape[:2], 255, np.uint8)
    mask = cv2.warpPerspective(ones, M, (w, h), flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return win, mask
