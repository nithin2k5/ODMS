"""Geometry helpers shared by the UI, calibration and measurement modules.

Coordinates follow the OpenCV convention: the centre of pixel (i, j) is at
x = i, y = j.  All functions use float64 internally.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class ViewTransform:
    """Mapping between source-image pixels and a zoomed/letterboxed preview.

    A rectangular region ``crop`` of the source image is resized to
    ``disp_w`` x ``disp_h`` pixels and drawn on the canvas with its top-left
    corner at ``(offset_x, offset_y)``.
    """

    crop_x: int
    crop_y: int
    crop_w: int
    crop_h: int
    disp_w: int
    disp_h: int
    offset_x: int
    offset_y: int

    @property
    def scale_x(self) -> float:
        return self.disp_w / self.crop_w

    @property
    def scale_y(self) -> float:
        return self.disp_h / self.crop_h

    def to_image(self, cx: float, cy: float) -> tuple[float, float]:
        """Canvas pixel -> source-image pixel (sub-pixel)."""
        x = (cx - self.offset_x + 0.5) / self.scale_x - 0.5 + self.crop_x
        y = (cy - self.offset_y + 0.5) / self.scale_y - 0.5 + self.crop_y
        return x, y

    def to_canvas(self, x: float, y: float) -> tuple[float, float]:
        """Source-image pixel -> canvas pixel."""
        cx = (x - self.crop_x + 0.5) * self.scale_x - 0.5 + self.offset_x
        cy = (y - self.crop_y + 0.5) * self.scale_y - 0.5 + self.offset_y
        return cx, cy

    def length_to_canvas(self, length: float) -> float:
        return length * 0.5 * (self.scale_x + self.scale_y)

    def contains_canvas(self, cx: float, cy: float) -> bool:
        return (self.offset_x <= cx < self.offset_x + self.disp_w
                and self.offset_y <= cy < self.offset_y + self.disp_h)

    @classmethod
    def compute(cls, image_w: int, image_h: int, canvas_w: int, canvas_h: int,
                zoom: float = 1.0,
                center: tuple[float, float] | None = None) -> "ViewTransform":
        """Fit the image into the canvas, optionally zoomed about ``center``.

        ``zoom`` = 1 shows the whole image letterboxed; larger values show a
        smaller region of the image around ``center`` (image coordinates).
        """
        image_w = max(int(image_w), 1)
        image_h = max(int(image_h), 1)
        canvas_w = max(int(canvas_w), 1)
        canvas_h = max(int(canvas_h), 1)
        zoom = max(float(zoom), 1.0)
        scale = min(canvas_w / image_w, canvas_h / image_h) * zoom

        vis_w = min(float(image_w), canvas_w / scale)
        vis_h = min(float(image_h), canvas_h / scale)
        if center is None:
            center = (image_w / 2.0, image_h / 2.0)
        crop_w = min(image_w, max(1, int(math.ceil(vis_w))))
        crop_h = min(image_h, max(1, int(math.ceil(vis_h))))
        crop_x = int(round(center[0] - crop_w / 2.0))
        crop_y = int(round(center[1] - crop_h / 2.0))
        crop_x = min(max(crop_x, 0), image_w - crop_w)
        crop_y = min(max(crop_y, 0), image_h - crop_h)

        disp_w = max(1, int(round(crop_w * scale)))
        disp_h = max(1, int(round(crop_h * scale)))
        offset_x = (canvas_w - disp_w) // 2
        offset_y = (canvas_h - disp_h) // 2
        return cls(crop_x, crop_y, crop_w, crop_h, disp_w, disp_h,
                   offset_x, offset_y)


def distance(p1: Sequence[float], p2: Sequence[float]) -> float:
    return float(math.hypot(p2[0] - p1[0], p2[1] - p1[1]))


def as_points(points: Iterable[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(list(points) if not isinstance(points, np.ndarray) else points,
                     dtype=np.float64)
    return arr.reshape(-1, 2)


def apply_homography(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homography to an (N, 2) array of points."""
    pts = as_points(points)
    homog = np.hstack([pts, np.ones((len(pts), 1))]) @ np.asarray(H, dtype=np.float64).T
    w = homog[:, 2:3]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    return homog[:, :2] / w


def translation(dx: float, dy: float) -> np.ndarray:
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]])


def circle_points(cx: float, cy: float, r: float, n: int = 180) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    return np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t)])


def polygon_area(points: np.ndarray) -> float:
    pts = as_points(points)
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def triangle_area(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float:
    return 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))


def point_in_rect(x: float, y: float, rect: Sequence[float] | None) -> bool:
    if rect is None:
        return True
    rx, ry, rw, rh = rect
    return rx <= x <= rx + rw and ry <= y <= ry + rh


def normalize_rect(x0: float, y0: float, x1: float, y1: float) -> tuple[int, int, int, int]:
    """Two corner points -> integer (x, y, w, h) with positive size."""
    left, right = sorted((x0, x1))
    top, bottom = sorted((y0, y1))
    return (int(round(left)), int(round(top)),
            max(1, int(round(right - left))), max(1, int(round(bottom - top))))
