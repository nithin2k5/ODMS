"""Synthetic test images of circular parts with exact ground truth.

Shapes are defined in millimetres on the part plane and rendered through a
plane-to-image homography with supersampling, so edge positions are known to
sub-pixel precision.  Used by the tests, the validation example and the
sample images in ``samples/``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .geometry import apply_homography


@dataclass
class Disk:
    cx: float
    cy: float
    r: float
    value: float

    def mask(self, x, y):
        return (x - self.cx) ** 2 + (y - self.cy) ** 2 <= self.r ** 2


@dataclass
class Rect:
    x0: float
    y0: float
    x1: float
    y1: float
    value: float

    def mask(self, x, y):
        return (x >= self.x0) & (x <= self.x1) & (y >= self.y0) & (y <= self.y1)


@dataclass
class Bar:
    """Thick line segment (e.g. a wheel spoke)."""

    ax: float
    ay: float
    bx: float
    by: float
    half_width: float
    value: float

    def mask(self, x, y):
        dx, dy = self.bx - self.ax, self.by - self.ay
        L2 = dx * dx + dy * dy
        t = np.clip(((x - self.ax) * dx + (y - self.ay) * dy) / L2, 0.0, 1.0)
        px, py = self.ax + t * dx, self.ay + t * dy
        return (x - px) ** 2 + (y - py) ** 2 <= self.half_width ** 2


@dataclass
class Scene:
    background: float
    shapes: list = field(default_factory=list)
    truth_mm: dict[str, float] = field(default_factory=dict)
    fiducials_mm: list[tuple[float, float]] = field(default_factory=list)


def render(scene: Scene, size: tuple[int, int], world_to_image: np.ndarray,
           supersample: int = 4, blur_sigma: float = 0.7, noise_sigma: float = 1.5,
           shading: float = 12.0, seed: int = 0) -> np.ndarray:
    """Render ``scene`` into a BGR uint8 image of ``size`` (w, h)."""
    w, h = size
    k = supersample
    H_inv = np.linalg.inv(world_to_image)

    def shade(px: np.ndarray) -> np.ndarray:
        wp = apply_homography(H_inv, px)
        val = np.full(len(px), scene.background, np.float32)
        for shape in scene.shapes:
            val[shape.mask(wp[:, 0], wp[:, 1])] = shape.value
        return val

    # Coarse pass at pixel centres, then supersample only pixels near a
    # transition, which is where anti-aliasing matters.
    gy, gx = np.mgrid[0:h, 0:w]
    out = shade(np.column_stack([gx.ravel(), gy.ravel()]).astype(np.float64)).reshape(h, w)
    edge = (cv2.dilate(out, np.ones((3, 3), np.uint8)) != cv2.erode(out, np.ones((3, 3), np.uint8)))
    ey, ex = np.nonzero(edge)
    sub = (np.arange(k) + 0.5) / k - 0.5
    sx, sy = np.meshgrid(sub, sub)
    offsets = np.column_stack([sx.ravel(), sy.ravel()])
    chunk = 20000
    for i in range(0, len(ex), chunk):
        cx = ex[i:i + chunk].astype(np.float64)
        cy = ey[i:i + chunk].astype(np.float64)
        pts = (np.column_stack([cx, cy])[:, None, :] + offsets[None, :, :]).reshape(-1, 2)
        out[ey[i:i + chunk], ex[i:i + chunk]] = shade(pts).reshape(-1, k * k).mean(axis=1)

    rng = np.random.default_rng(seed)
    if shading:
        yy, xx = np.mgrid[0:h, 0:w]
        out += shading * ((xx / w) - 0.5) + 0.5 * shading * ((yy / h) - 0.5)
    if blur_sigma > 0:
        out = cv2.GaussianBlur(out, (0, 0), blur_sigma)
    if noise_sigma > 0:
        out += rng.normal(0.0, noise_sigma, out.shape).astype(np.float32)
    gray = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def ruler(scene: Scene, x0: float, x1: float, y: float, marks_mm: list[float]) -> None:
    """A white strip with thin black cross marks at ``marks_mm`` (x positions)."""
    scene.shapes.append(Rect(x0, y - 8, x1, y + 8, 235))
    for m in marks_mm:
        scene.shapes.append(Rect(m - 0.4, y - 5, m + 0.4, y + 5, 30))
        scene.shapes.append(Rect(m - 5, y - 0.4, m + 5, y + 0.4, 30))
        scene.fiducials_mm.append((m, y))


def wheel_scene() -> Scene:
    """Wheel-like part: tyre, rim, recess, spokes and hub (top view)."""
    s = Scene(background=200)
    s.shapes += [
        Disk(0, 0, 65.0, 50),       # tyre, outer diameter 130 mm
        Disk(0, 0, 51.25, 170),     # rim, diameter 102.5 mm
        Disk(0, 0, 46.0, 110),      # recessed wheel face, diameter 92 mm
    ]
    for i in range(5):
        a = 2 * math.pi * i / 5 + 0.3
        s.shapes.append(Bar(14 * math.cos(a), 14 * math.sin(a), 47.0 * math.cos(a),
                            47.0 * math.sin(a), 3.0, 170))
    s.shapes += [Disk(0, 0, 15.0, 170), Disk(0, 0, 5.5, 30)]
    s.truth_mm = {"Wheel outer diameter": 130.0, "Rim diameter": 102.5,
                  "Wheel recess diameter": 92.0, "Hub diameter": 30.0}
    ruler(s, -60, 60, 85, [-50.0, 50.0])
    return s


def flange_scene() -> Scene:
    """Flange plate: outer edge, bolt holes, recess and bore."""
    s = Scene(background=70)
    s.shapes.append(Disk(0, 0, 45.0, 190))          # flange outer diameter 90 mm
    for i in range(6):
        a = 2 * math.pi * i / 6
        s.shapes.append(Disk(35.0 * math.cos(a), 35.0 * math.sin(a), 4.5, 40))
    s.shapes += [Disk(0, 0, 25.0, 140),               # recess diameter 50 mm
                 Disk(0, 0, 12.0, 25)]                # bore diameter 24 mm
    s.truth_mm = {"Flange outer diameter": 90.0, "Recess diameter": 50.0,
                  "Bore diameter": 24.0, "Bolt hole diameter": 9.0}
    ruler(s, -60, 60, 62, [-50.0, 50.0])
    return s


def flange_tilted_scene() -> Scene:
    """Flange with four bright reference dots at known plane coordinates."""
    s = Scene(background=70)
    base = flange_scene()
    s.shapes = [sh for sh in base.shapes if not isinstance(sh, Rect)]
    s.truth_mm = dict(base.truth_mm)
    for x, y in [(-70, -70), (70, -70), (70, 70), (-70, 70)]:
        s.shapes.append(Disk(x, y, 3.0, 245))
        s.fiducials_mm.append((x, y))
    return s


def straight_mapping(px_per_mm: float, center_px: tuple[float, float]) -> np.ndarray:
    return np.array([[px_per_mm, 0, center_px[0]], [0, px_per_mm, center_px[1]], [0, 0, 1.0]])


def tilted_mapping() -> np.ndarray:
    """Plane mm -> image px for a view tilted roughly 30 degrees."""
    world = np.float32([[-70, -70], [70, -70], [70, 70], [-70, 70]])
    image = np.float32([[400, 215], [880, 225], [990, 745], [300, 730]])
    return cv2.getPerspectiveTransform(world, image).astype(np.float64)


def make_samples(size: tuple[int, int] = (1280, 960), px_per_mm: float = 4.0, seed: int = 0):
    """Return ``{name: (image, scene, world_to_image)}`` for the sample set."""
    samples = {}
    m = straight_mapping(px_per_mm, (size[0] / 2, size[1] / 2 - 40))
    samples["wheel"] = (render(wheel_scene(), size, m, seed=seed), wheel_scene(), m)
    m = straight_mapping(px_per_mm * 1.6, (size[0] / 2, size[1] / 2 - 40))
    samples["flange"] = (render(flange_scene(), size, m, seed=seed + 1), flange_scene(), m)
    m = tilted_mapping()
    samples["flange_tilted"] = (render(flange_tilted_scene(), size, m, seed=seed + 2),
                                flange_tilted_scene(), m)
    return samples


def fiducials_px(scene: Scene, world_to_image: np.ndarray) -> np.ndarray:
    return apply_homography(world_to_image, np.array(scene.fiducials_mm, dtype=np.float64))
