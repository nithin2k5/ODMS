"""Sub-pixel edge extraction along radial rays inside a search band.

Rays are cast from the approximate centre.  Along each ray the intensity
profile between ``r_inner`` and ``r_outer`` is sampled with bilinear
interpolation, differentiated, and the strongest gradient peaks with the
requested polarity are taken as edge candidates.  Peak position is refined
with a parabolic fit and the peak width (FWHM) is recorded as a blur measure.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class EdgeSamples:
    points: np.ndarray      # (N, 2) edge positions in the input image
    ray_index: np.ndarray   # (N,) index of the ray each point came from
    strength: np.ndarray    # (N,) gradient magnitude, grey levels / px
    width: np.ndarray       # (N,) FWHM of the derivative peak, px
    sign: np.ndarray        # (N,) +1 intensity rising outward, -1 falling
    n_rays: int
    rays_valid: int         # rays with at least part of the band inside the image

    @property
    def rays_with_edge(self) -> int:
        return int(np.unique(self.ray_index).size)

    def subset(self, mask: np.ndarray) -> "EdgeSamples":
        return EdgeSamples(self.points[mask], self.ray_index[mask], self.strength[mask],
                           self.width[mask], self.sign[mask], self.n_rays, self.rays_valid)

    @classmethod
    def empty(cls, n_rays: int = 0) -> "EdgeSamples":
        z = np.zeros(0)
        return cls(np.zeros((0, 2)), z.astype(int), z, z, z.astype(int), n_rays, 0)


def extract_radial_edges(image: np.ndarray, center: tuple[float, float],
                         r_inner: float, r_outer: float, *,
                         n_rays: int = 360, radial_step: float = 0.5,
                         polarity: str = "any", min_strength: float = 6.0,
                         candidates_per_ray: int = 1,
                         valid_mask: np.ndarray | None = None) -> EdgeSamples:
    """Find edge candidates on rays between ``r_inner`` and ``r_outer``.

    ``image`` is a float32 grey image.  ``valid_mask`` (uint8, same size)
    marks pixels that contain real image data; samples outside it are ignored.
    """
    r_inner = max(float(r_inner), 0.0)
    r_outer = float(r_outer)
    if r_outer - r_inner < 2 * radial_step or n_rays < 8:
        return EdgeSamples.empty(n_rays)

    radii = np.arange(r_inner, r_outer + 1e-9, radial_step, dtype=np.float64)
    angles = np.linspace(0.0, 2.0 * math.pi, n_rays, endpoint=False)
    cos_a, sin_a = np.cos(angles), np.sin(angles)
    map_x = (center[0] + cos_a[:, None] * radii[None, :]).astype(np.float32)
    map_y = (center[1] + sin_a[:, None] * radii[None, :]).astype(np.float32)

    img = image.astype(np.float32, copy=False)
    profile = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE)
    h, w = img.shape[:2]
    inside = (map_x >= 0) & (map_x <= w - 1) & (map_y >= 0) & (map_y <= h - 1)
    if valid_mask is not None:
        vm = cv2.remap(valid_mask.astype(np.float32), map_x, map_y, cv2.INTER_NEAREST,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        inside &= vm > 0.5

    deriv = np.gradient(profile, radial_step, axis=1)
    if polarity == "rising":
        score = deriv.copy()
    elif polarity == "falling":
        score = -deriv
    else:
        score = np.abs(deriv)
    # A derivative sample is valid only if its neighbours are valid too.
    ok = inside.copy()
    ok[:, 1:] &= inside[:, :-1]
    ok[:, :-1] &= inside[:, 1:]
    ok[:, 0] = False
    ok[:, -1] = False
    score = np.where(ok, score, -np.inf)
    rays_valid = int(np.count_nonzero(ok.any(axis=1)))

    pts, ray_ids, strengths, widths, signs = [], [], [], [], []
    suppress = max(2, int(round(2.0 / radial_step)))
    work = score.copy()
    for _ in range(max(1, candidates_per_ray)):
        idx = np.argmax(work, axis=1)
        rows = np.arange(n_rays)
        peak = work[rows, idx]
        keep = np.isfinite(peak) & (peak >= min_strength)
        # Peaks on the very edge of the band are unreliable (could be a
        # stronger edge just outside the band).
        keep &= (idx > 0) & (idx < len(radii) - 1)
        for ray in np.flatnonzero(keep):
            i = idx[ray]
            s = score[ray]
            y0, y1, y2 = s[i - 1], s[i], s[i + 1]
            denom = y0 - 2 * y1 + y2
            offset = 0.0
            if np.isfinite(denom) and abs(denom) > 1e-9:
                offset = float(np.clip(0.5 * (y0 - y2) / denom, -0.5, 0.5))
            r = radii[i] + offset * radial_step
            pts.append((center[0] + cos_a[ray] * r, center[1] + sin_a[ray] * r))
            ray_ids.append(ray)
            strengths.append(float(y1))
            widths.append(_peak_width(s, i, radial_step))
            signs.append(1 if deriv[ray, i] > 0 else -1)
        lo = np.clip(idx - suppress, 0, None)
        hi = np.clip(idx + suppress + 1, None, len(radii))
        for ray in range(n_rays):
            work[ray, lo[ray]:hi[ray]] = -np.inf

    if not pts:
        e = EdgeSamples.empty(n_rays)
        e.rays_valid = rays_valid
        return e
    return EdgeSamples(np.asarray(pts, dtype=np.float64), np.asarray(ray_ids, dtype=int),
                       np.asarray(strengths), np.asarray(widths),
                       np.asarray(signs, dtype=int), n_rays, rays_valid)


def _peak_width(score_row: np.ndarray, i: int, step: float) -> float:
    """Full width at half maximum of the derivative peak at index ``i``."""
    half = score_row[i] / 2.0
    left = i
    while left > 0 and np.isfinite(score_row[left - 1]) and score_row[left - 1] > half:
        left -= 1
    right = i
    n = len(score_row)
    while right < n - 1 and np.isfinite(score_row[right + 1]) and score_row[right + 1] > half:
        right += 1
    return float((right - left + 1) * step)
