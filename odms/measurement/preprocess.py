"""Grey-scale conversion, contrast adjustment and light denoising."""
from __future__ import annotations

import cv2
import numpy as np

from .models import ProcessingParams


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def enhance(gray: np.ndarray, params: ProcessingParams) -> np.ndarray:
    """Return a float32 image ready for gradient analysis."""
    img = gray if gray.dtype == np.uint8 else np.clip(gray, 0, 255).astype(np.uint8)
    if params.clahe and img.size:
        # Tile count scaled to the window so small windows are not over-tiled.
        tiles = int(np.clip(min(img.shape[:2]) // 64, 1, 8))
        clahe = cv2.createCLAHE(clipLimit=max(params.clahe_clip, 0.1),
                                tileGridSize=(tiles, tiles))
        img = clahe.apply(img)
    out = img.astype(np.float32)
    if params.blur_sigma > 0.05:
        out = cv2.GaussianBlur(out, (0, 0), params.blur_sigma)
    return out


def annulus_mask(shape: tuple[int, int], center: tuple[float, float],
                 r_inner: float, r_outer: float) -> np.ndarray:
    h, w = shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.hypot(xx - center[0], yy - center[1])
    return (d >= r_inner) & (d <= r_outer)


def saturation_fraction(gray: np.ndarray, mask: np.ndarray, level: int = 250) -> float:
    """Fraction of masked pixels at or near saturation (specular glare)."""
    count = int(mask.sum())
    if count == 0:
        return 0.0
    return float(((gray >= level) & mask).sum()) / count
