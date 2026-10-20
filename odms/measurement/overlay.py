"""Drawing measurement results onto the view image."""
from __future__ import annotations

import cv2
import numpy as np

from .models import SpecResult, Status

# BGR colours
GREEN = (60, 200, 60)
RED = (40, 40, 230)
ORANGE = (0, 150, 255)
YELLOW = (0, 220, 240)
CYAN = (230, 210, 0)
GREY = (170, 170, 170)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

_SHIFT = 2
_SCALE = 1 << _SHIFT


def _pts(points: np.ndarray) -> np.ndarray:
    return np.round(np.asarray(points) * _SCALE).astype(np.int32).reshape(-1, 1, 2)


def _pt(p) -> tuple[int, int]:
    return int(round(p[0] * _SCALE)), int(round(p[1] * _SCALE))


def status_colour(result: SpecResult) -> tuple[int, int, int]:
    if result.is_draft:
        return CYAN
    return {
        Status.PASS: GREEN,
        Status.MEASURED: GREEN,
        Status.FAIL: RED,
        Status.CALIBRATION_REQUIRED: YELLOW,
        Status.REVALIDATION_REQUIRED: YELLOW,
        Status.CALIBRATION_INVALID: YELLOW,
        Status.REJECTED: ORANGE,
        Status.NOT_DETECTED: RED,
    }.get(result.status, GREY)


def label_text(result: SpecResult) -> str:
    prefix = "[setup] " if result.is_draft else ""
    if result.status in (Status.NOT_DETECTED, Status.INCOMPLETE):
        return f"{prefix}{result.name}: {result.status.value}"
    if result.status is Status.REJECTED:
        return f"{prefix}{result.name}: rejected"
    if result.diameter_mm is not None:
        tail = f" {result.status.value}" if result.status in (Status.PASS, Status.FAIL) else ""
        return f"{prefix}{result.name}: {result.diameter_mm:.3f} mm{tail}"
    if result.diameter_px is not None:
        return f"{prefix}{result.name}: {result.diameter_px:.2f} px ({result.status.value})"
    return f"{prefix}{result.name}"


def _draw_label(img: np.ndarray, text: str, anchor: tuple[float, float],
                colour, scale: float, thickness: int) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), base = cv2.getTextSize(text, font, scale, thickness)
    h, w = img.shape[:2]
    x = int(np.clip(anchor[0], 2, max(2, w - tw - 4)))
    y = int(np.clip(anchor[1], th + 4, h - base - 2))
    cv2.rectangle(img, (x - 3, y - th - 4), (x + tw + 3, y + base + 1), BLACK, -1)
    cv2.putText(img, text, (x, y), font, scale, colour, thickness, cv2.LINE_AA)


def draw_results(image: np.ndarray, results: list[SpecResult],
                 rois: dict[str, tuple[int, int, int, int] | None] | None = None,
                 show_edge_points: bool = False) -> np.ndarray:
    h, w = image.shape[:2]
    t = max(1, int(round(min(h, w) / 480)))
    font_scale = max(0.45, min(h, w) / 1100)
    for res in results:
        colour = status_colour(res)
        roi = (rois or {}).get(res.uid)
        if roi is not None and res.is_draft:
            x, y, rw, rh = roi
            cv2.rectangle(image, (x, y), (x + rw, y + rh), colour, 1, cv2.LINE_AA)

        # Search band (dashed look via thin lines) is always shown for the
        # draft and for anything that is not cleanly measured.
        if res.band_outer_view is not None and (res.is_draft or not res.accepted):
            for band in (res.band_inner_view, res.band_outer_view):
                if band is not None and len(band) > 2:
                    cv2.polylines(image, [_pts(band)], True, GREY if not res.is_draft else colour,
                                  1, cv2.LINE_AA, _SHIFT)

        if show_edge_points and res.edge_points_view is not None:
            pts = res.edge_points_view
            mask = res.inlier_mask if res.inlier_mask is not None else np.ones(len(pts), bool)
            for p, inl in zip(pts, mask):
                cv2.circle(image, _pt(p), 1 * _SCALE, GREEN if inl else RED, -1,
                           cv2.LINE_AA, _SHIFT)

        if res.outline_view is not None and res.detected:
            cv2.polylines(image, [_pts(res.outline_view)], True, colour, t, cv2.LINE_AA, _SHIFT)
        if res.center_view is not None and res.detected:
            cx, cy = res.center_view
            s = 6 * t
            cv2.line(image, _pt((cx - s, cy)), _pt((cx + s, cy)), colour, t, cv2.LINE_AA, _SHIFT)
            cv2.line(image, _pt((cx, cy - s)), _pt((cx, cy + s)), colour, t, cv2.LINE_AA, _SHIFT)
        if res.diameter_line_view is not None and res.detected:
            a, b = res.diameter_line_view
            cv2.line(image, _pt(a), _pt(b), colour, max(1, t // 2 + 0), cv2.LINE_AA, _SHIFT)

        anchor_src = res.outline_view if res.outline_view is not None else res.band_outer_view
        if anchor_src is not None and len(anchor_src):
            top = anchor_src[np.argmin(anchor_src[:, 1])]
            anchor = (top[0] - 40, top[1] - 8)
        elif res.center_view is not None:
            anchor = res.center_view
        else:
            continue
        _draw_label(image, label_text(res), anchor, colour, font_scale, max(1, t))
    return image
