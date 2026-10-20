"""Short-window smoothing of valid readings, cleared when data goes stale."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass


@dataclass
class _Sample:
    diameter_px: float
    diameter_mm: float | None
    cx: float
    cy: float


class MeasurementSmoother:
    """Keeps a rolling window per measurement.

    The window is cleared when detection fails, when the measurement setup or
    calibration changes (``key``), when the fitted centre moves by more than
    ``move_tolerance_px`` or when the diameter jumps, so readings from a part
    that has moved or been replaced are never averaged together.
    """

    def __init__(self) -> None:
        self._windows: dict[str, deque[_Sample]] = {}
        self._keys: dict[str, str] = {}

    def reset(self, uid: str | None = None) -> None:
        if uid is None:
            self._windows.clear()
            self._keys.clear()
        else:
            self._windows.pop(uid, None)
            self._keys.pop(uid, None)

    def invalidate(self, uid: str) -> None:
        window = self._windows.get(uid)
        if window is not None:
            window.clear()

    def update(self, uid: str, key: str, diameter_px: float, diameter_mm: float | None,
               center: tuple[float, float], window: int,
               move_tolerance_px: float) -> tuple[float, float | None, int]:
        if self._keys.get(uid) != key:
            self._keys[uid] = key
            self._windows[uid] = deque(maxlen=max(window, 1))
        buf = self._windows[uid]
        if buf.maxlen != max(window, 1):
            buf = self._windows[uid] = deque(buf, maxlen=max(window, 1))
        if buf:
            last = buf[-1]
            moved = math.hypot(center[0] - last.cx, center[1] - last.cy) > move_tolerance_px
            jump_limit = max(move_tolerance_px, 0.005 * last.diameter_px)
            jumped = abs(diameter_px - last.diameter_px) > jump_limit
            if moved or jumped:
                buf.clear()
        buf.append(_Sample(diameter_px, diameter_mm, center[0], center[1]))
        mean_px = sum(s.diameter_px for s in buf) / len(buf)
        if any(s.diameter_mm is None for s in buf):
            mean_mm = None
        else:
            mean_mm = sum(s.diameter_mm for s in buf) / len(buf)
        return mean_px, mean_mm, len(buf)
