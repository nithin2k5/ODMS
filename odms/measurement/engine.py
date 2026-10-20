"""Frame-level measurement: calibration transforms, per-spec measurement,
smoothing, PASS/FAIL decisions and the annotated output image."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from ..calibration.lens import Undistorter
from ..calibration.models import CalibrationSet, CalState
from ..calibration.perspective import rectify_window
from ..geometry import apply_homography, circle_points
from .measure import measure_circle
from .models import MeasurementSpec, SpecResult, Status
from .overlay import draw_results
from .preprocess import to_gray
from .tracking import MeasurementSmoother

MAX_WINDOW_PX = 6000
DRAFT_UID = "__draft__"


@dataclass
class EngineConfig:
    """Immutable snapshot handed from the UI thread to the worker."""

    specs: list[MeasurementSpec] = field(default_factory=list)
    draft: MeasurementSpec | None = None
    calibration: CalibrationSet | None = None
    camera_id: str = "none"
    show_edge_points: bool = False
    version: int = 0


@dataclass
class FrameResult:
    frame_id: int
    timestamp: float
    camera_id: str
    resolution: tuple[int, int]
    raw: np.ndarray
    view: np.ndarray
    annotated: np.ndarray
    results: list[SpecResult]
    draft_result: SpecResult | None
    lens_applied: bool
    calibration_id: str | None
    processing_ms: float
    config_version: int = 0


class MeasurementEngine:
    def __init__(self) -> None:
        self.smoother = MeasurementSmoother()
        self._undistorter = Undistorter()

    def reset(self) -> None:
        self.smoother.reset()

    # -- view image -------------------------------------------------------
    def prepare_view(self, raw: np.ndarray, config: EngineConfig) -> tuple[np.ndarray, bool]:
        cal = config.calibration
        resolution = (raw.shape[1], raw.shape[0])
        if cal is not None and cal.lens_active(config.camera_id, resolution):
            return self._undistorter.apply(raw, cal.lens), True
        return raw, False

    # -- frame --------------------------------------------------------------
    def process(self, raw: np.ndarray, config: EngineConfig, frame_id: int = 0) -> FrameResult:
        t0 = time.perf_counter()
        view, lens_applied = self.prepare_view(raw, config)
        gray = to_gray(view)
        resolution = (raw.shape[1], raw.shape[0])

        results = [self.measure_spec(gray, spec, config, resolution) for spec in config.specs]
        live = {s.uid for s in config.specs}
        draft_result = None
        if config.draft is not None:
            draft_result = self.measure_spec(gray, config.draft, config, resolution,
                                             is_draft=True)
        annotated = view.copy() if view.ndim == 3 else np.dstack([view] * 3)
        drawn = results + ([draft_result] if draft_result else [])
        rois = {s.uid: s.roi for s in config.specs}
        if config.draft is not None:
            rois[DRAFT_UID] = config.draft.roi
        draw_results(annotated, drawn, rois, show_edge_points=config.show_edge_points)

        # Forget smoothing state of measurements that no longer exist.
        for uid in self.smoother.known_uids():
            if uid not in live and uid != DRAFT_UID:
                self.smoother.reset(uid)

        cal = config.calibration
        return FrameResult(frame_id=frame_id, timestamp=time.time(), camera_id=config.camera_id,
                           resolution=resolution, raw=raw, view=view, annotated=annotated,
                           results=results, draft_result=draft_result,
                           lens_applied=lens_applied,
                           calibration_id=cal.calibration_id if cal else None,
                           processing_ms=(time.perf_counter() - t0) * 1000.0,
                           config_version=config.version)

    # -- one measurement ------------------------------------------------------
    def measure_spec(self, gray: np.ndarray, spec: MeasurementSpec, config: EngineConfig,
                     resolution: tuple[int, int], is_draft: bool = False) -> SpecResult:
        uid = DRAFT_UID if is_draft else spec.uid
        res = SpecResult(uid=uid, name=spec.name, plane=spec.plane, status=Status.INCOMPLETE,
                         is_draft=is_draft, min_mm=spec.min_mm, max_mm=spec.max_mm)
        if not spec.is_complete:
            res.reason = "Select the centre and edge of the ring to measure"
            return res

        cal = config.calibration
        if cal is None:
            state, why = CalState.MISSING, "no calibration loaded"
        else:
            state, why = cal.plane_state(spec.plane, config.camera_id, resolution)
        res.calibration_state = state.value if not why else f"{state.value}: {why}"
        plane = (cal.planes.get(spec.plane)
                 if cal is not None and state in (CalState.VALID, CalState.REVALIDATE) else None)
        use_rect = plane is not None and plane.mode == "perspective"

        center = spec.center
        res.band_inner_view = circle_points(center[0], center[1], spec.r_inner, 120) \
            if spec.r_inner > 0 else None
        res.band_outer_view = circle_points(center[0], center[1], spec.r_outer, 120)

        if use_rect:
            T = plane.T
            T_inv = np.linalg.inv(T)
            space = plane.space_id
            c = apply_homography(T, np.array([center]))[0]
            outer = apply_homography(T, res.band_outer_view)
            r_out = float(np.max(np.hypot(*(outer - c).T)))
            if spec.r_inner > 0:
                inner = apply_homography(T, res.band_inner_view)
                r_in = float(np.min(np.hypot(*(inner - c).T)))
            else:
                r_in = 0.0
        else:
            T = T_inv = None
            space = "view"
            c = np.asarray(center, dtype=np.float64)
            r_in, r_out = float(spec.r_inner), float(spec.r_outer)
        res.space = space

        def to_view(points: np.ndarray) -> np.ndarray:
            return apply_homography(T_inv, points) if use_rect else np.asarray(points)

        margin = 4.0 + 3.0 * spec.params.blur_sigma
        x0 = int(math.floor(c[0] - r_out - margin))
        y0 = int(math.floor(c[1] - r_out - margin))
        x1 = int(math.ceil(c[0] + r_out + margin)) + 1
        y1 = int(math.ceil(c[1] + r_out + margin)) + 1
        if x1 - x0 > MAX_WINDOW_PX or y1 - y0 > MAX_WINDOW_PX:
            res.status, res.reason = Status.NOT_DETECTED, "Search band too large"
            return res
        if use_rect:
            window, valid = rectify_window(gray, T, x0, y0, x1 - x0, y1 - y0)
        else:
            h, w = gray.shape[:2]
            xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
            if xb - xa < 8 or yb - ya < 8:
                res.status, res.reason = Status.NOT_DETECTED, "Search band is outside the image"
                return res
            window, valid = gray[ya:yb, xa:xb], None
            x0, y0 = xa, ya
        offset = np.array([x0, y0], dtype=np.float64)

        roi = spec.roi

        def roi_filter(points_window: np.ndarray) -> np.ndarray:
            pv = to_view(points_window + offset)
            rx, ry, rw, rh = roi
            return ((pv[:, 0] >= rx) & (pv[:, 0] <= rx + rw)
                    & (pv[:, 1] >= ry) & (pv[:, 1] <= ry + rh))

        locked = spec.locked_radius if spec.locked_space == space else None
        outcome = measure_circle(window, (c[0] - x0, c[1] - y0), r_in, r_out, spec.params,
                                 valid_mask=valid, locked_radius=locked,
                                 point_filter=roi_filter if roi else None)
        res.quality = outcome.quality
        res.reason = outcome.reason
        res.detected = outcome.detected
        res.accepted = outcome.accepted
        if len(outcome.edges.points):
            res.edge_points_view = to_view(outcome.edges.points + offset)
        if outcome.fit is not None:
            fit = outcome.fit
            res.inlier_mask = fit.inliers
            cs = np.array([fit.cx, fit.cy]) + offset
            res.radius_px = fit.r
            res.outline_view = to_view(circle_points(cs[0], cs[1], fit.r, 180))
            res.center_view = tuple(to_view(cs[None, :])[0])
            ends = to_view(np.array([[cs[0] - fit.r, cs[1]], [cs[0] + fit.r, cs[1]]]))
            res.diameter_line_view = (tuple(ends[0]), tuple(ends[1]))

        if not outcome.detected:
            res.status = Status.NOT_DETECTED
            res.detected = False
            self.smoother.invalidate(uid)
            return res
        if not outcome.accepted:
            res.status = Status.REJECTED
            self.smoother.invalidate(uid)
            return res

        d_px = 2.0 * outcome.fit.r
        d_mm = d_px / plane.px_per_mm if state is CalState.VALID else None
        key = f"{spec.fingerprint()}|{space}|{cal.calibration_id if cal else ''}|{state.value}"
        sm_px, sm_mm, n = self.smoother.update(
            uid, key, d_px, d_mm, (float(cs[0]), float(cs[1])),
            spec.params.smoothing_window, spec.params.move_tolerance_px)
        res.diameter_px, res.diameter_mm, res.smoothing_count = sm_px, sm_mm, n
        res.raw_diameter_mm = d_mm

        if state is CalState.VALID:
            res.status = pass_fail(sm_mm, spec.min_mm, spec.max_mm)
            res.reason = ""
        elif state is CalState.REVALIDATE:
            res.status, res.reason = Status.REVALIDATION_REQUIRED, why
        elif state is CalState.INVALID:
            res.status, res.reason = Status.CALIBRATION_INVALID, why
        else:
            res.status, res.reason = Status.CALIBRATION_REQUIRED, why
        return res


def pass_fail(value_mm: float | None, min_mm: float | None, max_mm: float | None) -> Status:
    if value_mm is None:
        return Status.MEASURED
    if min_mm is None and max_mm is None:
        return Status.MEASURED
    if min_mm is not None and value_mm < min_mm:
        return Status.FAIL
    if max_mm is not None and value_mm > max_mm:
        return Status.FAIL
    return Status.PASS
