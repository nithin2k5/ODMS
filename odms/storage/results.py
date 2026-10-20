"""Saving captured results, annotated images and CSV export."""
from __future__ import annotations

import csv
import math
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from ..calibration.models import CalibrationSet
from ..measurement.engine import FrameResult
from ..measurement.models import SpecResult

CSV_FIELDS = [
    "timestamp", "part_profile", "measurement", "plane", "diameter_mm", "diameter_px",
    "units", "status", "reason", "min_mm", "max_mm", "detection_quality", "fit_rms_px",
    "inlier_ratio", "angular_coverage", "max_gap_deg", "edge_width_px", "rays_with_edge",
    "smoothing_samples", "measurement_space", "calibration_id", "plane_calibration_id",
    "calibration_state", "px_per_mm", "lens_applied", "camera_id", "resolution",
    "annotated_image", "raw_image",
]


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.{digits}f}"
    return str(value)


def imwrite(path: Path, image: np.ndarray) -> None:
    """``cv2.imwrite`` that also works with non-ASCII Windows paths."""
    ok, buf = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise IOError(f"could not encode image for {path}")
    buf.tofile(str(path))


def imread(path: str | Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def result_row(r: SpecResult, frame: FrameResult, profile_name: str,
               calibration: CalibrationSet | None, timestamp: str,
               annotated_path: str = "", raw_path: str = "") -> dict[str, str]:
    plane = calibration.planes.get(r.plane) if calibration else None
    q = r.quality
    return {
        "timestamp": timestamp,
        "part_profile": profile_name,
        "measurement": r.name,
        "plane": r.plane,
        "diameter_mm": _fmt(r.diameter_mm),
        "diameter_px": _fmt(r.diameter_px, 3),
        "units": "mm" if r.diameter_mm is not None else ("px" if r.diameter_px else ""),
        "status": r.status.value,
        "reason": r.reason,
        "min_mm": _fmt(r.min_mm),
        "max_mm": _fmt(r.max_mm),
        "detection_quality": _fmt(q.score, 1) if r.detected else "",
        "fit_rms_px": _fmt(q.rms_px, 3) if r.detected else "",
        "inlier_ratio": _fmt(q.inlier_ratio, 3) if r.detected else "",
        "angular_coverage": _fmt(q.coverage, 3) if r.detected else "",
        "max_gap_deg": _fmt(q.max_gap_deg, 1) if r.detected else "",
        "edge_width_px": _fmt(q.edge_width_px, 2) if r.detected else "",
        "rays_with_edge": str(q.rays_with_edge),
        "smoothing_samples": str(r.smoothing_count),
        "measurement_space": r.space,
        "calibration_id": calibration.calibration_id if calibration else "",
        "plane_calibration_id": plane.plane_id if plane else "",
        "calibration_state": r.calibration_state,
        "px_per_mm": _fmt(plane.px_per_mm, 6) if plane else "",
        "lens_applied": "yes" if frame.lens_applied else "no",
        "camera_id": frame.camera_id,
        "resolution": f"{frame.resolution[0]}x{frame.resolution[1]}",
        "annotated_image": annotated_path,
        "raw_image": raw_path,
    }


def write_csv(path: str | Path, rows: Iterable[dict[str, str]], append: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or not append or path.stat().st_size == 0
    with open(path, "a" if append else "w", newline="", encoding="utf-8-sig"
              if new_file else "utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name).strip("_") or "part"


class ResultStore:
    """Session results plus a persistent log in ``directory``."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.session_rows: list[dict[str, str]] = []

    @property
    def log_path(self) -> Path:
        return self.directory / "results_log.csv"

    def save_capture(self, frame: FrameResult, profile_name: str,
                     calibration: CalibrationSet | None) -> tuple[list[dict[str, str]], Path]:
        """Save annotated + raw images and append one row per measurement."""
        stamp = datetime.now()
        ts = stamp.isoformat(timespec="milliseconds")
        image_dir = self.directory / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        base = f"{stamp:%Y%m%d_%H%M%S_%f}"[:-3] + f"_{_safe(profile_name)}"
        annotated = image_dir / f"{base}_annotated.png"
        raw = image_dir / f"{base}_raw.png"
        imwrite(annotated, frame.annotated)
        imwrite(raw, frame.raw)
        rows = [result_row(r, frame, profile_name, calibration, ts, str(annotated), str(raw))
                for r in frame.results]
        write_csv(self.log_path, rows, append=True)
        self.session_rows.extend(rows)
        return rows, annotated

    def export_csv(self, path: str | Path, rows: list[dict[str, str]] | None = None) -> int:
        rows = self.session_rows if rows is None else rows
        write_csv(path, rows)
        return len(rows)
