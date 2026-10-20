"""Saving and loading calibration sets as JSON."""
from __future__ import annotations

import json
from pathlib import Path

from .models import CalibrationSet


def save_calibration(path: str | Path, calset: CalibrationSet) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(calset.to_dict(), indent=2), encoding="utf-8")
    tmp.replace(path)


def load_calibration(path: str | Path, require_revalidation: bool = True) -> CalibrationSet:
    """Load a calibration set.

    The physical setup may have changed while the file sat on disk, so by
    default every plane is flagged for revalidation before it can produce
    millimetre values.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    calset = CalibrationSet.from_dict(data)
    if require_revalidation:
        calset.mark_setup_changed()
    return calset
