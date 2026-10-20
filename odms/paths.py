"""Default locations for files written by the application."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ODMS_DATA_DIR lets an installation keep data outside the source tree.
DATA_DIR = Path(os.environ.get("ODMS_DATA_DIR", PROJECT_ROOT / "data"))
CALIBRATION_DIR = DATA_DIR / "calibration"
PROFILE_DIR = DATA_DIR / "profiles"
RESULTS_DIR = DATA_DIR / "results"
SAMPLES_DIR = PROJECT_ROOT / "samples"

# The calibration that was active when the application last closed.
LAST_CALIBRATION_FILE = CALIBRATION_DIR / "last_calibration.json"


def ensure_data_dirs() -> None:
    for directory in (CALIBRATION_DIR, PROFILE_DIR, RESULTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
