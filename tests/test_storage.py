import csv

import numpy as np
import pytest

from odms.calibration.models import CalibrationSet
from odms.calibration.scale import make_scale_plane
from odms.measurement.engine import EngineConfig, MeasurementEngine
from odms.measurement.models import MeasurementSpec, ProcessingParams
from odms.storage.profiles import PartProfile, load_profile, save_profile
from odms.storage.results import CSV_FIELDS, ResultStore, imread


def test_profile_round_trip(tmp_path):
    spec = MeasurementSpec(name="Flange outer diameter", plane="face", roi=(1, 2, 300, 400),
                           center=(150.5, 200.25), r_inner=120, r_outer=140, min_mm=89.9,
                           max_mm=90.1, params=ProcessingParams(polarity="falling", n_rays=240))
    profile = PartProfile(name="Flange A", specs=[spec], calibration_id="abc", unit="in")
    path = tmp_path / "flange.json"
    save_profile(path, profile)
    loaded = load_profile(path)
    assert loaded.name == "Flange A" and loaded.unit == "in"
    s = loaded.specs[0]
    assert s.uid == spec.uid and s.roi == (1, 2, 300, 400) and s.center == (150.5, 200.25)
    assert s.params.polarity == "falling" and s.params.n_rays == 240
    assert s.fingerprint() == spec.fingerprint()
    assert loaded.unique_name("Flange outer diameter") == "Flange outer diameter (2)"


def test_save_capture_and_export(tmp_path):
    img = np.full((240, 320, 3), 60, np.uint8)
    import cv2
    cv2.circle(img, (160, 120), 80, (200, 200, 200), -1, cv2.LINE_AA)
    cal = CalibrationSet(camera_id="image-file", resolution=(320, 240))
    cal.planes["default"] = make_scale_plane("default", (0, 0), (200, 0), 50.0, None)
    spec = MeasurementSpec(name="Disk", center=(161, 119), r_inner=70, r_outer=90)
    frame = MeasurementEngine().process(
        img, EngineConfig(specs=[spec], calibration=cal, camera_id="image-file"))

    store = ResultStore(tmp_path / "results")
    rows, annotated = store.save_capture(frame, "Test part", cal, unit="cm")
    assert annotated.exists() and imread(annotated) is not None
    assert rows[0]["units"] == "mm" and rows[0]["calibration_id"] == cal.calibration_id
    assert rows[0]["display_unit"] == "cm"
    assert float(rows[0]["diameter_display"]) == pytest.approx(
        float(rows[0]["diameter_mm"]) / 10, abs=1e-3)
    store.save_capture(frame, "Test part", cal)

    out = tmp_path / "export.csv"
    assert store.export_csv(out) == 2
    with open(out, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == CSV_FIELDS
        data = list(reader)
    assert len(data) == 2 and float(data[0]["diameter_mm"]) > 0
    with open(store.log_path, encoding="utf-8-sig") as f:
        assert len(list(csv.DictReader(f))) == 2
