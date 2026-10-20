"""Live scale reference: camera height changes after calibration."""
import functools

import numpy as np
import pytest

from odms.calibration.models import CalibrationSet
from odms.calibration.reference import ScaleReference
from odms.calibration.scale import make_scale_plane
from odms.calibration.store import load_calibration, save_calibration
from odms.measurement.engine import EngineConfig, MeasurementEngine
from odms.measurement.models import MeasurementSpec, Status
from odms.synthetic import Disk, flange_scene, render, straight_mapping

CAM = "image-file"
SIZE = (1280, 960)
PPM = 6.4                 # px/mm at the calibrated height
CENTER = (640.0, 440.0)
REF_D = 20.0              # reference disc diameter, mm
REF_AT = (-65.0, 0.0)     # reference position on the plane, mm


def scene():
    s = flange_scene()
    s.shapes.append(Disk(REF_AT[0], REF_AT[1], REF_D / 2, 230))
    return s


@functools.lru_cache(maxsize=None)
def _frame(scale, seed):
    return render(scene(), SIZE, straight_mapping(PPM * scale, CENTER), seed=seed)


def frame(scale=1.0, seed=0):
    """Image after the camera moved so the magnification changed by ``scale``.

    Raising the camera by 3 % of its distance gives scale = 1 / 1.03.
    """
    return _frame(round(scale, 6), seed).copy()


def flange_spec():
    r = 45.0 * PPM
    return MeasurementSpec(name="Flange outer diameter", center=CENTER,
                           r_inner=r - 25, r_outer=r + 25)


def calibration(mode=None):
    cal = CalibrationSet(camera_id=CAM, resolution=SIZE)
    plane = make_scale_plane("default", (CENTER[0] - 50 * PPM, 600), (CENTER[0] + 50 * PPM, 600),
                             100.0, None)
    if mode:
        cx = CENTER[0] + REF_AT[0] * PPM
        r = REF_D / 2 * PPM
        ref_spec = MeasurementSpec(name="ref", center=(cx, CENTER[1]), r_inner=r - 12,
                                   r_outer=r + 12)
        plane.reference = ScaleReference(spec=ref_spec.to_dict(), diameter_mm=REF_D,
                                         mode=mode, tolerance=0.005).to_dict()
    cal.planes["default"] = plane
    return cal


def run(engine, image, cal):
    return engine.process(image, EngineConfig(specs=[flange_spec()], calibration=cal,
                                              camera_id=CAM))


RAISE = 1 / 1.03   # camera raised by 3 % of its working distance


def raise_gradually(engine, cal, steps=4, hold=2):
    """Raise the camera over ``steps`` frames, then hold it there."""
    out = None
    for i in range(steps + hold + 1):
        s = 1 + (RAISE - 1) * min(i, steps) / steps
        out = run(engine, frame(s, seed=i), cal)
    return out


def test_without_reference_height_change_goes_unnoticed():
    engine = MeasurementEngine()
    out = raise_gradually(engine, calibration())
    r = out.results[0]
    # The flange appears smaller; the stale scale under-reports by ~3 %.
    assert r.status is Status.MEASURED
    assert r.diameter_mm == pytest.approx(90.0 * RAISE, abs=0.15)


def test_check_mode_blocks_millimetres_after_height_change():
    engine = MeasurementEngine()
    cal = calibration("check")
    first = run(engine, frame(), cal)
    assert first.references[0].ok
    assert first.results[0].diameter_mm == pytest.approx(90.0, abs=0.1)

    out = raise_gradually(engine, cal)
    ref = out.references[0]
    assert not ref.ok and ref.deviation == pytest.approx(RAISE - 1, abs=0.002)
    r = out.results[0]
    assert r.status is Status.REFERENCE_FAILED
    assert r.diameter_mm is None and r.diameter_px is not None


def test_live_mode_compensates_height_change():
    engine = MeasurementEngine()
    cal = calibration("live")
    out = raise_gradually(engine, cal)
    ref = out.references[0]
    assert ref.ok and ref.px_per_mm == pytest.approx(PPM * RAISE, rel=0.002)
    r = out.results[0]
    assert r.status is Status.MEASURED
    assert r.diameter_mm == pytest.approx(90.0, abs=0.1)
    assert r.px_per_mm == pytest.approx(PPM * RAISE, rel=0.002)


def test_live_mode_rejects_excessive_drift():
    engine = MeasurementEngine()
    cal = calibration("live")
    plane = cal.planes["default"]
    ref = ScaleReference.from_dict(plane.reference)
    ref.max_drift = 0.02
    plane.reference = ref.to_dict()
    out = raise_gradually(engine, cal)
    assert not out.references[0].ok
    assert out.results[0].diameter_mm is None


def test_reference_lost_blocks_millimetres():
    engine = MeasurementEngine()
    cal = calibration("check")
    image = frame()
    x = int(CENTER[0] + REF_AT[0] * PPM)
    image[:, x - 120:x + 120] = 70          # reference covered / removed
    out = run(engine, image, cal)
    assert not out.references[0].ok
    assert out.results[0].status is Status.REFERENCE_FAILED


def test_passing_reference_counts_as_revalidation(tmp_path):
    cal = calibration("check")
    save_calibration(tmp_path / "c.json", cal)
    loaded = load_calibration(tmp_path / "c.json")   # forces revalidation
    out = run(MeasurementEngine(), frame(), loaded)
    assert out.results[0].diameter_mm == pytest.approx(90.0, abs=0.1)

    # Without a reference the same loaded calibration gives pixels only.
    plain = calibration()
    save_calibration(tmp_path / "p.json", plain)
    out = run(MeasurementEngine(), frame(), load_calibration(tmp_path / "p.json"))
    assert out.results[0].status is Status.REVALIDATION_REQUIRED


def test_reference_round_trips_through_json(tmp_path):
    cal = calibration("live")
    save_calibration(tmp_path / "c.json", cal)
    ref = ScaleReference.from_dict(load_calibration(tmp_path / "c.json")
                                   .planes["default"].reference)
    assert ref.mode == "live" and ref.diameter_mm == REF_D
    assert MeasurementSpec.from_dict(ref.spec).r_outer == pytest.approx(REF_D / 2 * PPM + 12)
