"""End-to-end tests of the measurement engine on synthetic parts."""
import numpy as np
import pytest

from odms.calibration.models import CalibrationSet
from odms.calibration.perspective import make_perspective_plane
from odms.calibration.scale import make_scale_plane
from odms.geometry import apply_homography, circle_points
from odms.measurement.engine import EngineConfig, MeasurementEngine
from odms.measurement.models import MeasurementSpec, Status
from odms.synthetic import fiducials_px, make_samples

CAM = "image-file"


@pytest.fixture(scope="module")
def samples():
    return make_samples()


def spec_for(mapping, radius_mm, name, half_band_px=12.0, offset=(3.0, -2.0), **kw):
    c = apply_homography(mapping, np.array([[0.0, 0.0]]))[0]
    r = radius_mm * mapping[0, 0]
    return MeasurementSpec(name=name, center=(c[0] + offset[0], c[1] + offset[1]),
                           r_inner=r - half_band_px, r_outer=r + half_band_px, **kw)


def scale_calibration(image, scene, mapping, plane="default"):
    f = fiducials_px(scene, mapping)
    length = abs(scene.fiducials_mm[1][0] - scene.fiducials_mm[0][0])
    cal = CalibrationSet(camera_id=CAM, resolution=(image.shape[1], image.shape[0]))
    cal.planes[plane] = make_scale_plane(plane, f[0], f[1], length, None)
    return cal


def run(engine, image, specs, cal=None):
    return engine.process(image, EngineConfig(specs=specs, calibration=cal, camera_id=CAM))


def test_pixels_only_without_calibration(samples):
    image, scene, m = samples["wheel"]
    spec = spec_for(m, 65.0, "Wheel outer diameter")
    out = run(MeasurementEngine(), image, [spec]).results[0]
    assert out.status is Status.CALIBRATION_REQUIRED
    assert out.diameter_mm is None
    assert out.diameter_px == pytest.approx(130.0 * m[0, 0], abs=0.3)


@pytest.mark.parametrize("name,radius_mm", [
    ("Wheel outer diameter", 65.0),
    ("Rim diameter", 51.25),
    ("Hub diameter", 15.0),
])
def test_wheel_diameters_with_scale_calibration(samples, name, radius_mm):
    image, scene, m = samples["wheel"]
    cal = scale_calibration(image, scene, m)
    spec = spec_for(m, radius_mm, name, half_band_px=10.0)
    out = run(MeasurementEngine(), image, [spec], cal).results[0]
    assert out.status is Status.MEASURED, out.reason
    assert out.diameter_mm == pytest.approx(2 * radius_mm, abs=0.1)


def test_flange_outer_ignores_bolt_holes_in_wide_band(samples):
    image, scene, m = samples["flange"]
    cal = scale_calibration(image, scene, m)
    ppm = m[0, 0]
    # Band from 38 mm to 52 mm radius overlaps the bolt holes (30.5-39.5 mm).
    spec = spec_for(m, 45.0, "Flange outer diameter")
    spec.r_inner, spec.r_outer = 38.0 * ppm, 52.0 * ppm
    out = run(MeasurementEngine(), image, [spec], cal).results[0]
    assert out.accepted, out.reason
    assert out.diameter_mm == pytest.approx(90.0, abs=0.1)


def test_limits_produce_pass_and_fail(samples):
    image, scene, m = samples["flange"]
    cal = scale_calibration(image, scene, m)
    ok = spec_for(m, 25.0, "Recess diameter", min_mm=49.8, max_mm=50.2)
    bad = spec_for(m, 25.0, "Recess tight", min_mm=50.3, max_mm=50.6)
    results = run(MeasurementEngine(), image, [ok, bad], cal).results
    assert [r.status for r in results] == [Status.PASS, Status.FAIL]


def test_other_plane_is_not_given_millimetres(samples):
    image, scene, m = samples["flange"]
    cal = scale_calibration(image, scene, m, plane="flange face")
    spec = spec_for(m, 25.0, "Recess diameter", plane="recess floor")
    out = run(MeasurementEngine(), image, [spec], cal).results[0]
    assert out.diameter_mm is None
    assert out.status is Status.CALIBRATION_REQUIRED


def test_part_removed_reports_not_detected(samples):
    image, scene, m = samples["flange"]
    cal = scale_calibration(image, scene, m)
    spec = spec_for(m, 45.0, "Flange outer diameter")
    engine = MeasurementEngine()
    assert run(engine, image, [spec], cal).results[0].accepted
    blank = np.full_like(image, 70)
    out = run(engine, blank, [spec], cal).results[0]
    assert out.status is Status.NOT_DETECTED
    assert out.diameter_mm is None and out.diameter_px is None


def test_locked_ring_is_not_replaced_by_neighbour(samples):
    image, scene, m = samples["wheel"]
    ppm = m[0, 0]
    # The band covers the recess edge (46 mm, dark inside -> rising) and the
    # rim edge (51.25 mm, bright inside -> falling).  The operator confirmed
    # the recess edge.
    spec = spec_for(m, 48.5, "Wheel recess diameter")
    spec.r_inner, spec.r_outer = 43.0 * ppm, 54.0 * ppm
    spec.locked_radius, spec.locked_space = 46.0 * ppm, "view"

    spec.params.polarity = "rising"
    out = run(MeasurementEngine(), image, [spec]).results[0]
    assert out.accepted, out.reason
    assert out.radius_px == pytest.approx(46.0 * ppm, abs=0.3)

    # If only the neighbouring ring is visible, report "Not detected" rather
    # than silently measuring the wrong ring.
    spec.params.polarity = "falling"
    out = run(MeasurementEngine(), image, [spec]).results[0]
    assert out.status is Status.NOT_DETECTED
    assert out.diameter_px is None


def test_perspective_calibration_measures_tilted_flange(samples):
    image, scene, G = samples["flange_tilted"]
    h, w = image.shape[:2]
    dots_px = fiducials_px(scene, G)
    cal = CalibrationSet(camera_id=CAM, resolution=(w, h))
    cal.planes["default"] = make_perspective_plane("default", dots_px, scene.fiducials_mm,
                                                   (w, h), None)
    c = apply_homography(G, np.array([[0.0, 0.0]]))[0]
    ring = apply_homography(G, circle_points(0, 0, 45.0, 360))
    d = np.hypot(*(ring - c).T)
    spec = MeasurementSpec(name="Flange outer diameter", center=tuple(c),
                           r_inner=d.min() - 8, r_outer=d.max() + 8)
    out = run(MeasurementEngine(), image, [spec], cal).results[0]
    assert out.accepted, out.reason
    assert out.diameter_mm == pytest.approx(90.0, abs=0.15)

    # Without the perspective plane the same ellipse must not be reported.
    cal_scale = CalibrationSet(camera_id=CAM, resolution=(w, h))
    cal_scale.planes["default"] = make_scale_plane("default", dots_px[0], dots_px[1], 140.0,
                                                   None)
    out = run(MeasurementEngine(), image, [spec], cal_scale).results[0]
    assert not out.accepted
