import numpy as np
import pytest

from odms.measurement.candidates import detect_rings
from odms.synthetic import make_samples


@pytest.fixture(scope="module")
def samples():
    return make_samples()


def test_detects_concentric_rings_and_bolt_holes(samples):
    image, scene, m = samples["flange"]
    ppm = m[0, 0]
    rings = detect_rings(image)
    big = [r for r in rings if r.group == 1]
    assert [round(r.diameter / ppm, 1) for r in big] == [90.0, 50.0, 24.0]
    assert [r.polarity for r in big] == ["falling", "rising", "rising"]
    holes = [r for r in rings if r.group != 1]
    assert len(holes) == 6
    assert all(r.diameter / ppm == pytest.approx(9.0, abs=0.1) for r in holes)
    # Largest first.
    assert [r.r for r in rings] == sorted((r.r for r in rings), reverse=True)


def test_wheel_rings_and_roi(samples):
    image, scene, m = samples["wheel"]
    ppm = m[0, 0]
    found = [r.diameter / ppm for r in detect_rings(image)]
    for expected in (130.0, 102.5, 92.0):
        assert any(abs(d - expected) < 0.3 for d in found), (expected, found)
    # Restricting to an ROI around the bore only finds the bore.
    c = int(m[0, 2]), int(m[1, 2])
    roi = (c[0] - 40, c[1] - 40, 80, 80)
    inner = detect_rings(image, roi=roi)
    assert inner and all(r.r < 40 for r in inner)


def test_nothing_on_blank_image():
    assert detect_rings(np.full((400, 400, 3), 128, np.uint8)) == []
