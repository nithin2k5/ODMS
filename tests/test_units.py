import pytest

from odms.measurement.models import SpecResult, Status
from odms.measurement.overlay import label_text
from odms.storage.profiles import PartProfile
from odms.units import UNIT_NAMES, check_unit, format_length, format_plain, from_mm, to_mm


@pytest.mark.parametrize("unit,expected", [("mm", 25.4), ("cm", 2.54), ("m", 0.0254),
                                           ("um", 25400.0), ("in", 1.0), ("mil", 1000.0)])
def test_conversions(unit, expected):
    assert from_mm(25.4, unit) == pytest.approx(expected)
    assert to_mm(expected, unit) == pytest.approx(25.4)


def test_round_trip_all_units():
    for unit in UNIT_NAMES:
        assert to_mm(from_mm(90.0123, unit), unit) == pytest.approx(90.0123)


def test_formatting_and_none():
    assert format_length(90.0, "cm") == "9.0000 cm"
    assert format_length(90.0, "mm", with_unit=False) == "90.000"
    assert format_length(None, "cm") == "-"
    assert format_plain(None, "in") == ""
    assert float(format_plain(89.9, "cm")) == pytest.approx(8.99)
    assert from_mm(None, "cm") is None and to_mm(None, "cm") is None


def test_unknown_unit_falls_back_to_mm():
    assert check_unit("furlong") == "mm"
    assert PartProfile.from_dict({"format": "odms-profile", "unit": "bogus"}).unit == "mm"
    assert PartProfile.from_dict({"format": "odms-profile", "unit": "cm"}).unit == "cm"


def test_overlay_label_uses_unit():
    r = SpecResult(uid="a", name="Outer", plane="default", status=Status.PASS,
                   diameter_mm=90.0, diameter_px=576.0, detected=True, accepted=True)
    assert label_text(r, "cm") == "Outer: 9.0000 cm PASS"
    assert label_text(r) == "Outer: 90.000 mm PASS"
