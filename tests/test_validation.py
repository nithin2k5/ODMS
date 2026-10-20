import pytest

from odms.validation import Sample, summarize


def test_summarize_bias_and_repeatability():
    samples = [Sample(f"{i}.png", "A", "Outer", 50.0, v, "MEASURED", "", 90.0)
               for i, v in enumerate([50.02, 50.04, 50.00, 50.06])]
    samples.append(Sample("x.png", "A", "Outer", 50.0, None, "Not detected", "glare", None))
    (s,) = summarize(samples)
    assert s.n == 5 and s.n_measured == 4
    assert s.bias_mm == pytest.approx(0.03)
    assert s.repeatability_sd_mm == pytest.approx(0.02582, abs=1e-4)
    assert s.max_abs_error_mm == pytest.approx(0.06)
    assert s.rmse_mm == pytest.approx((0.02**2 + 0.04**2 + 0 + 0.06**2) ** 0.5 / 2)
