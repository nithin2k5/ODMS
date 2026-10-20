import numpy as np
import pytest

from odms.geometry import (ViewTransform, apply_homography, normalize_rect,
                           triangle_area)


@pytest.mark.parametrize("image_size,canvas_size", [
    ((1280, 720), (800, 600)),
    ((640, 480), (1920, 1080)),
    ((1920, 1080), (300, 900)),
])
def test_view_transform_round_trip(image_size, canvas_size):
    vt = ViewTransform.compute(*image_size, *canvas_size)
    for x, y in [(0, 0), (10.25, 33.5), (image_size[0] - 1, image_size[1] - 1)]:
        cx, cy = vt.to_canvas(x, y)
        bx, by = vt.to_image(cx, cy)
        assert bx == pytest.approx(x, abs=1e-9)
        assert by == pytest.approx(y, abs=1e-9)


def test_view_transform_letterbox_centres_image():
    vt = ViewTransform.compute(1000, 500, 1000, 1000)
    assert (vt.disp_w, vt.disp_h) == (1000, 500)
    assert vt.offset_y == 250
    # Canvas centre maps to image centre.
    x, y = vt.to_image(499.5, 499.5)
    assert x == pytest.approx(499.5)
    assert y == pytest.approx(249.5)


def test_view_transform_downscaled_corners():
    vt = ViewTransform.compute(2000, 1000, 1000, 500)
    # The image's outer pixel edges coincide with the displayed region's edges.
    assert vt.to_image(-0.5, -0.5) == pytest.approx((-0.5, -0.5))
    assert vt.to_image(999.5, 499.5) == pytest.approx((1999.5, 999.5))


def test_view_transform_zoom_shows_region_around_centre():
    vt = ViewTransform.compute(1000, 1000, 500, 500, zoom=4.0, center=(800, 200))
    assert vt.crop_w == 250 and vt.crop_h == 250
    x, y = vt.to_image(249.5, 249.5)
    assert x == pytest.approx(800, abs=1.0)
    assert y == pytest.approx(200, abs=1.0)


def test_view_transform_zoom_clamped_to_image():
    vt = ViewTransform.compute(1000, 1000, 500, 500, zoom=4.0, center=(990, 5))
    assert vt.crop_x + vt.crop_w == 1000
    assert vt.crop_y == 0


def test_apply_homography_identity_and_translation():
    pts = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert np.allclose(apply_homography(np.eye(3), pts), pts)
    H = np.array([[2.0, 0, 5], [0, 2.0, -1], [0, 0, 1]])
    assert np.allclose(apply_homography(H, pts), [[7, 3], [11, 7]])


def test_normalize_rect_and_triangle_area():
    assert normalize_rect(10, 20, 4, 2) == (4, 2, 6, 18)
    assert triangle_area((0, 0), (4, 0), (0, 3)) == pytest.approx(6.0)
