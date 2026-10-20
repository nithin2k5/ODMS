"""Drive the real Tk window through the image-file workflow.

Skipped automatically when no display is available.
"""
import importlib
import time

import numpy as np
import pytest

tk = pytest.importorskip("tkinter")


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("ODMS_DATA_DIR", str(tmp_path / "data"))
    import odms.paths
    importlib.reload(odms.paths)
    import odms.ui.main_window as mw
    importlib.reload(mw)
    from tkinter import messagebox, simpledialog
    monkeypatch.setattr(messagebox, "askyesno", lambda *a, **k: True)
    for name in ("showinfo", "showerror", "showwarning"):
        monkeypatch.setattr(messagebox, name, lambda *a, **k: None)
    monkeypatch.setattr(simpledialog, "askfloat", lambda *a, **k: 100.0)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    window = mw.MainWindow(root)
    yield window
    window.on_close()


def wait_current(app, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        app.root.update()
        if app.latest is not None and app.latest.config_version == app._version:
            return app.latest
        time.sleep(0.01)
    raise AssertionError("no processed frame for the current configuration")


def test_image_mode_workflow(app, tmp_path):
    from odms.storage.results import imwrite
    from odms.synthetic import fiducials_px, make_samples

    image, scene, m = make_samples()["flange"]
    path = tmp_path / "flange.png"
    imwrite(path, image)
    ppm, cx, cy = m[0, 0], m[0, 2], m[1, 2]

    app.load_image(path)
    wait_current(app)

    for name, click in [("Flange outer diameter", (cx + 45 * ppm + 0.6, cy + 2)),
                        ("Recess diameter", (cx - 1, cy - 25 * ppm))]:
        app.new_measurement()
        app.end_mode()
        app.measure_panel.name_var.set(name)
        app._apply_draft_fields()
        app.pick_edge(*click)
        wait_current(app)
        app.confirm_draft()
    results = wait_current(app).results
    assert [r.status.value for r in results] == ["Calibration required"] * 2
    assert all(r.diameter_mm is None for r in results)
    assert app.profile.specs[0].locked_radius == pytest.approx(45 * ppm, abs=0.3)

    f = fiducials_px(scene, m)
    app.start_scale_calibration()
    app._on_press(*f[0], None)
    app._on_press(*f[1], None)
    results = wait_current(app).results
    assert results[0].diameter_mm == pytest.approx(90.0, abs=0.1)
    assert results[1].diameter_mm == pytest.approx(50.0, abs=0.1)

    app.mark_setup_changed()
    assert all(r.diameter_mm is None for r in wait_current(app).results)

    app.save_result()
    assert len(app.store.session_rows) == 2
    out = tmp_path / "export.csv"
    assert app.store.export_csv(out) == 2


def test_view_transform_used_for_clicks(app, tmp_path):
    from odms.storage.results import imwrite

    image = np.zeros((300, 600, 3), np.uint8)
    path = tmp_path / "blank.png"
    imwrite(path, image)
    app.load_image(path)
    wait_current(app)
    vt = app.canvas.transform
    x, y = vt.to_image(*vt.to_canvas(123.0, 45.0))
    assert (x, y) == pytest.approx((123.0, 45.0))
