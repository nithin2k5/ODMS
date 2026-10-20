"""Drive the real Tk window through the image-file workflow.

Skipped automatically when no display is available.
"""
import importlib
import time

import numpy as np
import pytest

tk = pytest.importorskip("tkinter")


@pytest.fixture(scope="module")
def tk_root():
    # One Tcl interpreter per process: re-creating tk.Tk() after destroying
    # one can fail on Windows ("invalid command name tcl_findLibrary").
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"no display ({exc})")
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture()
def app(tk_root, tmp_path, monkeypatch):
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
    window = mw.MainWindow(tk.Toplevel(tk_root))
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

    # Display unit: limits typed in cm are stored in mm; results shown in cm.
    app.set_unit("cm")
    assert app.results_panel.tree.heading("diameter")["text"] == "Diameter (cm)"
    app.edit_measurement(app.profile.specs[0].uid)
    app.measure_panel.min_var.set("8.99")
    app.measure_panel.max_var.set("9.01")
    app._apply_draft_fields()
    wait_current(app)
    app.confirm_draft()
    spec = app.profile.specs[0]
    assert (spec.min_mm, spec.max_mm) == (pytest.approx(89.9), pytest.approx(90.1))
    assert wait_current(app).results[0].status.value == "PASS"
    app._poll_results()
    tree = app.results_panel.tree
    rows = {tree.set(i, "measurement"): tree.set(i, "diameter") for i in tree.get_children()}
    assert float(rows["Flange outer diameter"]) == pytest.approx(9.0, abs=0.01)
    assert float(rows["Recess diameter"]) == pytest.approx(5.0, abs=0.01)
    app.set_unit("mm")

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


def test_scale_reference_workflow(app, tmp_path, monkeypatch):
    import odms.ui.main_window as mw
    from odms.storage.results import imwrite
    from odms.synthetic import Disk, flange_scene, render, straight_mapping

    def image(scale):
        s = flange_scene()
        s.shapes.append(Disk(-65.0, 0.0, 10.0, 230))
        return render(s, (1280, 960), straight_mapping(6.4 * scale, (640, 440)), seed=1)

    for name, scale in (("cal.png", 1.0), ("raised.png", 1 / 1.03)):
        imwrite(tmp_path / name, image(scale))

    answers = {"diameter_mm": 20.0, "mode": "check", "tolerance": 0.005, "max_drift": 0.1,
               "calibrate": True}
    monkeypatch.setattr(mw.ReferenceDialog, "show", lambda self: dict(answers))

    app.load_image(tmp_path / "cal.png")
    wait_current(app)
    app.new_measurement()
    app.end_mode()
    app.measure_panel.name_var.set("Flange outer diameter")
    app._apply_draft_fields()
    app.pick_edge(640 + 45 * 6.4, 441)
    wait_current(app)
    app.confirm_draft()

    # Calibrate the plane from the reference circle itself (no plane existed).
    app.start_reference()
    app._on_press(640 - 65 * 6.4 + 64.0, 440.0, None)
    out = wait_current(app)
    assert out.references and out.references[0].ok
    assert out.results[0].diameter_mm == pytest.approx(90.0, abs=0.1)

    # Camera raised 3 %: check mode blocks millimetres.
    app.load_image(tmp_path / "raised.png")
    out = wait_current(app)
    assert not out.references[0].ok
    assert out.results[0].diameter_mm is None
    assert out.results[0].status.value == "Reference check failed"

    # Live mode: scale follows the reference.
    answers.update(mode="live", calibrate=False)
    app.load_image(tmp_path / "cal.png")
    wait_current(app)
    app.start_reference()
    app._on_press(640 - 65 * 6.4 + 64.0, 440.0, None)
    app.load_image(tmp_path / "raised.png")
    out = wait_current(app)
    assert out.references[0].ok
    assert out.results[0].diameter_mm == pytest.approx(90.0, abs=0.15)


def test_auto_detect_and_distance_calibration(app, tmp_path, monkeypatch):
    import odms.ui.main_window as mw
    from odms.storage.results import imwrite
    from odms.synthetic import make_samples

    image, scene, m = make_samples()["flange"]
    ppm = m[0, 0]
    imwrite(tmp_path / "flange.png", image)

    class FakeDetect:
        def __init__(self, parent, rings, describe, on_highlight):
            self.rings = rings
            on_highlight([0])           # preview path runs without error

        def show(self):
            return [0, 1, 2]            # outer, recess, bore (largest first)

    distance, focal = 500.0, 500.0 * ppm   # pinhole model: ppm = f / Z

    class FakeDistance:
        def __init__(self, *a, **k):
            pass

        def show(self):
            return {"distance_mm": distance, "uncert_mm": 2.0, "focal_px": focal,
                    "source": "test"}

    monkeypatch.setattr(mw, "AutoDetectDialog", FakeDetect)
    monkeypatch.setattr(mw, "DistanceDialog", FakeDistance)

    app.load_image(tmp_path / "flange.png")
    wait_current(app)
    app.auto_detect_rings()
    end = time.time() + 20
    while len(app.profile.specs) < 3 and time.time() < end:
        app.root.update()
        time.sleep(0.02)
    assert [s.name for s in app.profile.specs] == ["Ring 1", "Ring 2", "Ring 3"]
    results = wait_current(app).results
    assert all(r.status.value == "Calibration required" for r in results)

    app.start_distance_calibration()
    results = wait_current(app).results
    got = [r.diameter_mm for r in results]
    assert got == [pytest.approx(90.0, abs=0.1), pytest.approx(50.0, abs=0.1),
                   pytest.approx(24.0, abs=0.1)]
    plane = app.calibration.planes["default"]
    assert plane.method == "distance" and plane.distance_mm == distance
