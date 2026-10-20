"""Main application window: wiring between UI, camera, worker and storage."""
from __future__ import annotations

import copy
import logging
import queue
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

import cv2
import numpy as np

from .. import paths
from ..calibration.lens import find_checkerboard
from ..calibration.models import CalibrationSet, CalState, LensCalibration
from ..calibration.perspective import checkerboard_world_points, make_perspective_plane
from ..calibration.scale import click_sensitivity, make_scale_plane, revalidate_plane
from ..calibration.store import load_calibration, save_calibration
from ..camera.source import (BACKENDS, RESOLUTIONS, CameraState, CameraWorker,
                             default_backend, parse_resolution, probe_cameras)
from ..geometry import apply_homography, circle_points, distance, normalize_rect
from ..measurement.candidates import select_ring
from ..measurement.engine import EngineConfig, FrameResult
from ..measurement.models import MeasurementSpec, SpecResult
from ..measurement.overlay import label_text
from ..processing import ProcessingWorker, make_queues
from ..storage.profiles import PartProfile, load_profile, save_profile
from ..storage.results import ResultStore, imread, result_row
from .calibration_panel import CalibrationPanel
from .dialogs import FormDialog, LensCalibrationDialog, PerspectivePointsDialog, run_in_background
from .measurement_panel import MeasurementPanel, ProcessingPanel
from .preview import PreviewCanvas
from .results_panel import ResultsPanel

log = logging.getLogger(__name__)

IMAGE_SOURCE_ID = "image-file"
IMAGE_TYPES = [("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All files", "*.*")]
JSON_TYPES = [("JSON", "*.json"), ("All files", "*.*")]

MODE_HINTS = {
    "roi": "Drag a rectangle around the part (region of interest). Esc cancels.",
    "center": "Click the approximate centre of the ring to measure.",
    "edge": "Click exactly on the circular edge to measure (zoom with the mouse wheel).",
    "scale_ref": "Click both ends of the reference of known length. Zoom in for precision.",
    "persp": "Click 4+ reference points on the measured plane, then press 'Finish points'.",
    "revalidate": "Click both ends of a known reference lying on the plane to revalidate.",
}


class MainWindow:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        paths.ensure_data_dirs()
        root.title("ODMS - Optical Diameter Measurement")
        root.geometry("1440x900")
        root.minsize(1000, 640)

        self.frame_queue, self.result_queue = make_queues()
        self.worker = ProcessingWorker(self.frame_queue, self.result_queue)
        self.worker.start()
        self.camera: CameraWorker | None = None
        self._old_cameras: list[CameraWorker] = []
        self.source_mode = "none"            # none | camera | frozen | image
        self.source_id = "none"
        self.calibration: CalibrationSet | None = None
        self.profile = PartProfile()
        self.profile_path: Path | None = None
        self.draft: MeasurementSpec | None = None
        self.editing_uid: str | None = None
        self.show_edge_points = False
        self.latest: FrameResult | None = None
        self.store = ResultStore(paths.RESULTS_DIR)
        self.mode: str | None = None
        self.mode_points: list[tuple[float, float]] = []
        self._drag_start: tuple[float, float] | None = None
        self._drag_now: tuple[float, float] | None = None
        self._hover: tuple[float, float] | None = None
        self._version = 0
        self._configs: dict[int, EngineConfig] = {}
        self._fields_job = None

        self._build_ui()
        self._load_last_calibration()
        self.refresh_all()
        self.push_config()
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.after(15, self._poll_results)
        root.after(300, self._poll_status)

    # ================================================================== UI build
    def _build_ui(self) -> None:
        root = self.root
        style = ttk.Style(root)
        if "vista" not in style.theme_names() and "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Treeview", rowheight=22)

        self._build_menu()
        self._build_toolbar()

        status = ttk.Frame(root, padding=(6, 2))
        status.pack(fill="x", side="bottom")
        main = ttk.PanedWindow(root, orient="horizontal")
        main.pack(fill="both", expand=True)
        left = ttk.PanedWindow(main, orient="vertical")
        main.add(left, weight=4)
        self.canvas = PreviewCanvas(left, width=900, height=600)
        left.add(self.canvas, weight=4)
        self.results_panel = ResultsPanel(left, self.save_result, self.export_csv)
        left.add(self.results_panel, weight=1)

        right = ttk.Notebook(main, width=440)
        main.add(right, weight=0)
        self.measure_panel = MeasurementPanel(right, self)
        self.cal_panel = CalibrationPanel(right, self)
        self.proc_panel = ProcessingPanel(right, self)
        right.add(self._scrollable(right, self.measure_panel), text="Measurements")
        right.add(self.cal_panel, text="Calibration")
        right.add(self.proc_panel, text="Processing")

        self.status_var = tk.StringVar(value="Ready. Start a camera or load an image.")
        ttk.Label(status, textvariable=self.status_var, anchor="w").pack(side="left", fill="x",
                                                                        expand=True)
        self.coord_var = tk.StringVar()
        ttk.Label(status, textvariable=self.coord_var, width=30, anchor="e").pack(side="right")
        ttk.Label(status, textvariable=self.fps_var, anchor="e").pack(side="right", padx=8)
        ttk.Label(status, textvariable=self.cam_status_var, anchor="e").pack(side="right",
                                                                              padx=8)

        c = self.canvas
        c.on_press = self._on_press
        c.on_drag = self._on_drag
        c.on_release = self._on_release
        c.on_motion = self._on_motion
        root.bind("<Escape>", lambda e: self.end_mode("Cancelled."))
        root.bind("<Control-s>", lambda e: self.save_result())
        root.bind("<F5>", lambda e: self.toggle_freeze())

    def _scrollable(self, parent, inner: ttk.Frame) -> ttk.Frame:
        # The measurement panel is tall; it stays usable on small screens.
        return inner

    def _build_menu(self) -> None:
        m = tk.Menu(self.root)
        f = tk.Menu(m, tearoff=False)
        f.add_command(label="Load Image...", command=self.load_image_dialog)
        samples = sorted(paths.SAMPLES_DIR.glob("*.png")) if paths.SAMPLES_DIR.exists() else []
        if samples:
            s = tk.Menu(f, tearoff=False)
            for p in samples:
                s.add_command(label=p.name, command=lambda p=p: self.load_image(p))
            f.add_cascade(label="Load Sample Image", menu=s)
        f.add_separator()
        f.add_command(label="New Profile", command=self.new_profile)
        f.add_command(label="Open Profile...", command=self.open_profile_dialog)
        f.add_command(label="Save Profile", command=self.save_profile)
        f.add_command(label="Save Profile As...", command=lambda: self.save_profile(True))
        f.add_separator()
        f.add_command(label="Load Calibration...", command=self.load_calibration_dialog)
        f.add_command(label="Save Calibration...", command=self.save_calibration_dialog)
        f.add_separator()
        f.add_command(label="Save Result (Ctrl+S)", command=self.save_result)
        f.add_command(label="Export CSV...", command=self.export_csv)
        f.add_separator()
        f.add_command(label="Exit", command=self.on_close)
        m.add_cascade(label="File", menu=f)
        v = tk.Menu(m, tearoff=False)
        v.add_command(label="Fit image", command=lambda: self.canvas.reset_view())
        v.add_command(label="Zoom in", command=lambda: self.canvas.zoom_by(1.5))
        v.add_command(label="Zoom out", command=lambda: self.canvas.zoom_by(1 / 1.5))
        m.add_cascade(label="View", menu=v)
        h = tk.Menu(m, tearoff=False)
        h.add_command(label="Controls", command=self._show_help)
        m.add_cascade(label="Help", menu=h)
        self.root.config(menu=m)

    def _build_toolbar(self) -> None:
        self.fps_var = tk.StringVar(value="Processing: - fps")
        self.cam_status_var = tk.StringVar(value="Camera: stopped")
        bar = ttk.Frame(self.root, padding=(6, 4))
        bar.pack(fill="x")
        ttk.Label(bar, text="Camera").pack(side="left")
        self.cam_var = tk.StringVar(value="0")
        self.cam_combo = ttk.Combobox(bar, textvariable=self.cam_var, width=4,
                                      values=[str(i) for i in range(6)])
        self.cam_combo.pack(side="left", padx=(4, 2))
        ttk.Button(bar, text="Scan", width=6, command=self.scan_cameras).pack(side="left")
        ttk.Label(bar, text="Backend").pack(side="left", padx=(10, 0))
        self.backend_var = tk.StringVar(value=default_backend())
        ttk.Combobox(bar, textvariable=self.backend_var, values=list(BACKENDS), width=14,
                     state="readonly").pack(side="left", padx=4)
        ttk.Label(bar, text="Resolution").pack(side="left", padx=(6, 0))
        self.res_var = tk.StringVar(value="1280x960")
        ttk.Combobox(bar, textvariable=self.res_var, values=RESOLUTIONS, width=10).pack(
            side="left", padx=4)
        ttk.Button(bar, text="Start", command=self.start_camera).pack(side="left", padx=(8, 2))
        ttk.Button(bar, text="Stop", command=self.stop_camera).pack(side="left", padx=2)
        self.freeze_btn = ttk.Button(bar, text="Freeze Frame", command=self.toggle_freeze)
        self.freeze_btn.pack(side="left", padx=2)
        ttk.Button(bar, text="Load Image...", command=self.load_image_dialog).pack(side="left",
                                                                                    padx=2)
        ttk.Button(bar, text="Fit", width=4, command=lambda: self.canvas.reset_view()).pack(
            side="left", padx=(8, 2))

    def _show_help(self) -> None:
        messagebox.showinfo("Controls", (
            "Mouse wheel: zoom about the cursor\nRight/middle drag: pan\nFit: show whole image\n"
            "Esc: cancel the current marking step\nF5: freeze / resume live\n"
            "Ctrl+S: save result\n\nSee README.md for the full workflow."), parent=self.root)

    def set_status(self, text: str) -> None:
        self.status_var.set(text)

    # ============================================================= config push
    def push_config(self) -> None:
        """Send a fresh, independent snapshot of everything the worker needs."""
        self._version += 1
        specs = [s.copy() for s in self.profile.specs if s.uid != self.editing_uid]
        draft = self.draft.copy() if self.draft is not None else None
        cfg = EngineConfig(specs=specs, draft=draft,
                           calibration=copy.deepcopy(self.calibration),
                           camera_id=self.source_id, show_edge_points=self.show_edge_points,
                           version=self._version)
        self._configs[self._version] = cfg
        for v in [v for v in self._configs if v < self._version - 20]:
            del self._configs[v]
        self.worker.set_config(cfg)

    def config_for(self, frame: FrameResult) -> EngineConfig | None:
        return self._configs.get(frame.config_version)

    def profile_planes(self) -> list[str]:
        return sorted({s.plane for s in self.profile.specs}
                      | ({self.draft.plane} if self.draft else set()))

    def refresh_all(self) -> None:
        planes = sorted(set(self.calibration.planes) if self.calibration else set()
                        | {"default"} | set(self.profile_planes()))
        if self.calibration:
            planes = sorted(set(planes) | set(self.calibration.planes))
        self.measure_panel.refresh(self.profile, planes)
        self._refresh_calibration_panel()
        self._update_overlay()

    def _refresh_calibration_panel(self) -> None:
        res = self.latest.resolution if self.latest else None
        lens = self.latest.lens_applied if self.latest else False
        self.cal_panel.refresh(self.calibration, self.source_id, res, lens)

    # ================================================================== polling
    def _poll_results(self) -> None:
        latest = None
        try:
            while True:
                latest = self.result_queue.get_nowait()
        except queue.Empty:
            pass
        if latest is not None:
            first = self.latest is None or self.latest.resolution != latest.resolution
            self.latest = latest
            self.canvas.set_image(latest.annotated)
            self.results_panel.update_results(latest.results)
            self._update_draft_status(latest.draft_result)
            if first:
                self._refresh_calibration_panel()
        self.root.after(15, self._poll_results)

    def _poll_status(self) -> None:
        cam = self.camera
        if cam is not None:
            st = cam.status
            res = f" {st.resolution[0]}x{st.resolution[1]}" if st.resolution else ""
            text = f"Camera {cam.index}: {st.state.value}{res}"
            if st.state is CameraState.RUNNING:
                text += f" @ {st.fps:.1f} fps"
            elif st.message:
                text += f" - {st.message}"
            self.cam_status_var.set(text)
        elif self.source_mode == "image":
            self.cam_status_var.set("Image file mode")
        else:
            self.cam_status_var.set("Camera: stopped")
        mode = {"camera": "live", "frozen": "frozen frame", "image": "image file",
                "none": "idle"}[self.source_mode]
        ms = f", {self.latest.processing_ms:.0f} ms" if self.latest else ""
        self.fps_var.set(f"Processing ({mode}): {self.worker.fps:.1f} fps{ms}")
        err = self.worker.take_error()
        if err:
            self.set_status("Processing error: " + err.strip().splitlines()[-1])
        self._refresh_calibration_panel()
        self.root.after(500, self._poll_status)

    def _update_draft_status(self, r: SpecResult | None) -> None:
        if self.draft is None:
            return
        if r is None:
            self.measure_panel.set_draft_status("Waiting for an image...")
            return
        text = label_text(r)
        if r.detected:
            q = r.quality
            text += (f"\nDetection quality {q.score:.0f}/100: coverage {q.coverage:.0%}, "
                     f"inliers {q.inlier_ratio:.0%}, RMS {q.rms_px:.2f} px, "
                     f"edge width {q.edge_width_px:.1f} px")
        if r.reason:
            text += f"\n{r.reason}"
        self.measure_panel.set_draft_status(text)

    # =================================================================== camera
    def scan_cameras(self) -> None:
        self.set_status("Scanning for cameras...")
        backend = self.backend_var.get()

        def done(found, error):
            if error is not None:
                self.set_status(f"Camera scan failed: {error}")
                return
            if found:
                self.cam_combo.configure(values=[str(i) for i in found])
                if self.cam_var.get() not in {str(i) for i in found}:
                    self.cam_var.set(str(found[0]))
                self.set_status(f"Cameras found: {', '.join(map(str, found))}")
            else:
                self.set_status("No cameras found. Check the connection, or use image mode.")

        if self.camera is not None:
            self.set_status("Stop the running camera before scanning.")
            return
        run_in_background(self.root, lambda: probe_cameras(6, backend), done)

    def start_camera(self) -> None:
        try:
            index = int(self.cam_var.get())
        except ValueError:
            messagebox.showerror("Camera", "Camera index must be a number.", parent=self.root)
            return
        self._stop_camera_thread()
        resolution = parse_resolution(self.res_var.get())
        self.camera = CameraWorker(index, self.frame_queue, self.backend_var.get(), resolution)
        self.camera.start()
        self.source_mode = "camera"
        self.source_id = self.camera.source_id
        self.worker.set_live()
        self.freeze_btn.configure(text="Freeze Frame")
        self.push_config()
        self.set_status(f"Starting camera {index}...")

    def _stop_camera_thread(self) -> None:
        if self.camera is not None:
            self.camera.stop()
            self._old_cameras.append(self.camera)
            self.camera = None

    def stop_camera(self) -> None:
        if self.camera is None:
            return
        self._stop_camera_thread()
        # Keep working on the last frame so set-up can continue.
        if self.worker.freeze():
            self.source_mode = "frozen"
            self.set_status("Camera stopped. Showing the last frame.")
        else:
            self.source_mode = "none"
            self.set_status("Camera stopped.")
        self.freeze_btn.configure(text="Freeze Frame")

    def toggle_freeze(self) -> None:
        if self.source_mode == "camera":
            if self.worker.freeze():
                self.source_mode = "frozen"
                self.freeze_btn.configure(text="Resume Live")
                self.set_status("Frame frozen. Mark points or set up measurements on it.")
        elif self.source_mode == "frozen" and self.camera is not None:
            self.worker.set_live()
            self.source_mode = "camera"
            self.freeze_btn.configure(text="Freeze Frame")
            self.set_status("Live.")

    def load_image_dialog(self) -> None:
        path = filedialog.askopenfilename(parent=self.root, title="Load image",
                                          filetypes=IMAGE_TYPES,
                                          initialdir=str(paths.SAMPLES_DIR)
                                          if paths.SAMPLES_DIR.exists() else None)
        if path:
            self.load_image(Path(path))

    def load_image(self, path: Path) -> None:
        image = imread(path)
        if image is None:
            messagebox.showerror("Load image", f"Could not read {path}", parent=self.root)
            return
        self._stop_camera_thread()
        self.source_mode = "image"
        self.source_id = IMAGE_SOURCE_ID
        self.worker.set_static_frame(image, IMAGE_SOURCE_ID)
        self.freeze_btn.configure(text="Freeze Frame")
        self.push_config()
        self.root.title(f"ODMS - {path.name}")
        self.set_status(f"Loaded {path.name} ({image.shape[1]}x{image.shape[0]}).")

    # ============================================================ canvas modes
    def begin_mode(self, mode: str) -> None:
        if self.latest is None:
            messagebox.showinfo("No image", "Start a camera or load an image first.",
                                parent=self.root)
            return
        if mode in ("roi", "center", "edge") and self.draft is None:
            messagebox.showinfo("Measurement", "Press Add (or Edit) to set up a measurement "
                                "first.", parent=self.root)
            return
        self.mode = mode
        self.mode_points = []
        self._drag_start = self._drag_now = None
        self.canvas.configure(cursor="crosshair")
        self.cal_panel.set_point_count(0 if mode == "persp" else None)
        self.set_status(MODE_HINTS.get(mode, ""))
        self._update_overlay()

    def end_mode(self, message: str | None = None) -> None:
        self.mode = None
        self.mode_points = []
        self._drag_start = self._drag_now = None
        self.canvas.configure(cursor="")
        self.cal_panel.set_point_count(None)
        if message:
            self.set_status(message)
        self._update_overlay()

    def _inside(self, x: float, y: float) -> bool:
        size = self.canvas.image_size
        return size is not None and -0.5 <= x <= size[0] - 0.5 and -0.5 <= y <= size[1] - 0.5

    def _on_press(self, x: float, y: float, _e) -> None:
        if self.mode is None or not self._inside(x, y):
            return
        if self.mode == "roi":
            self._drag_start = self._drag_now = (x, y)
        elif self.mode == "center":
            self.draft.center = (x, y)
            self.end_mode("Centre set. Now pick the edge (step 3).")
            self._draft_geometry_changed()
        elif self.mode == "edge":
            self.end_mode()
            self.pick_edge(x, y)
        elif self.mode in ("scale_ref", "revalidate"):
            self.mode_points.append((x, y))
            if len(self.mode_points) == 2:
                pts = list(self.mode_points)
                mode = self.mode
                self.end_mode()
                if mode == "scale_ref":
                    self._finish_scale(*pts)
                else:
                    self._finish_revalidation(*pts)
                return
        elif self.mode == "persp":
            self.mode_points.append((x, y))
            self.cal_panel.set_point_count(len(self.mode_points))
        self._update_overlay()

    def _on_drag(self, x: float, y: float, _e) -> None:
        if self.mode == "roi" and self._drag_start is not None:
            self._drag_now = (x, y)
            self._update_overlay()

    def _on_release(self, x: float, y: float, _e) -> None:
        if self.mode == "roi" and self._drag_start is not None:
            size = self.canvas.image_size
            x = min(max(x, 0), size[0] - 1)
            y = min(max(y, 0), size[1] - 1)
            rect = normalize_rect(*self._drag_start, x, y)
            if rect[2] >= 8 and rect[3] >= 8:
                self.draft.roi = rect
                self.end_mode("ROI set. Now set the centre (step 2) or pick the edge (step 3).")
                self._draft_geometry_changed()
            else:
                self._drag_start = None
                self.set_status("ROI too small - drag a larger rectangle.")

    def _on_motion(self, x: float, y: float, _e) -> None:
        self._hover = (x, y)
        if self._inside(x, y):
            self.coord_var.set(f"x {x:8.1f}  y {y:8.1f}  zoom {self.canvas.zoom:.1f}x")
        if self.mode in ("scale_ref", "revalidate") and len(self.mode_points) == 1:
            self._update_overlay()

    def _update_overlay(self) -> None:
        shapes = []
        cyan, yellow = "#00e5ff", "#ffd400"
        if self.mode == "roi" and self._drag_start and self._drag_now:
            shapes.append({"type": "rect", "rect": normalize_rect(*self._drag_start,
                                                                  *self._drag_now),
                           "color": cyan, "dash": (4, 3)})
        if self.mode in ("scale_ref", "revalidate"):
            for i, p in enumerate(self.mode_points):
                shapes.append({"type": "marker", "point": p, "color": yellow,
                               "label": str(i + 1)})
            if len(self.mode_points) == 1 and self._hover:
                p0 = self.mode_points[0]
                shapes.append({"type": "line", "points": [p0, self._hover], "color": yellow,
                               "width": 1})
                self.coord_var.set(f"length {distance(p0, self._hover):.1f} px")
        if self.mode == "persp":
            for i, p in enumerate(self.mode_points):
                shapes.append({"type": "marker", "point": p, "color": yellow,
                               "label": str(i + 1)})
            if len(self.mode_points) > 1:
                shapes.append({"type": "line", "points": self.mode_points, "color": yellow,
                               "width": 1, "dash": (3, 3)})
        if self.draft is not None and self.draft.center is not None:
            shapes.append({"type": "marker", "point": self.draft.center, "color": cyan,
                           "size": 5, "width": 1})
        self.canvas.set_overlay(shapes)

    # ============================================================== measurement
    def new_measurement(self) -> None:
        if self.draft is not None and not self._discard_draft_ok():
            return
        n = len(self.profile.specs) + 1
        self.draft = MeasurementSpec(name=self.profile.unique_name(f"Diameter {n}"),
                                     plane=self.cal_panel.target_plane)
        self.editing_uid = None
        self._load_draft_into_panels()
        self.set_status("New measurement: draw an ROI (optional), set the centre (optional) "
                        "and pick the edge.")
        self.begin_mode("roi") if self.latest is not None else None

    def edit_measurement(self, uid: str) -> None:
        spec = self.profile.spec_by_uid(uid)
        if spec is None:
            return
        if self.draft is not None and not self._discard_draft_ok():
            return
        self.draft = spec.copy()
        self.draft.locked_radius = self.draft.locked_space = None  # re-confirmed on save
        self.editing_uid = uid
        self._load_draft_into_panels()
        self.push_config()
        self.set_status(f"Editing '{spec.name}'. Adjust, then Confirm & save.")

    def delete_measurement(self, uid: str) -> None:
        spec = self.profile.spec_by_uid(uid)
        if spec and messagebox.askyesno("Delete", f"Delete measurement '{spec.name}'?",
                                        parent=self.root):
            self.profile.specs = [s for s in self.profile.specs if s.uid != uid]
            if self.editing_uid == uid:
                self.cancel_draft()
            self.refresh_all()
            self.push_config()

    def move_measurement(self, uid: str, delta: int) -> None:
        specs = self.profile.specs
        i = next((k for k, s in enumerate(specs) if s.uid == uid), None)
        if i is None or not 0 <= i + delta < len(specs):
            return
        specs[i], specs[i + delta] = specs[i + delta], specs[i]
        self.refresh_all()
        self.measure_panel.tree.selection_set(uid)
        self.push_config()

    def _discard_draft_ok(self) -> bool:
        return messagebox.askyesno("Discard changes", "Discard the measurement currently being "
                                   "set up?", parent=self.root)

    def _load_draft_into_panels(self) -> None:
        self.measure_panel.load_draft(self.draft)
        self.proc_panel.load(self.draft.params if self.draft else None)
        self.refresh_all()

    def draft_fields_changed(self) -> None:
        if self._fields_job is not None:
            self.root.after_cancel(self._fields_job)
        self._fields_job = self.root.after(200, self._apply_draft_fields)

    def _apply_draft_fields(self) -> None:
        self._fields_job = None
        if self.draft is None:
            return
        problems = self.measure_panel.read_into(self.draft)
        self.proc_panel.read_into(self.draft.params)
        if problems:
            self.set_status("Check values: " + "; ".join(problems))
        self.push_config()
        self._update_overlay()

    def _draft_geometry_changed(self) -> None:
        self.measure_panel.load_draft(self.draft)
        self.push_config()
        self._update_overlay()

    def _perspective_plane_for(self, plane_name: str):
        cal = self.calibration
        if cal is None or self.latest is None:
            return None
        state, _ = cal.plane_state(plane_name, self.source_id, self.latest.resolution)
        plane = cal.planes.get(plane_name)
        if plane is not None and plane.mode == "perspective" and state in (CalState.VALID,
                                                                           CalState.REVALIDATE):
            return plane
        return None

    def pick_edge(self, x: float, y: float) -> None:
        if self.draft is None or self.latest is None:
            return
        draft = self.draft
        self.measure_panel.read_into(draft)
        half = self.measure_panel.band_half_width
        view = self.latest.view
        plane = self._perspective_plane_for(draft.plane)
        self.set_status("Searching for the ring through the clicked point...")
        self.root.update_idletasks()

        ring_view = None   # (center, r_inner, r_outer) in view coordinates
        if plane is not None and plane.output_size and \
                plane.output_size[0] * plane.output_size[1] < 30e6:
            # Find the ring on the rectified plane, where it is a true circle.
            T = plane.T
            rect = cv2.warpPerspective(view, T, tuple(int(v) for v in plane.output_size))
            click_r = tuple(apply_homography(T, np.array([[x, y]]))[0])
            hint_r = (tuple(apply_homography(T, np.array([draft.center]))[0])
                      if draft.center else None)
            ring = select_ring(rect, click_r, center_hint=hint_r)
            if ring is not None:
                T_inv = np.linalg.inv(T)
                c_view = tuple(apply_homography(T_inv, np.array([ring[:2]]))[0])
                pts = apply_homography(T_inv, circle_points(ring[0], ring[1], ring[2], 180))
                d = np.hypot(*(pts - np.array(c_view)).T)
                ring_view = (c_view, max(d.min() - half, 0.0), d.max() + half)
        else:
            ring = select_ring(view, (x, y), roi=draft.roi, center_hint=draft.center)
            if ring is not None:
                ring_view = ((ring[0], ring[1]), max(ring[2] - half, 0.0), ring[2] + half)

        if ring_view is None:
            center = draft.center
            if center is None and draft.roi is not None:
                rx, ry, rw, rh = draft.roi
                center = (rx + rw / 2.0, ry + rh / 2.0)
            if center is None:
                messagebox.showinfo("Pick edge", "No ring through that point was found "
                                    "automatically. Set the approximate centre (step 2) and "
                                    "pick the edge again.", parent=self.root)
                self.set_status("Set the approximate centre, then pick the edge again.")
                return
            r = distance(center, (x, y))
            ring_view = (center, max(r - half, 0.0), r + half)
            self.set_status("Ring not confirmed automatically; band placed around the click. "
                            "Check the preview and adjust the radii if needed.")
        else:
            self.set_status("Ring found. Check the preview (cyan), then Confirm & save.")
        draft.center, draft.r_inner, draft.r_outer = ring_view
        draft.locked_radius = draft.locked_space = None
        self._draft_geometry_changed()

    def _current_draft_result(self) -> SpecResult | None:
        if self.latest is None or self.latest.config_version != self._version:
            return None
        return self.latest.draft_result

    def recentre_draft(self) -> None:
        if self.draft is None:
            return
        r = self._current_draft_result()
        if r is None or r.center_view is None or r.outline_view is None:
            self.set_status("No fitted boundary to re-centre on (yet).")
            return
        half = self.measure_panel.band_half_width
        c = np.array(r.center_view)
        d = np.hypot(*(r.outline_view - c).T)
        self.draft.center = tuple(c)
        self.draft.r_inner = max(float(d.min()) - half, 0.0)
        self.draft.r_outer = float(d.max()) + half
        self._draft_geometry_changed()
        self.set_status("Band re-centred on the fitted boundary.")

    def confirm_draft(self) -> None:
        draft = self.draft
        if draft is None:
            return
        if self._fields_job is not None:
            self.root.after_cancel(self._fields_job)
            self._apply_draft_fields()
        problems = self.measure_panel.read_into(draft)
        if problems:
            messagebox.showerror("Measurement", "\n".join(problems), parent=self.root)
            return
        if not draft.name:
            messagebox.showerror("Measurement", "Enter a name.", parent=self.root)
            return
        if any(s.name == draft.name and s.uid != self.editing_uid for s in self.profile.specs):
            messagebox.showerror("Measurement", f"A measurement named '{draft.name}' already "
                                 "exists.", parent=self.root)
            return
        if not draft.is_complete:
            messagebox.showerror("Measurement", "Pick the edge to measure first.",
                                 parent=self.root)
            return
        r = self._current_draft_result()
        if r is None:
            self.set_status("Processing the latest change - press Confirm again in a moment.")
            return
        spec = draft.copy()
        if r.accepted and r.radius_px is not None:
            # Lock the measurement to the confirmed ring.
            spec.locked_radius, spec.locked_space = r.radius_px, r.space
            if spec.params.polarity == "auto":
                spec.params.polarity = {1: "rising", -1: "falling"}.get(r.quality.edge_sign,
                                                                        "any")
            c = np.array(r.center_view)
            d = np.hypot(*(r.outline_view - c).T)
            band = max((spec.r_outer - spec.r_inner) / 2.0, 2.0)
            spec.center = tuple(c)
            spec.r_inner = max(float(d.min()) - band, 0.0)
            spec.r_outer = float(d.max()) + band
        else:
            reason = r.reason or r.status.value
            if not messagebox.askyesno(
                    "Boundary not confirmed",
                    f"The selected boundary is not currently accepted:\n\n{reason}\n\n"
                    "Save anyway? It will not be locked to a confirmed ring, so check it "
                    "again when a good part is in view.", parent=self.root):
                return
        if self.editing_uid:
            spec.uid = self.editing_uid
            self.profile.specs = [spec if s.uid == spec.uid else s for s in self.profile.specs]
        else:
            self.profile.specs.append(spec)
        self.draft = None
        self.editing_uid = None
        self._load_draft_into_panels()
        self.push_config()
        self.set_status(f"Saved measurement '{spec.name}'. Remember to save the profile.")

    def cancel_draft(self) -> None:
        self.draft = None
        self.editing_uid = None
        self.end_mode()
        self._load_draft_into_panels()
        self.push_config()

    def set_show_edge_points(self, show: bool) -> None:
        self.show_edge_points = show
        self.push_config()

    # ================================================================== profiles
    def new_profile(self) -> None:
        if self.profile.specs and not messagebox.askyesno(
                "New profile", "Discard the current profile's unsaved changes?",
                parent=self.root):
            return
        self.profile = PartProfile()
        self.profile_path = None
        self.cancel_draft()
        self.refresh_all()

    def open_profile_dialog(self) -> None:
        path = filedialog.askopenfilename(parent=self.root, title="Open part profile",
                                          filetypes=JSON_TYPES, initialdir=str(paths.PROFILE_DIR))
        if not path:
            return
        try:
            profile = load_profile(path)
        except (OSError, ValueError, KeyError) as exc:
            messagebox.showerror("Open profile", f"Could not load profile:\n{exc}",
                                 parent=self.root)
            return
        self.profile = profile
        self.profile_path = Path(path)
        self.cancel_draft()
        self.refresh_all()
        warn = []
        cal = self.calibration
        lens_id = cal.lens.lens_id if cal and cal.lens else None
        if profile.lens_id != lens_id:
            warn.append("The profile was set up with a different lens correction; ring "
                        "positions may be offset. Check each ring and re-pick if needed.")
        if profile.camera_id and profile.camera_id != self.source_id:
            warn.append(f"The profile was set up on {profile.camera_id}; the current source is "
                        f"{self.source_id}.")
        self.set_status(f"Opened profile '{profile.name}' with {len(profile.specs)} "
                        "measurement(s).")
        if warn:
            messagebox.showwarning("Profile", "\n\n".join(warn), parent=self.root)

    def save_profile(self, as_new: bool = False) -> None:
        path = self.profile_path
        if as_new or path is None:
            name = "".join(c if c.isalnum() or c in "-_ " else "_" for c in self.profile.name)
            chosen = filedialog.asksaveasfilename(
                parent=self.root, title="Save part profile", defaultextension=".json",
                filetypes=JSON_TYPES, initialdir=str(paths.PROFILE_DIR),
                initialfile=f"{name.strip() or 'profile'}.json")
            if not chosen:
                return
            path = Path(chosen)
        cal = self.calibration
        self.profile.calibration_id = cal.calibration_id if cal else None
        self.profile.lens_id = cal.lens.lens_id if cal and cal.lens else None
        self.profile.camera_id = self.source_id
        try:
            save_profile(path, self.profile)
        except OSError as exc:
            messagebox.showerror("Save profile", str(exc), parent=self.root)
            return
        self.profile_path = path
        self.set_status(f"Profile saved to {path}")

    # =============================================================== calibration
    def _ensure_calset(self) -> CalibrationSet | None:
        if self.latest is None:
            messagebox.showinfo("Calibration", "Start a camera or load an image first.",
                                parent=self.root)
            return None
        res = self.latest.resolution
        cal = self.calibration
        if cal is not None and cal.matches_source(self.source_id, res)[0]:
            return cal
        if cal is not None and not messagebox.askyesno(
                "New calibration",
                f"The current calibration belongs to {cal.camera_id} at "
                f"{cal.resolution[0]}x{cal.resolution[1]}, not {self.source_id} at "
                f"{res[0]}x{res[1]}.\n\nStart a new calibration for the current source? "
                "(Save the old one first if you still need it.)", parent=self.root):
            return None
        self.calibration = CalibrationSet(camera_id=self.source_id, resolution=res)
        return self.calibration

    def _calibration_changed(self, new_id: bool = True) -> None:
        if self.calibration is not None:
            if new_id:
                self.calibration.touch()
            try:
                save_calibration(paths.LAST_CALIBRATION_FILE, self.calibration)
            except OSError as exc:
                log.warning("could not autosave calibration: %s", exc)
        self.refresh_all()
        self.push_config()

    def _load_last_calibration(self) -> None:
        if paths.LAST_CALIBRATION_FILE.exists():
            try:
                self.calibration = load_calibration(paths.LAST_CALIBRATION_FILE)
                self.set_status("Loaded the last calibration. Revalidate each plane with a "
                                "known reference before relying on millimetres.")
            except (OSError, ValueError, KeyError) as exc:
                log.warning("could not load last calibration: %s", exc)

    def _confirm_overwrite_plane(self, cal: CalibrationSet, name: str) -> bool:
        return name not in cal.planes or messagebox.askyesno(
            "Replace plane", f"Replace the existing calibration of plane '{name}'?",
            parent=self.root)

    def _lens_id(self, cal: CalibrationSet) -> str | None:
        return cal.lens.lens_id if cal.lens else None

    def start_scale_calibration(self) -> None:
        if self._ensure_calset() is not None:
            self.begin_mode("scale_ref")

    def _finish_scale(self, p1, p2) -> None:
        cal = self._ensure_calset()
        if cal is None:
            return
        name = self.cal_panel.target_plane
        length = simpledialog.askfloat(
            "Reference length", f"Marked distance: {distance(p1, p2):.2f} px\n\n"
            f"Actual length of the reference on plane '{name}' (mm):",
            parent=self.root, minvalue=1e-6)
        if length is None:
            self.set_status("Scale calibration cancelled.")
            return
        try:
            plane = make_scale_plane(name, p1, p2, length, self._lens_id(cal),
                                     self.cal_panel.setup_info())
        except ValueError as exc:
            messagebox.showerror("Scale calibration", str(exc), parent=self.root)
            return
        sens = click_sensitivity(p1, p2)
        if not messagebox.askyesno(
                "Scale calibration",
                f"Plane '{name}': {plane.px_per_mm:.5f} px/mm "
                f"({1000 / plane.px_per_mm:.2f} um per pixel).\n\n"
                f"A 1-pixel marking error changes the scale by {sens:.2%}.\n"
                "The reference must lie at the same height as the boundaries measured on this "
                "plane, with the camera square to it.\n\nUse this calibration?",
                parent=self.root) or not self._confirm_overwrite_plane(cal, name):
            self.set_status("Scale calibration discarded.")
            return
        cal.planes[name] = plane
        self._calibration_changed()
        self.set_status(f"Plane '{name}' calibrated: {plane.px_per_mm:.5f} px/mm.")

    def start_perspective_points(self) -> None:
        if self._ensure_calset() is not None:
            self.begin_mode("persp")

    def finish_perspective_points(self) -> None:
        if self.mode != "persp" or len(self.mode_points) < 4:
            return
        pts = list(self.mode_points)
        self.end_mode()
        world = PerspectivePointsDialog(self.root, pts).show()
        if world is None:
            self.set_status("Perspective calibration cancelled.")
            return
        self._apply_perspective(pts, world, "marked points")

    def _apply_perspective(self, image_pts, world_pts, source: str) -> None:
        cal = self._ensure_calset()
        if cal is None:
            return
        name = self.cal_panel.target_plane
        try:
            plane = make_perspective_plane(name, image_pts, world_pts, self.latest.resolution,
                                           self._lens_id(cal), self.cal_panel.setup_info())
        except ValueError as exc:
            messagebox.showerror("Perspective calibration", str(exc), parent=self.root)
            return
        n = len(image_pts)
        fit_note = ("exactly determined by 4 points - no redundancy to check the marking; "
                    "use more points if possible" if n == 4 else
                    f"reference fit RMS {plane.reprojection_rms_mm:.3f} mm over {n} points")
        if not messagebox.askyesno(
                "Perspective calibration",
                f"Plane '{name}' from {source}: rectified scale {plane.px_per_mm:.4f} px/mm, "
                f"{fit_note}.\n\nMeasurements on this plane are made on the rectified "
                "(perspective-corrected) image. Use this calibration?",
                parent=self.root) or not self._confirm_overwrite_plane(cal, name):
            self.set_status("Perspective calibration discarded.")
            return
        cal.planes[name] = plane
        self._calibration_changed()
        self.set_status(f"Plane '{name}' perspective calibration applied.")

    def perspective_from_checkerboard(self) -> None:
        if self._ensure_calset() is None:
            return
        form = FormDialog(self.root, "Perspective from checkerboard", [
            ("cols", "Inner corners (columns)", 9, int),
            ("rows", "Inner corners (rows)", 6, int),
            ("square", "Square size (mm)", 25.0, float)],
            message="Lay a flat checkerboard ON the measured plane (same height as the "
                    "boundaries to measure). Its corners define the plane coordinates.").show()
        if not form:
            return
        view = self.latest.view.copy()
        pattern = (form["cols"], form["rows"])
        self.set_status("Detecting checkerboard...")

        def done(corners, error):
            if error is not None or corners is None:
                messagebox.showerror("Checkerboard", "Checkerboard not found. Check the "
                                     "corner counts, focus and lighting.", parent=self.root)
                self.set_status("Checkerboard not found.")
                return
            world = checkerboard_world_points(pattern, form["square"])
            self._apply_perspective(corners, world, f"checkerboard ({len(corners)} corners)")

        run_in_background(self.root, lambda: find_checkerboard(view, pattern), done)

    def open_lens_calibration(self) -> None:
        if self.latest is None:
            messagebox.showinfo("Lens calibration", "Start a camera first.", parent=self.root)
            return
        LensCalibrationDialog(self.root, lambda: self.latest.raw if self.latest else None,
                              self._apply_lens)

    def _apply_lens(self, lens: LensCalibration) -> None:
        cal = self._ensure_calset()
        if cal is None:
            return
        if tuple(lens.image_size) != tuple(cal.resolution):
            messagebox.showerror("Lens calibration", "The checkerboard images do not match the "
                                 "current resolution.", parent=self.root)
            return
        if cal.planes and not messagebox.askyesno(
                "Lens calibration", "Applying lens correction changes the image geometry. "
                "All existing plane calibrations become invalid and must be redone, and ring "
                "positions in the profile should be re-checked.\n\nApply?", parent=self.root):
            return
        cal.lens = lens
        self._calibration_changed()
        self.set_status(f"Lens correction applied (RMS {lens.rms_px:.3f} px).")

    def remove_lens(self) -> None:
        cal = self.calibration
        if cal is None or cal.lens is None:
            return
        if messagebox.askyesno("Lens calibration", "Remove lens correction? Plane calibrations "
                               "made with it become invalid.", parent=self.root):
            cal.lens = None
            self._calibration_changed()

    def start_revalidation(self) -> None:
        cal = self.calibration
        name = self.cal_panel.selected_plane() or self.cal_panel.target_plane
        if cal is None or name not in cal.planes:
            messagebox.showinfo("Revalidate", "Select a calibrated plane first.",
                                parent=self.root)
            return
        if self.latest is None:
            return
        ok, why = cal.matches_source(self.source_id, self.latest.resolution)
        if not ok:
            messagebox.showerror("Revalidate", f"This calibration cannot be revalidated for "
                                 f"the current source: {why}. Recalibrate instead.",
                                 parent=self.root)
            return
        self._revalidate_plane = name
        self.begin_mode("revalidate")

    def _finish_revalidation(self, p1, p2) -> None:
        cal = self.calibration
        name = getattr(self, "_revalidate_plane", None)
        if cal is None or name not in cal.planes:
            return
        plane = cal.planes[name]
        length = simpledialog.askfloat(
            "Revalidate", f"Known length of the marked reference on plane '{name}' (mm):",
            parent=self.root, minvalue=1e-6)
        if length is None:
            return
        tol = self.cal_panel.tolerance
        passed, measured, rel = revalidate_plane(plane, p1, p2, length, tol)
        self._calibration_changed(new_id=False)
        text = (f"Reference {length:.4f} mm, measured {measured:.4f} mm, error {rel:+.3%} "
                f"(tolerance {tol:.2%}).")
        if passed:
            messagebox.showinfo("Revalidation passed", text + f"\n\nPlane '{name}' is valid.",
                                parent=self.root)
        else:
            messagebox.showwarning("Revalidation failed", text + "\n\nThe setup has changed. "
                                   "Recalibrate this plane.", parent=self.root)
        self.set_status(("Revalidation passed: " if passed else "Revalidation FAILED: ") + text)

    def mark_setup_changed(self) -> None:
        cal = self.calibration
        if cal is None or not cal.planes:
            return
        if messagebox.askyesno("Setup changed", "Camera position, zoom, focus or part height "
                               "changed? All planes will require revalidation before "
                               "millimetre values are shown.", parent=self.root):
            cal.mark_setup_changed()
            self._calibration_changed(new_id=False)

    def delete_plane(self, name: str) -> None:
        cal = self.calibration
        if cal and name in cal.planes and messagebox.askyesno(
                "Delete plane", f"Delete calibration of plane '{name}'?", parent=self.root):
            del cal.planes[name]
            self._calibration_changed()

    def new_calibration(self) -> None:
        if self.calibration is not None and not messagebox.askyesno(
                "New calibration", "Discard the current calibration? (Save it first if you "
                "need it.)", parent=self.root):
            return
        self.calibration = None
        if paths.LAST_CALIBRATION_FILE.exists():
            paths.LAST_CALIBRATION_FILE.unlink()
        self.refresh_all()
        self.push_config()

    def load_calibration_dialog(self) -> None:
        path = filedialog.askopenfilename(parent=self.root, title="Load calibration",
                                          filetypes=JSON_TYPES,
                                          initialdir=str(paths.CALIBRATION_DIR))
        if not path:
            return
        try:
            self.calibration = load_calibration(path)
        except (OSError, ValueError, KeyError) as exc:
            messagebox.showerror("Load calibration", f"Could not load:\n{exc}",
                                 parent=self.root)
            return
        self._calibration_changed(new_id=False)
        self.set_status("Calibration loaded. Each plane must be revalidated with a known "
                        "reference before millimetres are shown.")

    def save_calibration_dialog(self) -> None:
        if self.calibration is None:
            messagebox.showinfo("Save calibration", "There is no calibration to save.",
                                parent=self.root)
            return
        path = filedialog.asksaveasfilename(
            parent=self.root, title="Save calibration", defaultextension=".json",
            filetypes=JSON_TYPES, initialdir=str(paths.CALIBRATION_DIR),
            initialfile=f"calibration_{self.calibration.calibration_id}.json")
        if path:
            save_calibration(path, self.calibration)
            self.set_status(f"Calibration saved to {path}")

    # =================================================================== results
    def save_result(self) -> None:
        frame = self.latest
        if frame is None:
            messagebox.showinfo("Save result", "No image to save.", parent=self.root)
            return
        if not frame.results:
            messagebox.showinfo("Save result", "The profile has no measurements yet.",
                                parent=self.root)
            return
        cfg = self.config_for(frame)
        cal = cfg.calibration if cfg else self.calibration
        try:
            rows, path = self.store.save_capture(frame, self.profile.name, cal)
        except OSError as exc:
            messagebox.showerror("Save result", str(exc), parent=self.root)
            return
        self.results_panel.set_saved_count(len(self.store.session_rows))
        summary = ", ".join(f"{r['measurement']}={r['diameter_mm'] or r['status']}" for r in rows)
        self.set_status(f"Saved {path.name}: {summary}")

    def export_csv(self) -> None:
        rows = self.store.session_rows
        if not rows:
            if self.latest is None or not self.latest.results:
                messagebox.showinfo("Export CSV", "Nothing to export yet.", parent=self.root)
                return
            if not messagebox.askyesno("Export CSV", "No results have been saved this "
                                       "session. Export the current live readings instead?",
                                       parent=self.root):
                return
            cfg = self.config_for(self.latest)
            cal = cfg.calibration if cfg else self.calibration
            from datetime import datetime
            ts = datetime.now().isoformat(timespec="milliseconds")
            rows = [result_row(r, self.latest, self.profile.name, cal, ts)
                    for r in self.latest.results]
        path = filedialog.asksaveasfilename(parent=self.root, title="Export CSV",
                                            defaultextension=".csv",
                                            filetypes=[("CSV", "*.csv")],
                                            initialdir=str(paths.RESULTS_DIR),
                                            initialfile="odms_results.csv")
        if path:
            try:
                n = self.store.export_csv(path, rows)
            except OSError as exc:
                messagebox.showerror("Export CSV", str(exc), parent=self.root)
                return
            self.set_status(f"Exported {n} row(s) to {path}")

    # ===================================================================== close
    def on_close(self) -> None:
        self._stop_camera_thread()
        for cam in self._old_cameras:
            cam.join(timeout=3.0)
        self.worker.stop()
        self.worker.join(timeout=3.0)
        if self.calibration is not None:
            try:
                save_calibration(paths.LAST_CALIBRATION_FILE, self.calibration)
            except OSError:
                pass
        self.root.destroy()
