"""Modal dialogs and a helper for running slow work off the UI thread."""
from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable

import numpy as np

from ..calibration.lens import RECOMMENDED_VIEWS, calibrate_lens, find_checkerboard
from ..calibration.models import LensCalibration
from ..storage.results import imread
from ..units import to_mm


def run_in_background(widget: tk.Misc, work: Callable[[], Any],
                      on_done: Callable[[Any, BaseException | None], None]) -> None:
    """Run ``work`` in a thread and call ``on_done`` on the Tk thread."""
    box: queue.Queue = queue.Queue(maxsize=1)

    def target():
        try:
            box.put((work(), None))
        except BaseException as exc:  # reported to the UI
            box.put((None, exc))

    threading.Thread(target=target, daemon=True).start()

    def poll():
        try:
            value, error = box.get_nowait()
        except queue.Empty:
            widget.after(50, poll)
            return
        on_done(value, error)

    widget.after(50, poll)


class _Modal(tk.Toplevel):
    def __init__(self, parent: tk.Misc, title: str) -> None:
        super().__init__(parent)
        self.title(title)
        self.transient(parent.winfo_toplevel())
        self.resizable(False, False)
        self.result = None
        self.body = ttk.Frame(self, padding=12)
        self.body.pack(fill="both", expand=True)
        self.protocol("WM_DELETE_WINDOW", self.cancel)
        self.bind("<Escape>", lambda e: self.cancel())

    def buttons(self, ok_text: str = "OK") -> None:
        bar = ttk.Frame(self, padding=(12, 0, 12, 12))
        bar.pack(fill="x")
        ttk.Button(bar, text="Cancel", command=self.cancel).pack(side="right")
        ttk.Button(bar, text=ok_text, command=self.ok).pack(side="right", padx=6)
        self.bind("<Return>", lambda e: self.ok())

    def show(self):
        self.grab_set()
        self.wait_visibility()
        self.focus_set()
        self.wait_window()
        return self.result

    def ok(self) -> None:
        self.destroy()

    def cancel(self) -> None:
        self.result = None
        self.destroy()


class FormDialog(_Modal):
    """Simple form: ``fields`` is a list of (key, label, initial, type)."""

    def __init__(self, parent, title: str, fields: list[tuple[str, str, Any, type]],
                 message: str = "") -> None:
        super().__init__(parent, title)
        self._fields = fields
        self._vars: dict[str, tk.StringVar] = {}
        row = 0
        if message:
            ttk.Label(self.body, text=message, wraplength=380, justify="left").grid(
                row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
            row = 1
        for key, label, initial, _ in fields:
            ttk.Label(self.body, text=label).grid(row=row, column=0, sticky="w", pady=2)
            var = tk.StringVar(value="" if initial is None else str(initial))
            entry = ttk.Entry(self.body, textvariable=var, width=16)
            entry.grid(row=row, column=1, sticky="ew", padx=(8, 0), pady=2)
            if row == (1 if message else 0):
                entry.focus_set()
            self._vars[key] = var
            row += 1
        self.buttons()

    def ok(self) -> None:
        out = {}
        for key, label, _, typ in self._fields:
            text = self._vars[key].get().strip()
            try:
                out[key] = typ(text)
            except ValueError:
                messagebox.showerror("Invalid value", f"{label}: '{text}' is not valid",
                                     parent=self)
                return
        self.result = out
        self.destroy()


class PerspectivePointsDialog(_Modal):
    """Enter the real-world plane coordinates (mm) of each marked point."""

    def __init__(self, parent, image_points: list[tuple[float, float]],
                 initial: list[tuple[float, float]] | None = None, unit: str = "mm") -> None:
        super().__init__(parent, "Perspective reference coordinates")
        ttk.Label(self.body, wraplength=460, justify="left", text=(
            f"Enter the known position of every marked point on the measured plane, in {unit}, "
            "in any planar coordinate system (e.g. corner 1 = 0,0). Points "
            "must lie on the same plane as the measured boundary and must not be collinear."
        )).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 8))
        for c, text in enumerate(("Point", "Image (px)", f"X ({unit})", f"Y ({unit})")):
            ttk.Label(self.body, text=text, font=("Segoe UI", 9, "bold")).grid(
                row=1, column=c, sticky="w", padx=4)
        self._vars = []
        for i, (x, y) in enumerate(image_points):
            ttk.Label(self.body, text=str(i + 1)).grid(row=i + 2, column=0, padx=4)
            ttk.Label(self.body, text=f"{x:.1f}, {y:.1f}").grid(row=i + 2, column=1, padx=4)
            init = initial[i] if initial and i < len(initial) else ("", "")
            vx, vy = tk.StringVar(value=str(init[0])), tk.StringVar(value=str(init[1]))
            ttk.Entry(self.body, textvariable=vx, width=10).grid(row=i + 2, column=2, padx=4,
                                                                  pady=1)
            ttk.Entry(self.body, textvariable=vy, width=10).grid(row=i + 2, column=3, padx=4,
                                                                  pady=1)
            self._vars.append((vx, vy))
        self.buttons("Compute")

    def ok(self) -> None:
        try:
            self.result = [(float(vx.get()), float(vy.get())) for vx, vy in self._vars]
        except ValueError:
            messagebox.showerror("Invalid value", "Every point needs numeric X and Y values.",
                                 parent=self)
            return
        self.destroy()


class LensCalibrationDialog(tk.Toplevel):
    """Collect checkerboard views and compute a lens (distortion) calibration."""

    def __init__(self, parent, get_raw_frame: Callable[[], np.ndarray | None],
                 on_apply: Callable[[LensCalibration], None], unit: str = "mm") -> None:
        super().__init__(parent)
        self.title("Lens calibration (checkerboard)")
        self.transient(parent.winfo_toplevel())
        self.unit = unit
        self._get_frame = get_raw_frame
        self._on_apply = on_apply
        self._views: list[np.ndarray] = []
        self._size: tuple[int, int] | None = None
        self._result: LensCalibration | None = None
        self._busy = False

        f = ttk.Frame(self, padding=12)
        f.pack(fill="both", expand=True)
        ttk.Label(f, wraplength=440, justify="left", text=(
            "Print a flat checkerboard and count its INNER corners. Capture "
            f"{RECOMMENDED_VIEWS}+ views covering the whole image, including the corners, "
            "with the board tilted in different directions. Keep focus and zoom fixed - "
            "changing them invalidates the calibration.")).grid(row=0, column=0, columnspan=4,
                                                               sticky="w")
        self.cols = tk.StringVar(value="9")
        self.rows = tk.StringVar(value="6")
        self.square = tk.StringVar(value="25.0")
        for i, (label, var) in enumerate((("Inner corners (columns)", self.cols),
                                          ("Inner corners (rows)", self.rows),
                                          (f"Square size ({unit})", self.square))):
            ttk.Label(f, text=label).grid(row=1 + i, column=0, sticky="w", pady=2)
            ttk.Entry(f, textvariable=var, width=10).grid(row=1 + i, column=1, sticky="w")
        bar = ttk.Frame(f)
        bar.grid(row=4, column=0, columnspan=4, sticky="w", pady=8)
        ttk.Button(bar, text="Capture current frame", command=self.capture).pack(side="left")
        ttk.Button(bar, text="Add image files...", command=self.add_files).pack(side="left",
                                                                                 padx=6)
        ttk.Button(bar, text="Clear", command=self.clear).pack(side="left")
        self.status = tk.StringVar(value="Views: 0")
        ttk.Label(f, textvariable=self.status, wraplength=440).grid(row=5, column=0,
                                                                    columnspan=4, sticky="w")
        bar2 = ttk.Frame(f)
        bar2.grid(row=6, column=0, columnspan=4, sticky="e", pady=(10, 0))
        ttk.Button(bar2, text="Close", command=self.destroy).pack(side="right")
        self.apply_btn = ttk.Button(bar2, text="Apply", command=self.apply, state="disabled")
        self.apply_btn.pack(side="right", padx=6)
        ttk.Button(bar2, text="Compute", command=self.compute).pack(side="right")

    def _pattern(self) -> tuple[tuple[int, int], float] | None:
        try:
            pattern = (int(self.cols.get()), int(self.rows.get()))
            square = to_mm(float(self.square.get()), self.unit)
            if min(pattern) < 2 or square <= 0:
                raise ValueError
            return pattern, square
        except ValueError:
            messagebox.showerror("Checkerboard", "Enter integer corner counts (>= 2) and a "
                                 "positive square size.", parent=self)
            return None

    def _add_image(self, image: np.ndarray, label: str) -> None:
        cfg = self._pattern()
        if cfg is None or self._busy:
            return
        size = (image.shape[1], image.shape[0])
        if self._size is not None and size != self._size:
            messagebox.showerror("Checkerboard", f"{label}: resolution {size} differs from "
                                 f"earlier views {self._size}.", parent=self)
            return
        self._busy = True
        self.status.set(f"Searching for checkerboard in {label}...")

        def done(corners, error):
            self._busy = False
            if error is not None or corners is None:
                self.status.set(f"Views: {len(self._views)}  -  checkerboard NOT found in "
                                f"{label}. Check corner counts, focus and lighting.")
                return
            self._size = size
            self._views.append(corners)
            self.status.set(f"Views: {len(self._views)}  -  added {label}")

        run_in_background(self, lambda: find_checkerboard(image, cfg[0]), done)

    def capture(self) -> None:
        frame = self._get_frame()
        if frame is None:
            messagebox.showinfo("Checkerboard", "No camera frame available.", parent=self)
            return
        self._add_image(frame.copy(), "current frame")

    def add_files(self) -> None:
        paths = filedialog.askopenfilenames(parent=self, title="Checkerboard images",
                                            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp "
                                                        "*.tif *.tiff"), ("All", "*.*")])
        for p in paths:
            img = imread(p)
            if img is not None:
                self._add_image(img, p.split("/")[-1])
                self.wait_variable_idle()

    def wait_variable_idle(self) -> None:
        while self._busy:
            self.update()
            self.after(30)

    def clear(self) -> None:
        self._views.clear()
        self._size = None
        self._result = None
        self.apply_btn.configure(state="disabled")
        self.status.set("Views: 0")

    def compute(self) -> None:
        cfg = self._pattern()
        if cfg is None or self._busy:
            return
        if self._size is None:
            messagebox.showinfo("Lens calibration", "Add checkerboard views first.", parent=self)
            return
        self._busy = True
        self.status.set("Calibrating...")
        views, size = list(self._views), self._size

        def done(lens, error):
            self._busy = False
            if error is not None:
                self.status.set(f"Calibration failed: {error}")
                return
            self._result = lens
            quality = "good" if lens.rms_px < 0.5 else ("acceptable" if lens.rms_px < 1.0
                                                        else "poor - recapture views")
            self.status.set(f"Reprojection RMS {lens.rms_px:.3f} px from {lens.n_views} views "
                            f"({quality}). Press Apply to use it.")
            self.apply_btn.configure(state="normal")

        run_in_background(self, lambda: calibrate_lens(views, cfg[0], cfg[1], size), done)

    def apply(self) -> None:
        if self._result is not None:
            self._on_apply(self._result)
            self.destroy()
