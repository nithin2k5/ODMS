"""Calibration tab: status of the calibration set and the guided workflows."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

from ..calibration.models import CalibrationSet, CalState, SetupInfo

if TYPE_CHECKING:  # pragma: no cover
    from .main_window import MainWindow

SETUP_FIELDS = (
    ("camera_height_mm", "Camera height / distance (mm)"),
    ("zoom", "Zoom setting"),
    ("focus", "Focus setting"),
    ("part_height_mm", "Measured surface height (mm)"),
    ("lighting", "Lighting"),
    ("notes", "Notes"),
)

STATE_TEXT = {
    CalState.VALID: "Valid",
    CalState.REVALIDATE: "Revalidation required",
    CalState.INVALID: "Invalid",
    CalState.MISSING: "Missing",
    CalState.REFERENCE_FAILED: "Reference check failed",
}


class CalibrationPanel(ttk.Frame):
    def __init__(self, master, app: "MainWindow") -> None:
        super().__init__(master, padding=8)
        self.app = app
        self.columnconfigure(0, weight=1)

        info = ttk.LabelFrame(self, text="Current calibration", padding=6)
        info.grid(row=0, column=0, sticky="ew")
        self.info_var = tk.StringVar(value="No calibration")
        ttk.Label(info, textvariable=self.info_var, justify="left", wraplength=330).pack(
            anchor="w")

        planes = ttk.LabelFrame(self, text="Planes (one per measured surface height)",
                                padding=6)
        planes.grid(row=1, column=0, sticky="nsew", pady=(6, 0))
        self.rowconfigure(1, weight=1)
        self.tree = ttk.Treeview(planes, columns=("mode", "scale", "state"), height=4)
        self.tree.heading("#0", text="Plane")
        self.tree.heading("mode", text="Mode")
        self.tree.heading("scale", text="px/mm")
        self.tree.heading("state", text="State")
        self.tree.column("#0", width=100)
        self.tree.column("mode", width=80)
        self.tree.column("scale", width=70, anchor="e")
        self.tree.column("state", width=150)
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        target = ttk.Frame(planes)
        target.pack(fill="x", pady=(4, 0))
        ttk.Label(target, text="Plane to calibrate:").pack(side="left")
        self.plane_var = tk.StringVar(value="default")
        self.plane_combo = ttk.Combobox(target, textvariable=self.plane_var, width=18)
        self.plane_combo.pack(side="left", padx=4)

        setup = ttk.LabelFrame(self, text="Setup record (saved with the calibration)",
                               padding=6)
        setup.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        setup.columnconfigure(1, weight=1)
        self.setup_vars: dict[str, tk.StringVar] = {}
        for i, (key, label) in enumerate(SETUP_FIELDS):
            ttk.Label(setup, text=label).grid(row=i, column=0, sticky="w")
            var = tk.StringVar()
            ttk.Entry(setup, textvariable=var).grid(row=i, column=1, sticky="ew", padx=(6, 0),
                                                    pady=1)
            self.setup_vars[key] = var

        steps = ttk.LabelFrame(self, text="Calibrate", padding=6)
        steps.grid(row=3, column=0, sticky="ew", pady=(6, 0))
        steps.columnconfigure((0, 1), weight=1)
        b = dict(sticky="ew", padx=2, pady=2)
        ttk.Button(steps, text="Lens: checkerboard...",
                   command=app.open_lens_calibration).grid(row=0, column=0, **b)
        ttk.Button(steps, text="Remove lens correction",
                   command=app.remove_lens).grid(row=0, column=1, **b)
        ttk.Button(steps, text="Scale: mark reference",
                   command=app.start_scale_calibration).grid(row=1, column=0, **b)
        ttk.Button(steps, text="Perspective: mark points",
                   command=app.start_perspective_points).grid(row=1, column=1, **b)
        self.finish_btn = ttk.Button(steps, text="Finish points (0)", state="disabled",
                                     command=app.finish_perspective_points)
        self.finish_btn.grid(row=2, column=1, **b)
        ttk.Button(steps, text="Perspective: checkerboard...",
                   command=app.perspective_from_checkerboard).grid(row=2, column=0, **b)

        ref = ttk.LabelFrame(self, text="Live scale reference (measured every frame)",
                             padding=6)
        ref.grid(row=4, column=0, sticky="ew", pady=(6, 0))
        ref.columnconfigure((0, 1), weight=1)
        ttk.Button(ref, text="Define reference circle",
                   command=app.start_reference).grid(row=0, column=0, **b)
        ttk.Button(ref, text="Remove reference",
                   command=app.remove_reference).grid(row=0, column=1, **b)
        self.ref_var = tk.StringVar(value="No scale reference.")
        ttk.Label(ref, textvariable=self.ref_var, wraplength=330, justify="left").grid(
            row=1, column=0, columnspan=2, sticky="w")

        check = ttk.LabelFrame(self, text="Validity", padding=6)
        check.grid(row=5, column=0, sticky="ew", pady=(6, 0))
        check.columnconfigure((0, 1), weight=1)
        ttk.Button(check, text="Revalidate plane...",
                   command=app.start_revalidation).grid(row=0, column=0, **b)
        ttk.Button(check, text="Setup changed",
                   command=app.mark_setup_changed).grid(row=0, column=1, **b)
        tol = ttk.Frame(check)
        tol.grid(row=1, column=0, columnspan=2, sticky="w")
        ttk.Label(tol, text="Revalidation tolerance (%)").pack(side="left")
        self.tol_var = tk.StringVar(value="0.5")
        ttk.Spinbox(tol, from_=0.05, to=10, increment=0.05, width=6,
                    textvariable=self.tol_var).pack(side="left", padx=4)
        ttk.Button(check, text="Delete plane", command=self._delete).grid(row=2, column=0, **b)

        files = ttk.Frame(self)
        files.grid(row=6, column=0, sticky="ew", pady=(6, 0))
        files.columnconfigure((0, 1, 2), weight=1)
        ttk.Button(files, text="New", command=app.new_calibration).grid(row=0, column=0, **b)
        ttk.Button(files, text="Load...", command=app.load_calibration_dialog).grid(
            row=0, column=1, **b)
        ttk.Button(files, text="Save...", command=app.save_calibration_dialog).grid(
            row=0, column=2, **b)

        self.hint_var = tk.StringVar(value=(
            "Mark references on the same plane, at the same height, as the boundary to be "
            "measured. Different surface heights need separate planes."))
        ttk.Label(self, textvariable=self.hint_var, wraplength=340, justify="left",
                  foreground="#555").grid(row=7, column=0, sticky="ew", pady=(8, 0))

    # -- values for the controller ------------------------------------------------
    @property
    def target_plane(self) -> str:
        return self.plane_var.get().strip() or "default"

    @property
    def tolerance(self) -> float:
        try:
            return max(float(self.tol_var.get()), 0.01) / 100.0
        except ValueError:
            return 0.005

    def setup_info(self) -> SetupInfo:
        return SetupInfo(**{k: v.get().strip() for k, v in self.setup_vars.items()})

    def set_setup_info(self, setup: SetupInfo) -> None:
        for key, var in self.setup_vars.items():
            var.set(getattr(setup, key, ""))

    def set_point_count(self, n: int | None) -> None:
        if n is None:
            self.finish_btn.configure(text="Finish points (0)", state="disabled")
        else:
            self.finish_btn.configure(text=f"Finish points ({n})",
                                      state="normal" if n >= 4 else "disabled")

    # -- refresh -----------------------------------------------------------------
    def refresh(self, cal: CalibrationSet | None, camera_id: str,
                resolution: tuple[int, int] | None, lens_applied: bool,
                references: list | None = None) -> None:
        ref_by_plane = {r.plane: r for r in references or []}
        if not ref_by_plane:
            has_ref = cal is not None and any(p.reference for p in cal.planes.values())
            self.ref_var.set("Reference defined; waiting for a usable frame." if has_ref
                             else "No scale reference.")
        else:
            self.ref_var.set("\n".join(r.summary for r in ref_by_plane.values()))
        if cal is None:
            self.info_var.set("No calibration. Measurements are shown in pixels only.\n"
                              f"Source: {camera_id}"
                              + (f" at {resolution[0]}x{resolution[1]}" if resolution else ""))
        else:
            ok, why = (cal.matches_source(camera_id, resolution) if resolution
                       else (False, "no image"))
            lens = "none"
            if cal.lens is not None:
                lens = (f"RMS {cal.lens.rms_px:.3f} px, {cal.lens.n_views} views, "
                        f"{'applied' if lens_applied else 'NOT applied'}")
            lines = [f"ID {cal.calibration_id}   updated {cal.updated}",
                     f"Camera {cal.camera_id} at {cal.resolution[0]}x{cal.resolution[1]}",
                     f"Lens correction: {lens}"]
            if not ok:
                lines.append(f"NOT USABLE for current source: {why}")
            self.info_var.set("\n".join(lines))

        names = sorted(cal.planes) if cal else []
        values = sorted(set(names) | {"default"} | set(self.app.profile_planes()))
        self.plane_combo.configure(values=values)

        selected = self.selected_plane()
        self.tree.delete(*self.tree.get_children())
        for name in names:
            plane = cal.planes[name]
            state, why = (cal.plane_state(name, camera_id, resolution) if resolution
                          else (CalState.INVALID, "no image"))
            mode = "straight-on" if plane.mode == "scale" else "perspective"
            scale = f"{plane.px_per_mm:.4f}"
            text = STATE_TEXT[state]
            rr = ref_by_plane.get(name)
            if rr is not None:
                text = (f"Valid (reference {rr.mode})" if rr.ok
                        else STATE_TEXT[CalState.REFERENCE_FAILED])
                if rr.ok and rr.mode == "live" and rr.px_per_mm:
                    scale = f"{rr.px_per_mm:.4f}*"
            self.tree.insert("", "end", iid=name, text=name, values=(mode, scale, text))
        if selected in names:
            self.tree.selection_set(selected)

    def selected_plane(self) -> str | None:
        sel = self.tree.selection()
        return sel[0] if sel else None

    def _on_select(self, _event=None) -> None:
        name = self.selected_plane()
        if name:
            self.plane_var.set(name)
            cal = self.app.calibration
            if cal and name in cal.planes:
                self.set_setup_info(cal.planes[name].setup)

    def _delete(self) -> None:
        name = self.selected_plane()
        if name:
            self.app.delete_plane(name)
