"""Measurements tab (profile, list, ring set-up) and Processing tab."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

from ..measurement.models import POLARITIES, POLARITY_HELP, MeasurementSpec, ProcessingParams
from ..storage.profiles import PartProfile

if TYPE_CHECKING:  # pragma: no cover
    from .main_window import MainWindow


def _float_or_none(text: str) -> float | None:
    text = text.strip()
    if not text:
        return None
    return float(text)


class MeasurementPanel(ttk.Frame):
    def __init__(self, master, app: "MainWindow") -> None:
        super().__init__(master, padding=8)
        self.app = app
        self._loading = False
        self.columnconfigure(0, weight=1)
        b = dict(sticky="ew", padx=2, pady=2)

        prof = ttk.LabelFrame(self, text="Part profile", padding=6)
        prof.grid(row=0, column=0, sticky="ew")
        prof.columnconfigure(1, weight=1)
        ttk.Label(prof, text="Name").grid(row=0, column=0, sticky="w")
        self.profile_name = tk.StringVar()
        ttk.Entry(prof, textvariable=self.profile_name).grid(row=0, column=1, columnspan=3,
                                                             sticky="ew", padx=(6, 0))
        self.profile_name.trace_add("write", lambda *_: self._profile_renamed())
        bar = ttk.Frame(prof)
        bar.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(4, 0))
        bar.columnconfigure((0, 1, 2, 3), weight=1)
        ttk.Button(bar, text="New", command=app.new_profile).grid(row=0, column=0, **b)
        ttk.Button(bar, text="Open...", command=app.open_profile_dialog).grid(row=0, column=1, **b)
        ttk.Button(bar, text="Save", command=app.save_profile).grid(row=0, column=2, **b)
        ttk.Button(bar, text="Save as...",
                   command=lambda: app.save_profile(as_new=True)).grid(row=0, column=3, **b)

        lst = ttk.LabelFrame(self, text="Measurements", padding=6)
        lst.grid(row=1, column=0, sticky="nsew", pady=(6, 0))
        self.rowconfigure(1, weight=1)
        self.tree = ttk.Treeview(lst, columns=("plane", "limits"), height=5)
        self.tree.heading("#0", text="Name")
        self.tree.heading("plane", text="Plane")
        self.tree.heading("limits", text="Limits (mm)")
        self.tree.column("#0", width=200)
        self.tree.column("plane", width=80)
        self.tree.column("limits", width=110)
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<Double-1>", lambda e: self._edit())
        bar = ttk.Frame(lst)
        bar.pack(fill="x", pady=(4, 0))
        for i in range(5):
            bar.columnconfigure(i, weight=1)
        ttk.Button(bar, text="Add", command=app.new_measurement).grid(row=0, column=0, **b)
        ttk.Button(bar, text="Edit", command=self._edit).grid(row=0, column=1, **b)
        ttk.Button(bar, text="Delete", command=self._delete).grid(row=0, column=2, **b)
        ttk.Button(bar, text="Up", command=lambda: self._move(-1)).grid(row=0, column=3, **b)
        ttk.Button(bar, text="Down", command=lambda: self._move(1)).grid(row=0, column=4, **b)

        ed = ttk.LabelFrame(self, text="Set up measurement", padding=6)
        ed.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        ed.columnconfigure(1, weight=1, minsize=70)
        ed.columnconfigure(3, weight=1, minsize=70)
        self.editor = ed
        self.name_var = tk.StringVar()
        self.plane_var = tk.StringVar(value="default")
        self.r_in_var = tk.StringVar()
        self.r_out_var = tk.StringVar()
        self.half_var = tk.StringVar(value="12")
        self.pol_var = tk.StringVar(value="auto")
        self.min_var = tk.StringVar()
        self.max_var = tk.StringVar()

        ttk.Label(ed, text="Name").grid(row=0, column=0, sticky="w")
        ttk.Entry(ed, textvariable=self.name_var).grid(row=0, column=1, columnspan=3,
                                                       sticky="ew", padx=(6, 0))
        ttk.Label(ed, text="Plane").grid(row=1, column=0, sticky="w")
        self.plane_combo = ttk.Combobox(ed, textvariable=self.plane_var, width=14)
        self.plane_combo.grid(row=1, column=1, columnspan=3, sticky="ew", padx=(6, 0), pady=2)

        steps = ttk.Frame(ed)
        steps.grid(row=2, column=0, columnspan=4, sticky="ew", pady=4)
        steps.columnconfigure((0, 1, 2), weight=1)
        ttk.Button(steps, text="1. Draw ROI",
                   command=lambda: app.begin_mode("roi")).grid(row=0, column=0, **b)
        ttk.Button(steps, text="2. Set centre",
                   command=lambda: app.begin_mode("center")).grid(row=0, column=1, **b)
        ttk.Button(steps, text="3. Pick edge",
                   command=lambda: app.begin_mode("edge")).grid(row=0, column=2, **b)

        ttk.Label(ed, text="Inner radius (px)").grid(row=3, column=0, sticky="w")
        ttk.Spinbox(ed, from_=0, to=10000, increment=1, width=8,
                    textvariable=self.r_in_var).grid(row=3, column=1, sticky="ew", padx=(6, 0))
        ttk.Label(ed, text="Outer radius (px)").grid(row=3, column=2, sticky="w", padx=(6, 0))
        ttk.Spinbox(ed, from_=0, to=10000, increment=1, width=8,
                    textvariable=self.r_out_var).grid(row=3, column=3, sticky="ew", padx=(6, 0))
        ttk.Label(ed, text="Band +/- on pick (px)").grid(row=4, column=0, sticky="w")
        ttk.Spinbox(ed, from_=2, to=500, increment=1, width=8,
                    textvariable=self.half_var).grid(row=4, column=1, sticky="ew", padx=(6, 0))
        ttk.Label(ed, text="Edge polarity").grid(row=4, column=2, sticky="w", padx=(6, 0))
        pol = ttk.Combobox(ed, textvariable=self.pol_var, values=POLARITIES, width=8,
                           state="readonly")
        pol.grid(row=4, column=3, sticky="ew", padx=(6, 0))
        self.pol_help = tk.StringVar(value=POLARITY_HELP["auto"])
        ttk.Label(ed, textvariable=self.pol_help, foreground="#555", wraplength=330).grid(
            row=5, column=0, columnspan=4, sticky="w")
        ttk.Label(ed, text="Min (mm)").grid(row=6, column=0, sticky="w")
        ttk.Entry(ed, textvariable=self.min_var, width=10).grid(row=6, column=1, sticky="ew",
                                                                padx=(6, 0))
        ttk.Label(ed, text="Max (mm)").grid(row=6, column=2, sticky="w", padx=(6, 0))
        ttk.Entry(ed, textvariable=self.max_var, width=10).grid(row=6, column=3, sticky="ew",
                                                                padx=(6, 0))

        act = ttk.Frame(ed)
        act.grid(row=7, column=0, columnspan=4, sticky="ew", pady=(6, 0))
        act.columnconfigure((0, 1, 2), weight=1)
        ttk.Button(act, text="Re-centre on fit", command=app.recentre_draft).grid(row=0,
                                                                                 column=0, **b)
        ttk.Button(act, text="Confirm & save", command=app.confirm_draft).grid(row=0, column=1,
                                                                               **b)
        ttk.Button(act, text="Cancel", command=app.cancel_draft).grid(row=0, column=2, **b)
        self.draft_status = tk.StringVar(value="Press Add to set up a new measurement.")
        ttk.Label(ed, textvariable=self.draft_status, wraplength=340, justify="left").grid(
            row=8, column=0, columnspan=4, sticky="w", pady=(6, 0))

        for var in (self.name_var, self.plane_var, self.r_in_var, self.r_out_var, self.pol_var,
                    self.min_var, self.max_var):
            var.trace_add("write", lambda *_: self._changed())
        self.set_editor_enabled(False)

    # -- editor state ------------------------------------------------------------
    def set_editor_enabled(self, enabled: bool) -> None:
        def walk(widget):
            for child in widget.winfo_children():
                if isinstance(child, (ttk.Entry, ttk.Button, ttk.Spinbox, ttk.Combobox)):
                    if isinstance(child, ttk.Combobox) and child.cget("state") == "readonly" \
                            and enabled:
                        continue
                    try:
                        child.state(["!disabled"] if enabled else ["disabled"])
                    except tk.TclError:
                        pass
                walk(child)
        walk(self.editor)

    @property
    def band_half_width(self) -> float:
        try:
            return max(float(self.half_var.get()), 1.0)
        except ValueError:
            return 12.0

    def load_draft(self, spec: MeasurementSpec | None) -> None:
        self._loading = True
        try:
            if spec is None:
                for var in (self.name_var, self.r_in_var, self.r_out_var, self.min_var,
                            self.max_var):
                    var.set("")
                self.pol_var.set("auto")
                self.set_editor_enabled(False)
                self.draft_status.set("Press Add to set up a new measurement, or select one "
                                      "and press Edit.")
            else:
                self.set_editor_enabled(True)
                self.name_var.set(spec.name)
                self.plane_var.set(spec.plane)
                self.r_in_var.set(f"{spec.r_inner:.1f}" if spec.r_outer else "")
                self.r_out_var.set(f"{spec.r_outer:.1f}" if spec.r_outer else "")
                self.pol_var.set(spec.params.polarity)
                self.min_var.set("" if spec.min_mm is None else f"{spec.min_mm:g}")
                self.max_var.set("" if spec.max_mm is None else f"{spec.max_mm:g}")
            self.pol_help.set(POLARITY_HELP.get(self.pol_var.get(), ""))
        finally:
            self._loading = False

    def read_into(self, spec: MeasurementSpec) -> list[str]:
        """Copy editor values into ``spec``; returns a list of problems."""
        problems = []
        spec.name = self.name_var.get().strip()
        spec.plane = self.plane_var.get().strip() or "default"
        for var, attr in ((self.r_in_var, "r_inner"), (self.r_out_var, "r_outer")):
            text = var.get().strip()
            if text:
                try:
                    setattr(spec, attr, max(float(text), 0.0))
                except ValueError:
                    problems.append(f"{attr} is not a number")
        spec.params.polarity = self.pol_var.get()
        for var, attr in ((self.min_var, "min_mm"), (self.max_var, "max_mm")):
            try:
                setattr(spec, attr, _float_or_none(var.get()))
            except ValueError:
                problems.append(f"{attr} is not a number")
        if spec.min_mm is not None and spec.max_mm is not None and spec.min_mm > spec.max_mm:
            problems.append("min is greater than max")
        self.pol_help.set(POLARITY_HELP.get(spec.params.polarity, ""))
        return problems

    def set_draft_status(self, text: str) -> None:
        self.draft_status.set(text)

    def _changed(self) -> None:
        if not self._loading:
            self.app.draft_fields_changed()

    def _profile_renamed(self) -> None:
        if not self._loading:
            self.app.profile.name = self.profile_name.get().strip() or "Untitled part"

    # -- list --------------------------------------------------------------------
    def refresh(self, profile: PartProfile, planes: list[str]) -> None:
        self._loading = True
        try:
            if self.profile_name.get() != profile.name:
                self.profile_name.set(profile.name)
        finally:
            self._loading = False
        sel = self.tree.selection()
        self.tree.delete(*self.tree.get_children())
        for s in profile.specs:
            lo = "-" if s.min_mm is None else f"{s.min_mm:g}"
            hi = "-" if s.max_mm is None else f"{s.max_mm:g}"
            limits = "" if s.min_mm is None and s.max_mm is None else f"{lo} .. {hi}"
            self.tree.insert("", "end", iid=s.uid, text=s.name, values=(s.plane, limits))
        for iid in sel:
            if self.tree.exists(iid):
                self.tree.selection_add(iid)
        self.plane_combo.configure(values=planes)

    def _selected(self) -> str | None:
        sel = self.tree.selection()
        return sel[0] if sel else None

    def _edit(self) -> None:
        uid = self._selected()
        if uid:
            self.app.edit_measurement(uid)

    def _delete(self) -> None:
        uid = self._selected()
        if uid:
            self.app.delete_measurement(uid)

    def _move(self, delta: int) -> None:
        uid = self._selected()
        if uid:
            self.app.move_measurement(uid, delta)


# (attribute, label, kind, from, to, increment)
PARAM_FIELDS = [
    ("clahe", "Contrast enhancement (CLAHE)", "bool", None, None, None),
    ("clahe_clip", "CLAHE clip limit", "float", 0.5, 10.0, 0.5),
    ("blur_sigma", "Denoise blur sigma (px)", "float", 0.0, 5.0, 0.1),
    ("min_edge_strength", "Min edge strength (grey/px)", "float", 1.0, 100.0, 1.0),
    ("n_rays", "Rays around the circle", "int", 36, 1440, 36),
    ("candidates_per_ray", "Edge candidates per ray", "int", 1, 4, 1),
    ("ransac_threshold_px", "RANSAC inlier threshold (px)", "float", 0.3, 5.0, 0.1),
    ("min_coverage", "Min angular coverage (0-1)", "float", 0.1, 1.0, 0.05),
    ("max_gap_deg", "Max angular gap (deg)", "float", 10.0, 360.0, 5.0),
    ("min_inlier_ratio", "Min inlier ratio (0-1)", "float", 0.1, 1.0, 0.05),
    ("max_rms_px", "Max fit RMS (px)", "float", 0.1, 5.0, 0.1),
    ("max_edge_width_px", "Max edge width / blur (px)", "float", 1.0, 30.0, 0.5),
    ("max_saturation", "Max saturated fraction (glare)", "float", 0.0, 1.0, 0.05),
    ("max_ellipticity", "Max ellipticity (axis ratio - 1)", "float", 0.001, 0.5, 0.005),
    ("lock_tolerance", "Ring lock tolerance (relative)", "float", 0.005, 0.5, 0.005),
    ("smoothing_window", "Smoothing window (frames)", "int", 1, 50, 1),
    ("move_tolerance_px", "Reset smoothing if moved (px)", "float", 0.5, 50.0, 0.5),
]


class ProcessingPanel(ttk.Frame):
    """Pipeline parameters of the measurement being set up."""

    def __init__(self, master, app: "MainWindow") -> None:
        super().__init__(master, padding=8)
        self.app = app
        self._loading = False
        self.columnconfigure(1, weight=1)
        self.info = tk.StringVar(value="Add or edit a measurement to tune its processing.")
        ttk.Label(self, textvariable=self.info, wraplength=340, justify="left").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))
        self.vars: dict[str, tk.Variable] = {}
        self.widgets = []
        for i, (attr, label, kind, lo, hi, inc) in enumerate(PARAM_FIELDS, start=1):
            if kind == "bool":
                var = tk.BooleanVar(value=True)
                w = ttk.Checkbutton(self, text=label, variable=var)
                w.grid(row=i, column=0, columnspan=2, sticky="w", pady=1)
            else:
                var = tk.StringVar()
                ttk.Label(self, text=label).grid(row=i, column=0, sticky="w")
                w = ttk.Spinbox(self, from_=lo, to=hi, increment=inc, width=8, textvariable=var)
                w.grid(row=i, column=1, sticky="w", padx=(6, 0), pady=1)
            var.trace_add("write", lambda *_: self._changed())
            self.vars[attr] = var
            self.widgets.append(w)
        n = len(PARAM_FIELDS) + 1
        bar = ttk.Frame(self)
        bar.grid(row=n, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self.reset_btn = ttk.Button(bar, text="Reset to defaults", command=self._reset)
        self.reset_btn.pack(side="left")
        self.widgets.append(self.reset_btn)
        self.show_points = tk.BooleanVar(value=False)
        ttk.Checkbutton(self, text="Show edge points (green = inlier, red = rejected)",
                        variable=self.show_points,
                        command=lambda: app.set_show_edge_points(self.show_points.get())).grid(
            row=n + 1, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Label(self, foreground="#555", wraplength=340, justify="left", text=(
            "Detection quality summarises edge coverage, inlier support and fit residual. "
            "It is not a statement of dimensional accuracy; establish that with the "
            "validation procedure.")).grid(row=n + 2, column=0, columnspan=2, sticky="w",
                                            pady=(8, 0))
        self.load(None)

    def load(self, params: ProcessingParams | None) -> None:
        self._loading = True
        try:
            state = ["!disabled"] if params is not None else ["disabled"]
            for w in self.widgets:
                w.state(state)
            if params is None:
                self.info.set("Add or edit a measurement to tune its processing.")
                return
            self.info.set("Changes apply live to the measurement being set up.")
            for attr, *_ in PARAM_FIELDS:
                value = getattr(params, attr)
                var = self.vars[attr]
                var.set(value if isinstance(var, tk.BooleanVar) else f"{value:g}")
        finally:
            self._loading = False

    def read_into(self, params: ProcessingParams) -> None:
        for attr, _, kind, lo, hi, _ in PARAM_FIELDS:
            var = self.vars[attr]
            try:
                if kind == "bool":
                    value = bool(var.get())
                elif kind == "int":
                    value = int(float(var.get()))
                else:
                    value = float(var.get())
            except (ValueError, tk.TclError):
                continue
            if lo is not None:
                value = type(value)(min(max(value, lo), hi))
            setattr(params, attr, value)

    def _reset(self) -> None:
        self.load(ProcessingParams(polarity=self.app.draft.params.polarity)
                  if self.app.draft else None)
        self.app.draft_fields_changed()

    def _changed(self) -> None:
        if not self._loading:
            self.app.draft_fields_changed()
