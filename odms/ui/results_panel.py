"""Results table with Save Result / Export CSV controls."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable

from ..measurement.models import SpecResult, Status
from ..units import DEFAULT_UNIT, format_length

COLUMNS = (
    ("measurement", "Measurement", 190),
    ("diameter", "Diameter (mm)", 110),
    ("pixels", "Diameter (px)", 100),
    ("quality", "Detection quality", 120),
    ("status", "Status", 150),
    ("detail", "Detail", 420),
)

STATUS_TAGS = {
    Status.PASS: "pass",
    Status.MEASURED: "ok",
    Status.FAIL: "fail",
    Status.NOT_DETECTED: "lost",
    Status.REJECTED: "rejected",
    Status.CALIBRATION_REQUIRED: "cal",
    Status.REVALIDATION_REQUIRED: "cal",
    Status.CALIBRATION_INVALID: "cal",
    Status.INCOMPLETE: "lost",
}


class ResultsPanel(ttk.Frame):
    def __init__(self, master, on_save: Callable[[], None],
                 on_export: Callable[[], None]) -> None:
        super().__init__(master, padding=(4, 2))
        bar = ttk.Frame(self)
        bar.pack(fill="x")
        ttk.Label(bar, text="Results", font=("Segoe UI", 10, "bold")).pack(side="left")
        ttk.Button(bar, text="Export CSV...", command=on_export).pack(side="right", padx=2)
        ttk.Button(bar, text="Save Result", command=on_save).pack(side="right", padx=2)
        self.saved_var = tk.StringVar(value="Saved this session: 0")
        ttk.Label(bar, textvariable=self.saved_var).pack(side="right", padx=10)

        frame = ttk.Frame(self)
        frame.pack(fill="both", expand=True, pady=(2, 0))
        self.tree = ttk.Treeview(frame, columns=[c[0] for c in COLUMNS], show="headings",
                                 height=5)
        for key, title, width in COLUMNS:
            self.tree.heading(key, text=title)
            self.tree.column(key, width=width, minwidth=60,
                             stretch=key == "detail", anchor="w")
        sb = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.tree.tag_configure("pass", background="#d6f5d6")
        self.tree.tag_configure("ok", background="#eef7ee")
        self.tree.tag_configure("fail", background="#f8d0d0")
        self.tree.tag_configure("lost", background="#f1e1e1")
        self.tree.tag_configure("rejected", background="#ffe7c7")
        self.tree.tag_configure("cal", background="#fff6c2")
        self._items: dict[str, str] = {}
        self.unit = DEFAULT_UNIT

    def set_unit(self, unit: str) -> None:
        self.unit = unit
        self.tree.heading("diameter", text=f"Diameter ({unit})")

    def set_saved_count(self, n: int) -> None:
        self.saved_var.set(f"Saved this session: {n}")

    def update_results(self, results: list[SpecResult]) -> None:
        seen = set()
        for index, r in enumerate(results):
            seen.add(r.uid)
            values = self._values(r)
            tag = STATUS_TAGS.get(r.status, "")
            item = self._items.get(r.uid)
            if item is None or not self.tree.exists(item):
                self._items[r.uid] = self.tree.insert("", index, values=values, tags=(tag,))
            else:
                self.tree.item(item, values=values, tags=(tag,))
                if self.tree.index(item) != index:
                    self.tree.move(item, "", index)
        for uid in list(self._items):
            if uid not in seen:
                if self.tree.exists(self._items[uid]):
                    self.tree.delete(self._items[uid])
                del self._items[uid]

    def _values(self, r: SpecResult) -> tuple[str, ...]:
        unit = self.unit
        value = format_length(r.diameter_mm, unit, with_unit=False)
        px = f"{r.diameter_px:.2f}" if r.diameter_px is not None else "-"
        if r.detected:
            q = r.quality
            quality = f"{q.score:.0f}/100"
            fit = f"rms {q.rms_px:.2f}px, cover {q.coverage:.0%}, inliers {q.inlier_ratio:.0%}"
        else:
            quality, fit = "-", ""
        detail = r.reason
        if fit:
            detail = f"{fit}" + (f" | {r.reason}" if r.reason else "")
        if r.diameter_mm is not None and (r.min_mm is not None or r.max_mm is not None):
            lo = format_length(r.min_mm, unit, False) if r.min_mm is not None else "-inf"
            hi = format_length(r.max_mm, unit, False) if r.max_mm is not None else "+inf"
            detail = f"limits [{lo}, {hi}] {unit} | " + detail
        return (r.name, value, px, quality, r.status.value, detail)
