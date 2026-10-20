"""Live preview canvas with zoom, pan and image-coordinate mouse events."""
from __future__ import annotations

import tkinter as tk
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageTk

from ..geometry import ViewTransform

MouseCallback = Callable[[float, float, tk.Event], None]
ZOOM_STEP = 1.25
MAX_ZOOM = 32.0


class PreviewCanvas(tk.Canvas):
    """Shows a BGR image scaled to fit, plus interactive overlay shapes.

    Left-button events are reported in *original image* coordinates through
    ``on_press`` / ``on_drag`` / ``on_release`` / ``on_motion``.  The mouse
    wheel zooms about the cursor; right or middle drag pans.
    """

    def __init__(self, master, **kw) -> None:
        super().__init__(master, background="#1e1e1e", highlightthickness=0, **kw)
        self._image: np.ndarray | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._img_item = self.create_image(0, 0, anchor="nw")
        self._placeholder = self.create_text(
            10, 10, anchor="nw", fill="#bbbbbb", font=("Segoe UI", 11),
            text="No image.\nStart a camera or use File > Load Image.")
        self.zoom = 1.0
        self.center: tuple[float, float] | None = None
        self.transform: ViewTransform | None = None
        self.overlay_shapes: list[dict] = []
        self.on_press: MouseCallback | None = None
        self.on_drag: MouseCallback | None = None
        self.on_release: MouseCallback | None = None
        self.on_motion: MouseCallback | None = None
        self.on_view_changed: Callable[[], None] | None = None
        self._pan_start = None

        self.bind("<Configure>", lambda e: self.render())
        self.bind("<ButtonPress-1>", lambda e: self._mouse(self.on_press, e))
        self.bind("<B1-Motion>", lambda e: self._mouse(self.on_drag, e))
        self.bind("<ButtonRelease-1>", lambda e: self._mouse(self.on_release, e))
        self.bind("<Motion>", lambda e: self._mouse(self.on_motion, e))
        self.bind("<MouseWheel>", self._wheel)
        self.bind("<Button-4>", lambda e: self._zoom_at(e.x, e.y, ZOOM_STEP))
        self.bind("<Button-5>", lambda e: self._zoom_at(e.x, e.y, 1 / ZOOM_STEP))
        for button in (2, 3):
            self.bind(f"<ButtonPress-{button}>", self._pan_begin)
            self.bind(f"<B{button}-Motion>", self._pan_move)

    # -- public ---------------------------------------------------------------
    @property
    def image_size(self) -> tuple[int, int] | None:
        if self._image is None:
            return None
        return self._image.shape[1], self._image.shape[0]

    def set_image(self, image: np.ndarray | None) -> None:
        self._image = image
        self.render()

    def set_overlay(self, shapes: list[dict]) -> None:
        self.overlay_shapes = shapes
        self._draw_overlay()

    def reset_view(self) -> None:
        self.zoom, self.center = 1.0, None
        self.render()

    def zoom_by(self, factor: float) -> None:
        self._zoom_at(self.winfo_width() / 2, self.winfo_height() / 2, factor)

    def render(self) -> None:
        if self._image is None:
            self.itemconfigure(self._placeholder, state="normal")
            self.itemconfigure(self._img_item, image="")
            self.transform = None
            self.delete("overlay")
            return
        self.itemconfigure(self._placeholder, state="hidden")
        cw, ch = self.winfo_width(), self.winfo_height()
        if cw < 4 or ch < 4:
            return
        img = self._image
        h, w = img.shape[:2]
        vt = ViewTransform.compute(w, h, cw, ch, self.zoom, self.center)
        self.transform = vt
        crop = img[vt.crop_y:vt.crop_y + vt.crop_h, vt.crop_x:vt.crop_x + vt.crop_w]
        if vt.scale_x < 1.0:
            interp = cv2.INTER_AREA
        elif vt.scale_x >= 4.0:
            interp = cv2.INTER_NEAREST   # show real pixels when zoomed in
        else:
            interp = cv2.INTER_LINEAR
        disp = cv2.resize(crop, (vt.disp_w, vt.disp_h), interpolation=interp)
        if disp.ndim == 2:
            rgb = cv2.cvtColor(disp, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.itemconfigure(self._img_item, image=self._photo)
        self.coords(self._img_item, vt.offset_x, vt.offset_y)
        self._draw_overlay()

    # -- overlay -------------------------------------------------------------
    def _draw_overlay(self) -> None:
        self.delete("overlay")
        vt = self.transform
        if vt is None:
            return
        for s in self.overlay_shapes:
            kind = s["type"]
            colour = s.get("color", "#00e5ff")
            width = s.get("width", 2)
            dash = s.get("dash")
            if kind == "line":
                pts = [c for p in s["points"] for c in vt.to_canvas(*p)]
                if len(pts) >= 4:
                    self.create_line(*pts, fill=colour, width=width, dash=dash, tags="overlay")
            elif kind == "rect":
                x, y, w, h = s["rect"]
                x0, y0 = vt.to_canvas(x, y)
                x1, y1 = vt.to_canvas(x + w, y + h)
                self.create_rectangle(x0, y0, x1, y1, outline=colour, width=width, dash=dash,
                                      tags="overlay")
            elif kind == "circle":
                cx, cy, r = s["circle"]
                x, y = vt.to_canvas(cx, cy)
                rc = vt.length_to_canvas(r)
                self.create_oval(x - rc, y - rc, x + rc, y + rc, outline=colour, width=width,
                                 dash=dash, tags="overlay")
            elif kind == "marker":
                x, y = vt.to_canvas(*s["point"])
                k = s.get("size", 7)
                self.create_line(x - k, y, x + k, y, fill=colour, width=width, tags="overlay")
                self.create_line(x, y - k, x, y + k, fill=colour, width=width, tags="overlay")
                if s.get("label"):
                    self.create_text(x + k + 3, y - k - 3, text=s["label"], fill=colour,
                                     anchor="sw", font=("Segoe UI", 10, "bold"), tags="overlay")

    # -- mouse ---------------------------------------------------------------
    def _mouse(self, callback: MouseCallback | None, event: tk.Event) -> None:
        if callback is None or self.transform is None:
            return
        x, y = self.transform.to_image(event.x, event.y)
        callback(x, y, event)

    def _wheel(self, event: tk.Event) -> None:
        self._zoom_at(event.x, event.y, ZOOM_STEP if event.delta > 0 else 1 / ZOOM_STEP)

    def _zoom_at(self, cx: float, cy: float, factor: float) -> None:
        vt = self.transform
        if vt is None:
            return
        new_zoom = float(np.clip(self.zoom * factor, 1.0, MAX_ZOOM))
        if new_zoom == self.zoom:
            return
        px, py = vt.to_image(cx, cy)
        mx, my = vt.to_image(self.winfo_width() / 2, self.winfo_height() / 2)
        k = self.zoom / new_zoom
        self.center = (px + (mx - px) * k, py + (my - py) * k)
        self.zoom = new_zoom
        if self.zoom == 1.0:
            self.center = None
        self.render()
        if self.on_view_changed:
            self.on_view_changed()

    def _pan_begin(self, event: tk.Event) -> None:
        vt = self.transform
        if vt is None:
            return
        mid = vt.to_image(self.winfo_width() / 2, self.winfo_height() / 2)
        self._pan_start = (event.x, event.y, mid)

    def _pan_move(self, event: tk.Event) -> None:
        if self._pan_start is None or self.transform is None or self.zoom <= 1.0:
            return
        x0, y0, (mx, my) = self._pan_start
        s = self.transform.scale_x
        self.center = (mx - (event.x - x0) / s, my - (event.y - y0) / s)
        self.render()
