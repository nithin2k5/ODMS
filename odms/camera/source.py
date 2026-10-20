"""USB camera capture in a background thread.

Frames go into a small bounded queue; when the consumer falls behind the
oldest frame is dropped so the preview always shows the latest image.  Open
failures, read failures and disconnections are reported through
:attr:`CameraWorker.status` and retried; they never raise into the UI.
"""
from __future__ import annotations

import logging
import queue
import sys
import threading
import time
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

log = logging.getLogger(__name__)

BACKENDS: dict[str, int] = {"Auto": cv2.CAP_ANY}
for _name, _attr in (("DirectShow", "CAP_DSHOW"), ("Media Foundation", "CAP_MSMF"),
                     ("V4L2", "CAP_V4L2"), ("AVFoundation", "CAP_AVFOUNDATION")):
    if hasattr(cv2, _attr):
        BACKENDS[_name] = getattr(cv2, _attr)

RESOLUTIONS = ["Default", "640x480", "800x600", "1280x720", "1280x960",
               "1600x1200", "1920x1080", "2592x1944", "3840x2160"]

MAX_CONSECUTIVE_FAILURES = 15
RECONNECT_INTERVAL_S = 2.0


def default_backend() -> str:
    if sys.platform.startswith("win"):
        return "DirectShow"
    if sys.platform.startswith("linux"):
        return "V4L2"
    if sys.platform == "darwin":
        return "AVFoundation"
    return "Auto"


def camera_identity(index: int) -> str:
    """Identifier stored with calibrations.  OpenCV exposes no serial number,
    so the identity is the device index; keep cameras on the same ports."""
    return f"usb:{index}"


def parse_resolution(text: str) -> tuple[int, int] | None:
    try:
        w, h = text.lower().split("x")
        return int(w), int(h)
    except (ValueError, AttributeError):
        return None


def probe_cameras(max_index: int = 6, backend: str = "Auto") -> list[int]:
    """Indices that open and deliver a frame.  Slow; call off the UI thread."""
    found = []
    api = BACKENDS.get(backend, cv2.CAP_ANY)
    for index in range(max_index):
        cap = cv2.VideoCapture(index, api)
        try:
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    found.append(index)
        finally:
            cap.release()
    return found


def put_latest(q: queue.Queue, item) -> None:
    """Put ``item``, discarding the oldest entry if the queue is full."""
    while True:
        try:
            q.put_nowait(item)
            return
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass


class CameraState(str, Enum):
    STOPPED = "Stopped"
    CONNECTING = "Connecting"
    RUNNING = "Connected"
    DISCONNECTED = "Disconnected"
    UNAVAILABLE = "Unavailable"


@dataclass
class CameraStatus:
    state: CameraState
    message: str = ""
    fps: float = 0.0
    resolution: tuple[int, int] | None = None


@dataclass
class Frame:
    frame_id: int
    timestamp: float
    image: np.ndarray
    source_id: str


class CameraWorker(threading.Thread):
    def __init__(self, index: int, frame_queue: queue.Queue, backend: str = "Auto",
                 resolution: tuple[int, int] | None = None) -> None:
        super().__init__(name=f"camera-{index}", daemon=True)
        self.index = index
        self.backend = backend
        self.requested_resolution = resolution
        self.source_id = camera_identity(index)
        self._queue = frame_queue
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._status = CameraStatus(CameraState.CONNECTING, "Opening camera...")
        self._frame_id = 0

    @property
    def status(self) -> CameraStatus:
        with self._lock:
            return CameraStatus(**vars(self._status))

    def _set_status(self, state: CameraState, message: str = "", fps: float | None = None,
                    resolution: tuple[int, int] | None = None) -> None:
        with self._lock:
            self._status.state = state
            self._status.message = message
            if fps is not None:
                self._status.fps = fps
            if resolution is not None:
                self._status.resolution = resolution

    def stop(self) -> None:
        self._stop_event.set()

    def _open(self) -> cv2.VideoCapture | None:
        cap = cv2.VideoCapture(self.index, BACKENDS.get(self.backend, cv2.CAP_ANY))
        if not cap.isOpened():
            cap.release()
            return None
        if self.requested_resolution:
            w, h = self.requested_resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def run(self) -> None:
        cap = None
        failures = 0
        fps = 0.0
        last_t = None
        ever_connected = False
        try:
            while not self._stop_event.is_set():
                if cap is None:
                    self._set_status(CameraState.CONNECTING, f"Opening camera {self.index}...")
                    try:
                        cap = self._open()
                    except cv2.error as exc:  # pragma: no cover - driver specific
                        log.warning("camera open failed: %s", exc)
                        cap = None
                    if cap is None:
                        state = CameraState.DISCONNECTED if ever_connected else CameraState.UNAVAILABLE
                        self._set_status(state, f"Camera {self.index} not available - retrying",
                                         fps=0.0)
                        self._stop_event.wait(RECONNECT_INTERVAL_S)
                        continue
                    failures = 0
                    last_t = None
                try:
                    ok, image = cap.read()
                except cv2.error as exc:  # pragma: no cover - driver specific
                    log.warning("camera read raised: %s", exc)
                    ok, image = False, None
                if not ok or image is None or image.size == 0:
                    failures += 1
                    if failures >= MAX_CONSECUTIVE_FAILURES:
                        cap.release()
                        cap = None
                        self._set_status(CameraState.DISCONNECTED,
                                         "Frame reads failing - reconnecting", fps=0.0)
                        self._stop_event.wait(RECONNECT_INTERVAL_S)
                    else:
                        self._stop_event.wait(0.02)
                    continue
                failures = 0
                ever_connected = True
                now = time.perf_counter()
                if last_t is not None:
                    dt = now - last_t
                    if dt > 0:
                        fps = 1.0 / dt if fps == 0 else 0.9 * fps + 0.1 / dt
                last_t = now
                self._frame_id += 1
                put_latest(self._queue, Frame(self._frame_id, time.time(), image, self.source_id))
                self._set_status(CameraState.RUNNING, "", fps=fps,
                                 resolution=(image.shape[1], image.shape[0]))
        finally:
            if cap is not None:
                cap.release()
            self._set_status(CameraState.STOPPED, "", fps=0.0)
