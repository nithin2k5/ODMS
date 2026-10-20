"""Background worker that runs the measurement engine off the UI thread.

The worker takes the newest camera frame (or a static frozen/loaded frame),
processes it with the latest :class:`EngineConfig` snapshot and publishes a
:class:`FrameResult` into a bounded result queue that the Tk main loop polls
with ``after()``.  Tk widgets are never touched from this thread.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
import traceback

import numpy as np

from .camera.source import Frame, put_latest
from .measurement.engine import EngineConfig, FrameResult, MeasurementEngine

log = logging.getLogger(__name__)


class ProcessingWorker(threading.Thread):
    def __init__(self, frame_queue: queue.Queue, result_queue: queue.Queue,
                 engine: MeasurementEngine | None = None) -> None:
        super().__init__(name="processing", daemon=True)
        self.engine = engine or MeasurementEngine()
        self._frames = frame_queue
        self._results = result_queue
        self._stop_event = threading.Event()
        self._wake = threading.Event()
        self._lock = threading.Lock()
        self._config = EngineConfig()
        self._static: Frame | None = None
        self._static_counter = 0
        self._last_frame: Frame | None = None
        self._fps = 0.0
        self._error: str | None = None

    # -- called from the UI thread -------------------------------------------
    def set_config(self, config: EngineConfig) -> None:
        with self._lock:
            self._config = config
        self._wake.set()

    def set_static_frame(self, image: np.ndarray, source_id: str) -> None:
        with self._lock:
            self._static_counter -= 1
            self._static = Frame(self._static_counter, time.time(), image, source_id)
        self.engine.reset()
        self._wake.set()

    def freeze(self) -> bool:
        """Hold the most recently processed live frame."""
        with self._lock:
            frame = self._last_frame
        if frame is None:
            return False
        self.set_static_frame(frame.image, frame.source_id)
        return True

    def set_live(self) -> None:
        with self._lock:
            self._static = None
        self.engine.reset()
        self._wake.set()

    @property
    def is_static(self) -> bool:
        with self._lock:
            return self._static is not None

    @property
    def fps(self) -> float:
        return self._fps

    def take_error(self) -> str | None:
        err, self._error = self._error, None
        return err

    def stop(self) -> None:
        self._stop_event.set()
        self._wake.set()

    # -- thread --------------------------------------------------------------
    def run(self) -> None:
        last_key = None
        last_t = None
        while not self._stop_event.is_set():
            with self._lock:
                static = self._static
                config = self._config
            if static is not None:
                frame = static
                if (frame.frame_id, config.version) == last_key:
                    self._wake.wait(0.1)
                    self._wake.clear()
                    continue
            else:
                try:
                    frame = self._frames.get(timeout=0.1)
                except queue.Empty:
                    continue
                with self._lock:
                    if self._static is not None:  # frozen while we waited
                        continue
                    self._last_frame = frame
            last_key = (frame.frame_id, config.version)
            try:
                result = self.engine.process(frame.image, config, frame.frame_id)
            except Exception:  # keep the worker alive whatever happens
                self._error = traceback.format_exc(limit=4)
                log.exception("processing failed")
                time.sleep(0.05)
                continue
            now = time.perf_counter()
            if last_t is not None and static is None:
                dt = now - last_t
                if dt > 0:
                    self._fps = 1.0 / dt if self._fps == 0 else 0.85 * self._fps + 0.15 / dt
            elif static is not None:
                self._fps = 1000.0 / max(result.processing_ms, 1e-3)
            last_t = now
            put_latest(self._results, result)


def make_queues() -> tuple[queue.Queue, queue.Queue]:
    """Bounded frame and result queues (latest-value semantics)."""
    return queue.Queue(maxsize=2), queue.Queue(maxsize=2)


__all__ = ["ProcessingWorker", "FrameResult", "make_queues"]
