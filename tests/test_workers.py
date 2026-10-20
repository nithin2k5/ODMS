import queue
import time

import numpy as np

from odms.camera.source import CameraState, CameraWorker, put_latest
from odms.measurement.engine import EngineConfig
from odms.processing import ProcessingWorker, make_queues


def test_put_latest_keeps_newest():
    q = queue.Queue(maxsize=2)
    for i in range(5):
        put_latest(q, i)
    assert [q.get_nowait(), q.get_nowait()] == [3, 4]


def test_unavailable_camera_does_not_raise():
    frames, _ = make_queues()
    worker = CameraWorker(97, frames, backend="Auto")
    worker.start()
    deadline = time.time() + 10
    while time.time() < deadline and worker.status.state is CameraState.CONNECTING:
        time.sleep(0.05)
    assert worker.status.state in (CameraState.UNAVAILABLE, CameraState.CONNECTING)
    worker.stop()
    worker.join(timeout=10)
    assert not worker.is_alive()


def test_processing_worker_static_frame():
    frames, results = make_queues()
    worker = ProcessingWorker(frames, results)
    worker.start()
    try:
        worker.set_config(EngineConfig(camera_id="image-file", version=1))
        worker.set_static_frame(np.zeros((120, 160, 3), np.uint8), "image-file")
        result = results.get(timeout=5)
        assert result.resolution == (160, 120)
        # Unchanged frame + config is not reprocessed.
        time.sleep(0.3)
        assert results.empty()
        worker.set_config(EngineConfig(camera_id="image-file", version=2))
        assert results.get(timeout=5).config_version == 2
    finally:
        worker.stop()
        worker.join(timeout=5)
