import json
import time
from urllib.error import HTTPError
from urllib.request import urlopen
import numpy as np
from indoorcam.video import FrameHub, start_api, fit


def test_stale_source_health_and_snapshot_expire():
    hub = FrameHub()
    server = start_api(hub, "127.0.0.1", 0)
    root = f"http://127.0.0.1:{server.server_port}"
    try:
        hub.set(np.zeros((32,32,3), np.uint8), {"count": 2})
        assert json.load(urlopen(root+"/health"))["count"] == 2
        assert urlopen(root+"/snapshot.jpg").read().startswith(b"\xff\xd8")
        hub.updated = time.monotonic()-6
        try:
            urlopen(root+"/health")
            assert False, "stale camera must be unavailable"
        except HTTPError as exc:
            assert exc.code == 503
    finally:
        server.shutdown()
        server.server_close()


def test_fit_preserves_aspect():
    frame = fit(np.full((100,100,3), 255, np.uint8), 200, 100)
    assert frame.shape == (100,200,3)
    assert np.all(frame[:, :50] == 0)
    assert np.all(frame[:, 50:150] == 255)
