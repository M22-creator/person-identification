"""Web view: model switching and the comparison endpoints, with fake backends (no camera)."""
import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from person_id import Detection
from person_id.identifier import Identifier
from person_id.web import server


class Fake(Identifier):
    def __init__(self, dim):
        self.dim = dim
        self.last_timing = {"detect_ms": 2.0, "embed_ms": 6.0, "faces": 1}

    def detect(self, frame):
        return [Detection((0, 0, 100, 100), np.ones(self.dim))]


@pytest.fixture
def live(tmp_path, monkeypatch):
    built = []

    def build(name):
        built.append(name)
        return Fake({"sface": 128, "arcface_mbf": 512, "arcface_r50": 512}[name])

    monkeypatch.setattr(server, "build_identifier", build)
    monkeypatch.setattr(server, "missing_weights", lambda files: [] if "arcface_w600k_r50.onnx" not in files else ["x"])
    cfg = {
        "identifier": "sface", "source": "x.mp4", "width": 640, "height": 480, "fps": 30,
        "gallery_path": str(tmp_path / "g.npz"), "recognize_threshold": 0.5, "unknown_threshold": 0.3,
        "auto_enroll": False, "bench_results": str(tmp_path / "results.json"),
        "models": {
            "sface": {"gallery_path": str(tmp_path / "s.npz"), "recognize_threshold": 0.4, "unknown_threshold": 0.3},
            "arcface_mbf": {"gallery_path": str(tmp_path / "m.npz"), "recognize_threshold": 0.35, "unknown_threshold": 0.2},
        },
    }
    cfg["fallback"] = {k: cfg[k] for k in ("gallery_path", "recognize_threshold", "unknown_threshold")}
    lv = server.Live(cfg)
    lv.built = built
    return lv


def test_switch_uses_that_models_gallery_and_thresholds(live, tmp_path):
    assert live.identifier == "sface" and live.settings["recognize_threshold"] == 0.4
    live.gallery.add("alice", np.ones(128))
    live.switch("arcface_mbf")
    assert live.identifier == "arcface_mbf"
    assert live.settings == {"gallery_path": str(tmp_path / "m.npz"), "recognize_threshold": 0.35, "unknown_threshold": 0.2}
    assert live.pipeline.recognize_threshold == 0.35 and live.auto.same_threshold == 0.35
    assert len(live.gallery) == 0  # 512-D model starts with its own, empty gallery
    live._step(np.zeros((48, 64, 3), np.uint8))
    s = live.state()
    assert s["identifier"] == "arcface_mbf" and s["latency"]["embed_ms_per_face"] == 6.0
    live.switch("arcface_mbf")
    live.switch("sface")
    assert live.built == ["sface", "arcface_mbf"]  # loaded once, then cached


def test_state_lists_models_and_missing_weights(live):
    models = {m["name"]: m for m in live.state()["models"]}
    assert list(models) == ["sface", "arcface_mbf", "arcface_r50"]
    assert models["arcface_r50"]["available"] is False
    assert "--model arcface_r50" in models["arcface_r50"]["hint"]


def _serve(live):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(live))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def _post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_http_model_switch_compare_page_and_bench(live, tmp_path):
    httpd, base = _serve(live)
    try:
        assert _post(base + "/api/model", {"name": "arcface_mbf"}) == (200, {"ok": True})
        assert live.identifier == "arcface_mbf"
        code, body = _post(base + "/api/model", {"name": "arcface_r50"})
        assert code == 400 and "download_models --model arcface_r50" in body["error"]
        assert _post(base + "/api/model", {"name": "nope"})[0] == 400
        with urllib.request.urlopen(base + "/compare") as r:
            assert r.status == 200 and b"model comparison" in r.read().lower()
        with urllib.request.urlopen(base + "/api/bench") as r:
            assert "tools/bench_models.py" in json.loads(r.read())["error"]
        (tmp_path / "results.json").write_text(json.dumps({"meta": {"order": []}, "models": {}}))
        with urllib.request.urlopen(base + "/api/bench") as r:
            assert json.loads(r.read())["meta"] == {"order": []}
    finally:
        httpd.shutdown()


def test_stream_waits_for_the_first_frame_instead_of_crashing(live):
    httpd, base = _serve(live)
    got = {}

    def read():
        with urllib.request.urlopen(base + "/stream.mjpg", timeout=10) as r:
            got["head"] = r.read(200)

    t = threading.Thread(target=read)
    t.start()
    import time
    time.sleep(0.3)  # client connected before any frame exists
    live._step(np.zeros((48, 64, 3), np.uint8))
    t.join(timeout=10)
    httpd.shutdown()
    assert got["head"].startswith(b"--frame\r\nContent-Type: image/jpeg")


def _faces_after_step(live, frame):
    live._step(frame)
    return live.state()["faces"]


def test_identicon_mode_sends_no_face_image(live):
    # Default mode: nothing of the face is cropped or sent; the page draws a pattern from the ID.
    assert live.state()["thumbnails"] == "identicon"
    frame = np.random.default_rng(0).integers(0, 256, (120, 160, 3), dtype=np.uint8)
    faces = _faces_after_step(live, frame)
    assert faces and all(f["thumb"] is None for f in faces)
    assert all(t is None for t in live.thumbs.values())


def test_pixelate_mode_is_a_6x6_mosaic(live):
    import base64

    import cv2

    live.thumb_mode = "pixelate"
    frame = np.random.default_rng(1).integers(0, 256, (120, 160, 3), dtype=np.uint8)
    thumb = _faces_after_step(live, frame)[0]["thumb"]
    assert thumb.startswith("data:image/png;base64,")
    img = cv2.imdecode(np.frombuffer(base64.b64decode(thumb.split(",", 1)[1]), np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[:2] == (96, 96)
    assert len(np.unique(img.reshape(-1, 3), axis=0)) <= 36  # at most 6x6 distinct blocks


def test_photo_mode_and_bad_mode(live, tmp_path):
    live.thumb_mode = "photo"
    frame = np.random.default_rng(2).integers(0, 256, (120, 160, 3), dtype=np.uint8)
    assert _faces_after_step(live, frame)[0]["thumb"].startswith("data:image/jpeg;base64,")
    cfg = dict(live.cfg, thumbnails="faces")
    with pytest.raises(ValueError, match="thumbnails"):
        server.Live(cfg)
