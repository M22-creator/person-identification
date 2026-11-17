"""Live web view with automatic enrollment:  python -m person_id.web --source 4

Open http://localhost:8000 to see the camera with a box around every face,
who each face is, and new faces being learned into the gallery. A face the
gallery does not know is saved automatically as "person_N" after it has been
seen for a few seconds (see auto_enroll.py in this folder). Only run this where everyone in
front of the camera has agreed to be stored; pause it on the page otherwise.

This process owns the camera, so do not run person_id.run at the same time.
Pass --publisher to send the same results that person_id.run would.

The server listens on 127.0.0.1 only by default. The page shows live video of
people and can remove gallery entries, and it has no login, so do not expose
it on a shared network. To view it from your laptop use an SSH tunnel:

    ssh -L 8000:localhost:8000 <user>@<robot>

Uses only the standard library plus the packages the pipeline already needs.
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import threading
import time
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional

import numpy as np

from .auto_enroll import AutoEnroller, Progress
from ..camera import open_source
from ..config import DEFAULT_PATH, load_config, model_settings
from ..gallery import Gallery
from ..identifier import COMPARED, Identifier, build_identifier, missing_weights
from ..pipeline import Pipeline
from ..publishers import build_publisher
from ..types import Detection, IdentityResult, Status

PAGE_PATH = Path(__file__).with_name("index.html")
COMPARE_PATH = Path(__file__).with_name("compare.html")
ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,32}$")

# BGR box colors. The label text carries the same information as the color.
COLORS = {
    Status.RECOGNIZED: (90, 200, 90),
    Status.UNCERTAIN: (0, 190, 240),
    Status.UNKNOWN: (70, 70, 235),
}
LEARNING_COLOR = (255, 170, 106)


class _Tap(Identifier):
    """Keeps the last raw detections so auto-enrollment can reuse their embeddings."""

    def __init__(self, inner: Identifier) -> None:
        self.inner = inner
        self.last: List[Detection] = []

    def detect(self, frame: np.ndarray) -> List[Detection]:
        self.last = self.inner.detect(frame)
        return self.last


def _label(r: IdentityResult, learning: Optional[Progress]) -> str:
    if learning and learning.enrolled_as:
        return f"saved as {learning.enrolled_as}"
    if learning:
        return f"new face {learning.taken}/{learning.target}"
    if r.status is Status.RECOGNIZED:
        return f"{r.person_id} {r.confidence:.2f}"
    if r.status is Status.UNCERTAIN:
        return f"uncertain {r.confidence:.2f}"
    return "unknown"


class Live:
    """Camera loop in a background thread; HTTP handlers read its latest output."""

    def __init__(self, cfg: dict, loop: bool = False, publisher=None) -> None:
        self.cfg = cfg
        self.loop = loop
        self.publisher = publisher
        self.auto_on = bool(cfg.get("auto_enroll", True))
        self.lock = threading.Lock()  # guards the active model, gallery, auto, faces, thumbs, error
        self._switch_lock = threading.Lock()  # one model switch at a time
        self._identifiers: Dict[str, Identifier] = {}  # loaded once, kept for fast switching
        self._thumbs_by_model: Dict[str, Dict[str, str]] = {}
        self.latency: Dict[str, float] = {}
        self.switch(cfg["identifier"])
        self.frame_ready = threading.Condition()
        self.jpeg: Optional[bytes] = None
        self.frame_no = 0
        self.faces: List[dict] = []
        self.learning: Dict[int, dict] = {}       # candidate key -> progress + thumbnail
        self.fps = 0.0
        self._last_step = 0.0
        self.error: Optional[str] = None

    # ---- model selection -----------------------------------------------

    def switch(self, name: str) -> None:
        """Make `name` the active backend, with its own gallery file and thresholds.
        Half-learned faces are dropped; enrolled people stay in that model's gallery."""
        with self._switch_lock:
            if name not in self._identifiers:
                self._identifiers[name] = build_identifier(name)  # slow: loads weights
            settings = model_settings(self.cfg, name)
            gallery = Gallery.load(settings["gallery_path"])
            tap = _Tap(self._identifiers[name])
            pipeline = Pipeline(tap, gallery, settings["recognize_threshold"], settings["unknown_threshold"])
            auto = AutoEnroller(
                gallery,
                settings["gallery_path"],
                same_threshold=settings["recognize_threshold"],
                samples=int(self.cfg.get("auto_enroll_samples", 10)),
                min_face_px=int(self.cfg.get("auto_enroll_min_face_px", 48)),
            )
            with self.lock:
                self.identifier, self.settings = name, settings
                self.tap, self.gallery, self.pipeline, self.auto = tap, gallery, pipeline, auto
                self.thumbs = self._thumbs_by_model.setdefault(name, {})  # memory only, never saved
                self.faces, self.learning, self.latency = [], {}, {}

    def _record_latency(self, total_ms: float) -> None:
        """Exponential moving averages of where the time of one frame goes (lock held)."""
        t = dict(getattr(self.tap.inner, "last_timing", None) or {})
        sample = {"pipeline_ms": total_ms}
        if "detect_ms" in t:
            sample["detect_ms"] = t["detect_ms"]
            if t.get("faces"):
                sample["embed_ms_per_face"] = t["embed_ms"] / t["faces"]
        for k, v in sample.items():
            old = self.latency.get(k)
            self.latency[k] = v if old is None else 0.9 * old + 0.1 * v

    # ---- camera thread -------------------------------------------------

    def run(self) -> None:
        source = self.cfg["source"]
        is_file = source != "realsense" and not source.isdigit()
        try:
            while True:
                for frame in open_source(
                    source, self.cfg["width"], self.cfg["height"], self.cfg["fps"]
                ):
                    self._step(frame)
                    if is_file:
                        time.sleep(1.0 / max(1, self.cfg["fps"]))
                if not (is_file and self.loop):
                    break
            message = "Video source ended."
        except Exception as e:  # shown on the page instead of dying silently
            message = f"{type(e).__name__}: {e}"
        with self.lock:
            self.error = message
            self.faces, self.learning = [], {}

    def _step(self, frame: np.ndarray) -> None:
        import cv2

        started = time.time()
        with self.lock:
            t0 = time.perf_counter()
            results = self.pipeline.process(frame)
            self._record_latency((time.perf_counter() - t0) * 1000)
            if self.auto_on:
                progress = self.auto.update(self.tap.last, results, started)
            else:
                progress = [None] * len(results)
            live_keys = {c.key for c in self.auto.candidates}
        if self.publisher:
            self.publisher.publish(results)

        shown = frame.copy()
        faces, learning = [], {}
        for r, prog in zip(results, progress):
            if r.bbox is None:
                continue
            x1, y1, x2, y2 = r.bbox
            color = LEARNING_COLOR if prog else COLORS[r.status]
            cv2.rectangle(shown, (x1, y1), (x2, y2), color, 2)
            text = _label(r, prog)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            ty = y1 - 6 if y1 - th - 10 > 0 else y2 + th + 8
            cv2.rectangle(shown, (x1, ty - th - 5), (x1 + tw + 8, ty + 4), color, -1)
            cv2.putText(shown, text, (x1 + 4, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

            thumb = self._thumb(frame, r.bbox)
            face = {**r.to_dict(), "thumb": thumb, "learning": None}
            if prog:
                face["learning"] = {"taken": prog.taken, "target": prog.target, "enrolled_as": prog.enrolled_as}
                if prog.enrolled_as:
                    self.thumbs[prog.enrolled_as] = thumb
                else:
                    learning[prog.key] = {"key": prog.key, "taken": prog.taken, "target": prog.target, "thumb": thumb}
            faces.append(face)

        ok, buf = cv2.imencode(".jpg", shown, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with self.lock:
            # Candidates out of view this frame stay listed until they are forgotten.
            for key, item in self.learning.items():
                if key in live_keys and key not in learning:
                    learning[key] = item
            self.faces, self.learning = faces, learning
            if self._last_step:  # frames actually processed per second, camera wait included
                rate = 1.0 / max(started - self._last_step, 1e-6)
                self.fps = 0.9 * self.fps + 0.1 * rate if self.fps else rate
            self._last_step = started
        if ok:
            with self.frame_ready:
                self.jpeg = buf.tobytes()
                self.frame_no += 1
                self.frame_ready.notify_all()

    @staticmethod
    def _thumb(frame: np.ndarray, bbox) -> Optional[str]:
        import cv2

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None
        crop = frame[y1:y2, x1:x2]
        scale = 96.0 / crop.shape[0]
        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), 96))
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return "data:image/jpeg;base64," + base64.b64encode(buf).decode() if ok else None

    # ---- called from HTTP threads --------------------------------------

    def set_auto(self, enabled: bool) -> None:
        with self.lock:
            self.auto_on = enabled
            if not enabled:  # half-learned faces are dropped, nothing of them is kept
                self.auto.candidates = []
                self.learning = {}

    def remove(self, person_id: str) -> None:
        with self.lock:
            self.gallery.remove(person_id)
            self.gallery.save(self.settings["gallery_path"])
            self.thumbs.pop(person_id, None)

    def state(self) -> dict:
        with self.lock:
            return {
                "faces": self.faces,
                "learning": sorted(self.learning.values(), key=lambda c: c["key"]),
                "people": [{"id": p, "thumb": self.thumbs.get(p)} for p in self.gallery.person_ids],
                "auto_enroll": self.auto_on,
                "fps": round(self.fps, 1),
                "error": self.error,
                "identifier": self.identifier,
                "models": self.models(),
                "latency": {k: round(v, 1) for k, v in self.latency.items()},
                "source": self.cfg["source"],
                "recognize_threshold": self.settings["recognize_threshold"],
                "unknown_threshold": self.settings["unknown_threshold"],
            }

    def models(self) -> List[dict]:
        out = []
        for name, info in COMPARED.items():
            missing = missing_weights(info["files"])
            out.append({"name": name, "label": info["label"], "dim": info["dim"], "about": info["about"],
                        "available": not missing,
                        "hint": f"python -m person_id.download_models --model {name}" if missing else None})
        if self.identifier not in COMPARED:
            out.append({"name": self.identifier, "label": self.identifier, "available": True})
        return out

    def bench(self) -> dict:
        path = Path(self.cfg.get("bench_results", "data/bench/results.json"))
        if not path.exists():
            return {"error": f"No benchmark results at {path}. Run: python tools/bench_models.py "
                             "--lfw data/bench/lfw --out data/bench/results.json"}
        return json.loads(path.read_text())


def make_handler(live: Live):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # the polling would flood the terminal
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj: dict) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            if self.path == "/":
                self._send(200, PAGE_PATH.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/compare":
                self._send(200, COMPARE_PATH.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/bench":
                self._json(200, live.bench())
            elif self.path == "/api/state":
                self._json(200, live.state())
            elif self.path == "/stream.mjpg":
                self._stream()
            else:
                self._json(404, {"error": "not found"})

        def _stream(self):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            seen = -1
            try:
                while True:
                    with live.frame_ready:
                        # jpeg is None until the camera delivers its first frame.
                        if not live.frame_ready.wait_for(
                            lambda: live.jpeg is not None and live.frame_no != seen, timeout=5
                        ):
                            continue
                        seen, jpeg = live.frame_no, live.jpeg
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                        + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n"
                    )
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):
            # Requiring JSON keeps other websites open in the same browser from
            # posting here: a cross-site JSON request needs a CORS preflight,
            # which this server never answers.
            if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
                return self._json(415, {"error": "Content-Type must be application/json"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(min(length, 10_000)) or b"{}")
                if not isinstance(body, dict):
                    raise ValueError
            except ValueError:
                return self._json(400, {"error": "Invalid JSON"})

            if self.path == "/api/auto":
                if not isinstance(body.get("enabled"), bool):
                    return self._json(400, {"error": "'enabled' must be true or false."})
                live.set_auto(body["enabled"])
                return self._json(200, {"ok": True})
            if self.path == "/api/model":
                name = str(body.get("name", ""))
                info = {m["name"]: m for m in live.models()}.get(name)
                if info is None:
                    return self._json(400, {"error": f"Unknown model '{name}'."})
                if not info["available"]:
                    return self._json(400, {"error": f"Weights missing. On the robot run: {info['hint']}"})
                try:
                    live.switch(name)
                except Exception as e:  # e.g. onnxruntime not installed
                    return self._json(500, {"error": f"Could not load {name}: {type(e).__name__}: {e}"})
                return self._json(200, {"ok": True})
            if self.path == "/api/remove":
                pid = str(body.get("person_id", ""))
                if not ID_PATTERN.match(pid):
                    return self._json(400, {"error": "Invalid ID."})
                live.remove(pid)
                return self._json(200, {"ok": True})
            self._json(404, {"error": "not found"})

    return Handler


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=DEFAULT_PATH)
    p.add_argument("--source", help='"realsense", a device index, or a video file')
    p.add_argument("--identifier")
    p.add_argument("--host", default="127.0.0.1", help="0.0.0.0 exposes the page to the whole network")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--loop", action="store_true", help="replay a video file forever")
    p.add_argument("--publisher", choices=["none", "stdout", "ros2"], default="none",
                   help="also publish results, like person_id.run does")
    p.add_argument("--no-auto-enroll", action="store_true", help="start with auto-enrollment paused")
    a = p.parse_args()

    cfg = load_config(a.config, source=a.source, identifier=a.identifier)
    if a.no_auto_enroll:
        cfg["auto_enroll"] = False
    publisher = None if a.publisher == "none" else build_publisher(a.publisher, cfg["ros2_topic"])
    live = Live(cfg, loop=a.loop, publisher=publisher)
    threading.Thread(target=live.run, daemon=True).start()

    server = ThreadingHTTPServer((a.host, a.port), make_handler(live))
    server.daemon_threads = True
    print(f"Person ID live view: http://{a.host}:{a.port}  (Ctrl+C to stop)")
    if a.host not in ("127.0.0.1", "localhost", "::1"):
        print("WARNING: no login. Anyone who can reach this port can watch the camera and edit the gallery.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if publisher:
            publisher.close()
