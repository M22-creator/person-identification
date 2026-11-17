"""Identifier backends: frame in, list of Detection out.

To add a model: subclass Identifier, implement detect(), and register it
in BACKENDS. Nothing else in the repo depends on which face model you use.

Backends compared on the web view's /compare page all share the same YuNet
face detector, so only the recognition (embedding) model differs:

    sface        OpenCV SFace, 128-D                     (Apache-2.0)
    arcface_mbf  InsightFace buffalo_s w600k_mbf, 512-D  (non-commercial research only)
    arcface_r50  InsightFace buffalo_l w600k_r50, 512-D  (non-commercial research only)

Optional: "insightface" (the full insightface package, SCRFD detector), and
"dummy" (finds nobody). A gallery built with one backend cannot be used with
another: embedding sizes and score scales differ.
"""
from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .types import Detection


class Identifier(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """frame: HxWx3 BGR uint8. Return one Detection per person found."""


class DummyIdentifier(Identifier):
    """Finds nobody. Lets the rest of the pipeline run before a model is wired in."""

    def detect(self, frame: np.ndarray) -> List[Detection]:
        return []


MODEL_DIR = Path(os.environ.get("PERSON_ID_MODEL_DIR", Path(__file__).resolve().parent.parent / "models"))
YUNET_FILE = "face_detection_yunet_2023mar.onnx"
SFACE_FILE = "face_recognition_sface_2021dec.onnx"
ARCFACE_R50_FILE = "arcface_w600k_r50.onnx"
ARCFACE_MBF_FILE = "arcface_w600k_mbf.onnx"

# 5-point template of the standard 112x112 ArcFace crop (insightface face_align.arcface_dst),
# in YuNet's landmark order: right eye, left eye, nose tip, right mouth corner, left mouth corner.
ARCFACE_TEMPLATE = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32,
)


def similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform (Umeyama) mapping src -> dst points. Returns 2x3."""
    src, dst = np.asarray(src, np.float64), np.asarray(dst, np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s, d = src - mu_s, dst - mu_d
    cov = d.T @ s / len(src)
    u, sig, vt = np.linalg.svd(cov)
    sign = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[1, 1] = -1
    r = u @ sign @ vt
    scale = np.trace(np.diag(sig) @ sign) / s.var(0).sum()
    t = mu_d - scale * r @ mu_s
    return np.hstack([scale * r, t[:, None]]).astype(np.float32)


def missing_weights(files: List[str], model_dir: Optional[os.PathLike] = None) -> List[str]:
    d = Path(model_dir or MODEL_DIR)
    return [f for f in files if not (d / f).exists()]


class YuNetBackend(Identifier):
    """YuNet face detector shared by the compared backends; subclasses add embed().

    After each detect() call, ``last_timing`` holds {"detect_ms", "embed_ms", "faces"}
    so the web view can show where the time goes.
    """

    name = "yunet"
    weight_files: List[str] = []  # recognition weights, besides YuNet

    def __init__(
        self,
        model_dir: Optional[os.PathLike] = None,
        score_threshold: float = 0.8,
        min_face: int = 40,
    ) -> None:
        import cv2

        d = Path(model_dir or MODEL_DIR)
        missing = missing_weights([YUNET_FILE] + self.weight_files, d)
        if missing:
            raise FileNotFoundError(
                f"Missing model file(s) {missing} in {d}. "
                f"Run: python -m person_id.download_models --model {self.name}"
            )
        self._cv2 = cv2
        self.model_dir = d
        self.min_face = min_face
        self.detector = cv2.FaceDetectorYN.create(str(d / YUNET_FILE), "", (320, 320), score_threshold, 0.3, 5000)
        self._size: Tuple[int, int] = (0, 0)
        self.last_timing: Dict[str, float] = {"detect_ms": 0.0, "embed_ms": 0.0, "faces": 0}

    def faces(self, frame: np.ndarray) -> np.ndarray:
        """Raw YuNet rows: x, y, w, h, 5 landmarks (x, y), score. Shape (N, 15)."""
        h, w = frame.shape[:2]
        if (w, h) != self._size:
            self.detector.setInputSize((w, h))
            self._size = (w, h)
        _, faces = self.detector.detect(frame)
        return np.zeros((0, 15), np.float32) if faces is None else faces

    @abstractmethod
    def embed(self, frame: np.ndarray, face: np.ndarray) -> np.ndarray:
        """1-D embedding of one YuNet face row."""

    def detect(self, frame: np.ndarray) -> List[Detection]:
        h, w = frame.shape[:2]
        t0 = time.perf_counter()
        rows = self.faces(frame)
        t1 = time.perf_counter()
        out = []
        for f in rows:
            x, y, fw, fh = f[:4]
            if min(fw, fh) < self.min_face:
                continue  # too small to identify reliably
            bbox = (
                int(max(0, x)), int(max(0, y)),
                int(min(w, x + fw)), int(min(h, y + fh)),
            )
            out.append(Detection(bbox=bbox, embedding=self.embed(frame, f)))
        t2 = time.perf_counter()
        self.last_timing = {"detect_ms": (t1 - t0) * 1000, "embed_ms": (t2 - t1) * 1000, "faces": len(out)}
        return out


class SFaceIdentifier(YuNetBackend):
    """OpenCV model zoo: YuNet face detector + SFace 128-D face embedding.

    Needs only opencv-python (>= 4.8). Weights live in models/ (git-ignored);
    fetch them with `python -m person_id.download_models`.
    """

    name = "sface"
    weight_files = [SFACE_FILE]

    def __init__(self, model_dir: Optional[os.PathLike] = None, **kw) -> None:
        super().__init__(model_dir, **kw)
        self.recognizer = self._cv2.FaceRecognizerSF.create(str(self.model_dir / SFACE_FILE), "")

    def embed(self, frame: np.ndarray, face: np.ndarray) -> np.ndarray:
        aligned = self.recognizer.alignCrop(frame, face)
        return self.recognizer.feature(aligned).reshape(-1).astype(np.float32)


class ArcFaceOnnxIdentifier(YuNetBackend):
    """YuNet detector + an InsightFace ArcFace recognition model run with onnxruntime.

    Uses the model's standard preprocessing: 5-point similarity alignment to a
    112x112 crop, RGB, (x - 127.5) / 127.5. Needs onnxruntime (CPU wheels exist
    for x86_64 and aarch64). The weights are licensed by InsightFace for
    non-commercial research use only.
    """

    weight_file = ""

    def __init__(self, model_dir: Optional[os.PathLike] = None, threads: int = 0, **kw) -> None:
        self.weight_files = [self.weight_file]
        super().__init__(model_dir, **kw)
        import onnxruntime as ort

        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(
            str(self.model_dir / self.weight_file), opts, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def align(self, frame: np.ndarray, face: np.ndarray) -> np.ndarray:
        landmarks = np.asarray(face[4:14], np.float32).reshape(5, 2)
        m = similarity_transform(landmarks, ARCFACE_TEMPLATE)
        return self._cv2.warpAffine(frame, m, (112, 112), borderValue=0.0)

    def embed(self, frame: np.ndarray, face: np.ndarray) -> np.ndarray:
        crop = self.align(frame, face)
        blob = (crop[:, :, ::-1].astype(np.float32) - 127.5) / 127.5  # BGR -> RGB
        blob = np.ascontiguousarray(blob.transpose(2, 0, 1)[None])
        return self.session.run(None, {self.input_name: blob})[0].reshape(-1).astype(np.float32)


class ArcFaceR50Identifier(ArcFaceOnnxIdentifier):
    name = "arcface_r50"
    weight_file = ARCFACE_R50_FILE


class ArcFaceMBFIdentifier(ArcFaceOnnxIdentifier):
    name = "arcface_mbf"
    weight_file = ARCFACE_MBF_FILE


class InsightFaceIdentifier(Identifier):
    """SCRFD face detector + ArcFace embeddings (InsightFace "buffalo_l" pack).

    Needs: pip install insightface onnxruntime   (onnxruntime-gpu on a CUDA machine)
    Weights (~300 MB) download to ~/.insightface on first use. They are
    licensed for non-commercial research use only.

    Embeddings are 512-D and L2-normalized, so gallery cosine scores are
    comparable across frames. Thresholds in config/default.yaml must be tuned
    for this model; a gallery enrolled with another backend is not compatible.
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        det_size: int = 640,
        min_det_score: float = 0.5,
    ) -> None:
        # Imported here so the rest of the repo (and the tests) run without them.
        import onnxruntime as ort
        from insightface.app import FaceAnalysis

        available = ort.get_available_providers()
        use_gpu = "CUDAExecutionProvider" in available
        providers = (["CUDAExecutionProvider"] if use_gpu else []) + ["CPUExecutionProvider"]
        self.app = FaceAnalysis(
            name=model_name,
            allowed_modules=["detection", "recognition"],  # skip age/gender/landmarks
            providers=providers,
        )
        self.app.prepare(
            ctx_id=0 if use_gpu else -1,
            det_thresh=min_det_score,
            det_size=(det_size, det_size),
        )

    def detect(self, frame: np.ndarray) -> List[Detection]:
        detections = []
        for face in self.app.get(frame):
            x1, y1, x2, y2 = (int(v) for v in face.bbox)
            detections.append(
                Detection(bbox=(x1, y1, x2, y2), embedding=face.normed_embedding)
            )
        return detections


BACKENDS: Dict[str, Callable[[], Identifier]] = {
    "dummy": DummyIdentifier,
    "sface": SFaceIdentifier,
    "arcface_mbf": ArcFaceMBFIdentifier,
    "arcface_r50": ArcFaceR50Identifier,
    "insightface": InsightFaceIdentifier,
}

# Backends offered in the web view's model selector and on the /compare page, in a fixed order.
COMPARED = {
    "sface": {"label": "SFace", "dim": 128, "files": [YUNET_FILE, SFACE_FILE],
              "about": "OpenCV model zoo, MobileFaceNet-style, trained on CASIA-WebFace. Apache-2.0."},
    "arcface_mbf": {"label": "ArcFace MobileFaceNet", "dim": 512, "files": [YUNET_FILE, ARCFACE_MBF_FILE],
                    "about": "InsightFace buffalo_s, trained on WebFace600K. Non-commercial research only."},
    "arcface_r50": {"label": "ArcFace ResNet-50", "dim": 512, "files": [YUNET_FILE, ARCFACE_R50_FILE],
                    "about": "InsightFace buffalo_l, trained on WebFace600K. Non-commercial research only."},
}


def build_identifier(name: str) -> Identifier:
    if name not in BACKENDS:
        raise ValueError(f"Unknown identifier '{name}'. Available: {sorted(BACKENDS)}")
    return BACKENDS[name]()
