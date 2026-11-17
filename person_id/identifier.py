"""Identifier backends: frame in, list of Detection out.

To add a model: subclass Identifier, implement detect(), and register it
in BACKENDS. "sface" is the default backend, "insightface" is optional,
"dummy" finds nobody. Nothing else in the repo depends on which face model
you use.
"""
from __future__ import annotations

import os
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


class SFaceIdentifier(Identifier):
    """OpenCV model zoo: YuNet face detector + SFace 128-D face embedding.

    Needs only opencv-python (>= 4.8). Weights live in models/ (git-ignored);
    fetch them with `python -m person_id.download_models`.
    """

    def __init__(
        self,
        model_dir: Optional[os.PathLike] = None,
        score_threshold: float = 0.8,
        min_face: int = 40,
    ) -> None:
        import cv2

        d = Path(model_dir or MODEL_DIR)
        det_path, rec_path = d / YUNET_FILE, d / SFACE_FILE
        missing = [p.name for p in (det_path, rec_path) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing model file(s) {missing} in {d}. Run: python -m person_id.download_models"
            )
        self._cv2 = cv2
        self.min_face = min_face
        self.detector = cv2.FaceDetectorYN.create(str(det_path), "", (320, 320), score_threshold, 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(str(rec_path), "")
        self._size: Tuple[int, int] = (0, 0)

    def faces(self, frame: np.ndarray) -> np.ndarray:
        """Raw YuNet rows: x, y, w, h, 5 landmarks (x, y), score. Shape (N, 15)."""
        h, w = frame.shape[:2]
        if (w, h) != self._size:
            self.detector.setInputSize((w, h))
            self._size = (w, h)
        _, faces = self.detector.detect(frame)
        return np.zeros((0, 15), np.float32) if faces is None else faces

    def embed(self, frame: np.ndarray, face: np.ndarray) -> np.ndarray:
        aligned = self.recognizer.alignCrop(frame, face)
        return self.recognizer.feature(aligned).reshape(-1).astype(np.float32)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        h, w = frame.shape[:2]
        out = []
        for f in self.faces(frame):
            x, y, fw, fh = f[:4]
            if min(fw, fh) < self.min_face:
                continue  # too small to identify reliably
            bbox = (
                int(max(0, x)), int(max(0, y)),
                int(min(w, x + fw)), int(min(h, y + fh)),
            )
            out.append(Detection(bbox=bbox, embedding=self.embed(frame, f)))
        return out


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
    "insightface": InsightFaceIdentifier,
}


def build_identifier(name: str) -> Identifier:
    if name not in BACKENDS:
        raise ValueError(f"Unknown identifier '{name}'. Available: {sorted(BACKENDS)}")
    return BACKENDS[name]()
