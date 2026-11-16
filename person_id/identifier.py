"""Identifier backends: frame in, list of Detection out.

To add a model: subclass Identifier, implement detect(), and register it
in BACKENDS. "insightface" is the real backend; "dummy" finds nobody.
Nothing else in the repo depends on which face model you use.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Dict, List

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
    "insightface": InsightFaceIdentifier,
}


def build_identifier(name: str) -> Identifier:
    if name not in BACKENDS:
        raise ValueError(f"Unknown identifier '{name}'. Available: {sorted(BACKENDS)}")
    return BACKENDS[name]()
