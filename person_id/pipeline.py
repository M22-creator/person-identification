"""Frame -> IdentityResult list. Holds the recognized/unknown/uncertain logic."""
from __future__ import annotations

from typing import List

import numpy as np

from .gallery import Gallery
from .identifier import Identifier
from .types import IdentityResult, Status


class Pipeline:
    def __init__(
        self,
        identifier: Identifier,
        gallery: Gallery,
        recognize_threshold: float = 0.5,
        unknown_threshold: float = 0.3,
    ) -> None:
        if unknown_threshold > recognize_threshold:
            raise ValueError("unknown_threshold must be <= recognize_threshold")
        self.identifier = identifier
        self.gallery = gallery
        self.recognize_threshold = recognize_threshold
        self.unknown_threshold = unknown_threshold

    def process(self, frame: np.ndarray) -> List[IdentityResult]:
        """Always returns at least one result (no_person if nobody is found)."""
        detections = self.identifier.detect(frame)
        if not detections:
            return [IdentityResult(status=Status.NO_PERSON)]

        results = []
        for det in detections:
            pid, score = self.gallery.match(det.embedding)
            if pid is not None and score >= self.recognize_threshold:
                r = IdentityResult(Status.RECOGNIZED, pid, score, det.bbox)
            elif pid is None or score < self.unknown_threshold:
                # Unknown people are reported, never auto-enrolled.
                r = IdentityResult(Status.UNKNOWN, None, score, det.bbox)
            else:
                # Too weak to act on: downstream should not update preferences.
                r = IdentityResult(Status.UNCERTAIN, None, score, det.bbox)
            results.append(r)
        return results
