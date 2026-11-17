"""Automatic enrollment of new faces, used by the live web view.

A face the gallery does not know becomes a *candidate*. The candidate collects
samples while the same face keeps showing up; once it has enough, it is saved
to the gallery under the next free ID ("person_1", "person_2", ...).

Only use this where everyone in front of the camera has agreed to be stored.
person_id.run never enrolls anyone.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..gallery import Gallery
from ..types import Detection, IdentityResult, Status


def _unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x) + 1e-12)


@dataclass
class Candidate:
    key: int
    embeddings: List[np.ndarray] = field(default_factory=list)
    last_sample: float = 0.0
    last_seen: float = 0.0

    def centroid(self) -> np.ndarray:
        return _unit(np.mean(np.stack(self.embeddings), axis=0))

    def similarity(self, embedding: np.ndarray) -> float:
        # Against the average of the samples so far, not the closest single sample:
        # matching the closest sample lets one odd sample chain two people together.
        return float(self.centroid() @ embedding)


@dataclass
class Progress:
    """What happened to one detection in this frame."""
    key: int                         # candidate key, stable while it is being learned
    taken: int
    target: int
    enrolled_as: Optional[str] = None  # set on the frame the candidate is saved


class AutoEnroller:
    def __init__(
        self,
        gallery: Gallery,
        gallery_path: str,
        same_threshold: float = 0.5,   # cosine score for "same face as this candidate"
        samples: int = 8,
        sample_gap_s: float = 0.25,    # spacing, so samples are not near-identical frames
        forget_after_s: float = 5.0,   # drop a candidate that has not been seen for this long
        min_face_px: int = 48,         # ignore faces smaller than this: too blurry to store
        prefix: str = "person_",
    ) -> None:
        self.gallery = gallery
        self.gallery_path = gallery_path
        self.same_threshold = same_threshold
        self.samples = samples
        self.sample_gap_s = sample_gap_s
        self.forget_after_s = forget_after_s
        self.min_face_px = min_face_px
        self.prefix = prefix
        self.candidates: List[Candidate] = []
        self._next_key = 1
        # Per detection of the last update(): why that face is not being learned (or None).
        self.last_blocked: List[Optional[str]] = []

    def _next_id(self) -> str:
        pattern = re.compile(re.escape(self.prefix) + r"(\d+)$")
        used = [int(m.group(1)) for m in map(pattern.match, self.gallery.person_ids) if m]
        return f"{self.prefix}{max(used, default=0) + 1}"

    def update(
        self, detections: List[Detection], results: List[IdentityResult], now: float
    ) -> List[Optional[Progress]]:
        """detections[i] and results[i] describe the same face. Returns one entry
        per detection: None unless that face is a candidate being learned.

        Only "unknown" faces are learned. "uncertain" ones are left alone: they
        may be an enrolled person at a bad angle, and learning them would give
        one person two IDs. Before a candidate is saved its samples must agree
        with each other, and it must not match anyone already enrolled.
        """
        out: List[Optional[Progress]] = [None] * len(detections)
        self.last_blocked = [None] * len(detections)
        if len(detections) != len(results):  # the single no_person result
            detections = []
        taken_keys = set()
        for i, (det, res) in enumerate(zip(detections, results)):
            if res.status is Status.RECOGNIZED:
                continue
            if res.status is Status.UNCERTAIN:
                pid, score = self.gallery.match(det.embedding)
                self.last_blocked[i] = f"similar to {pid} ({score:.2f}): not sure this is a new person"
                continue
            x1, y1, x2, y2 = det.bbox
            size = min(x2 - x1, y2 - y1)
            if size < self.min_face_px:
                self.last_blocked[i] = f"face too small to learn ({size} px, needs {self.min_face_px}): come closer"
                continue
            emb = _unit(det.embedding)
            cand = self._match(emb, taken_keys)
            if cand is None:
                cand = Candidate(key=self._next_key)
                self._next_key += 1
                self.candidates.append(cand)
            taken_keys.add(cand.key)
            cand.last_seen = now
            if not cand.embeddings or now - cand.last_sample >= self.sample_gap_s:
                cand.embeddings.append(emb)
                cand.last_sample = now
            if len(cand.embeddings) >= self.samples:
                verdict = self._check(cand)
                if verdict is None:
                    out[i] = Progress(cand.key, len(cand.embeddings), self.samples, self._save(cand))
                    continue
                self.last_blocked[i] = verdict
                if cand not in self.candidates:  # discarded as a duplicate
                    continue
            out[i] = Progress(cand.key, len(cand.embeddings), self.samples)

        self.candidates = [c for c in self.candidates if now - c.last_seen <= self.forget_after_s]
        return out

    def _check(self, cand: Candidate) -> Optional[str]:
        """None if the candidate can be saved; otherwise why not (and fix it up)."""
        c = cand.centroid()
        sims = np.stack(cand.embeddings) @ c
        keep = [e for e, s in zip(cand.embeddings, sims) if s >= self.same_threshold]
        if len(keep) < len(cand.embeddings):
            dropped = len(cand.embeddings) - len(keep)
            cand.embeddings = keep  # keep learning; the outliers were probably someone else or a blur
            return f"dropped {dropped} sample(s) that did not match the rest; collecting more"
        pid, score = self.gallery.match(c)
        if pid is not None and score >= self.same_threshold:
            self.candidates.remove(cand)
            return f"already enrolled as {pid} ({score:.2f}); not saved again"
        return None

    def _match(self, emb: np.ndarray, taken_keys: set) -> Optional[Candidate]:
        best, best_score = None, self.same_threshold
        for c in self.candidates:
            if c.key in taken_keys:  # two faces in one frame are two people
                continue
            score = c.similarity(emb)
            if score >= best_score:
                best, best_score = c, score
        return best

    def _save(self, cand: Candidate) -> str:
        person_id = self._next_id()
        for e in cand.embeddings:
            self.gallery.add(person_id, e)
        self.gallery.save(self.gallery_path)
        self.candidates.remove(cand)
        return person_id
