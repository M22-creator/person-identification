"""Enrolled people: person_id -> embeddings, with cosine-similarity matching."""
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


class Gallery:
    def __init__(self) -> None:
        self._people: Dict[str, np.ndarray] = {}  # id -> (N, D) normalized

    def __len__(self) -> int:
        return len(self._people)

    @property
    def person_ids(self):
        return sorted(self._people)

    def add(self, person_id: str, embedding: np.ndarray) -> None:
        e = _normalize(embedding).reshape(1, -1)
        if person_id in self._people:
            self._people[person_id] = np.vstack([self._people[person_id], e])
        else:
            self._people[person_id] = e

    def remove(self, person_id: str) -> None:
        self._people.pop(person_id, None)

    def match(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """Best (person_id, cosine score). (None, 0.0) if the gallery is empty."""
        if not self._people:
            return None, 0.0
        q = _normalize(embedding)
        best_id, best = None, -1.0
        for pid, embs in self._people.items():
            score = float(np.max(embs @ q))
            if score > best:
                best_id, best = pid, score
        return best_id, best

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez(path, **self._people)

    @classmethod
    def load(cls, path: str) -> "Gallery":
        g = cls()
        if os.path.exists(path):
            with np.load(path) as data:
                g._people = {k: data[k] for k in data.files}
        return g
