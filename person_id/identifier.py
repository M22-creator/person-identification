"""Identifier backends: frame in, list of Detection out.

THIS IS WHERE YOUR MODEL GOES. Subclass Identifier, implement detect(),
and register it in BACKENDS. Nothing else in the repo depends on which
face model you use.
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


# TODO(Hongxuan): add your backend, e.g.
#
# class MyIdentifier(Identifier):
#     def __init__(self):
#         self.model = ...            # load weights once
#     def detect(self, frame):
#         faces = self.model(frame)
#         return [Detection(bbox=tuple(f.bbox), embedding=f.embedding) for f in faces]

BACKENDS: Dict[str, Callable[[], Identifier]] = {
    "dummy": DummyIdentifier,
    # "mine": MyIdentifier,
}


def build_identifier(name: str) -> Identifier:
    if name not in BACKENDS:
        raise ValueError(f"Unknown identifier '{name}'. Available: {sorted(BACKENDS)}")
    return BACKENDS[name]()
