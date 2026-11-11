from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

import numpy as np


class Status(str, Enum):
    RECOGNIZED = "recognized"  # matched an enrolled person
    UNKNOWN = "unknown"        # a person, but not enrolled
    UNCERTAIN = "uncertain"    # a person, match too weak to trust
    NO_PERSON = "no_person"    # nobody in frame


@dataclass
class Detection:
    """One person found in a frame by the identifier backend."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2 in pixels
    embedding: np.ndarray            # 1-D feature vector


@dataclass
class IdentityResult:
    """What other AVA modules receive."""
    status: Status
    person_id: Optional[str] = None
    confidence: float = 0.0
    bbox: Optional[Tuple[int, int, int, int]] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "person_id": self.person_id,
            "confidence": round(float(self.confidence), 4),
            "bbox": list(self.bbox) if self.bbox else None,
            "timestamp": self.timestamp,
        }
