"""Interaction history: when each enrolled person was seen, one entry per visit.

A visit starts when a person is recognized and ends once they have not been
recognized for `gap_s` seconds; coming back after that starts a new visit.
Only `recognized` results count (never `uncertain` or `unknown`), except the
frame on which the web view auto-enrolls someone, which opens their first visit.

Stored next to the gallery (data/gallery.npz -> data/gallery.visits.json),
git-ignored like it. Removing a person from the gallery removes their history.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional


class VisitLog:
    def __init__(self, path: os.PathLike, gap_s: float = 10.0, save_every_s: float = 5.0) -> None:
        self.path = Path(path)
        self.gap_s = gap_s
        self.save_every_s = save_every_s
        self.people: Dict[str, List[dict]] = {}
        self.open: Dict[str, dict] = {}   # person_id -> their current visit (also last in people[id])
        self._dirty = False
        self._saved_at = 0.0
        if self.path.exists():
            self.people = json.loads(self.path.read_text()).get("people", {})

    @staticmethod
    def path_for(gallery_path: os.PathLike) -> Path:
        p = Path(gallery_path)
        return p.with_name(p.stem + ".visits.json")

    def seen(self, scores: Dict[str, Optional[float]], now: float, enrolled: Iterable[str] = ()) -> None:
        """scores: person_id -> match score in this frame (None on the enrollment frame)."""
        enrolled = set(enrolled)
        changed = False
        for pid, score in scores.items():
            v = self.open.get(pid)
            if v is None or now - v["end"] > self.gap_s:
                v = {"start": now, "end": now, "frames": 0, "score_sum": 0.0, "scored": 0, "best": None,
                     "enrolled": pid in enrolled}
                self.people.setdefault(pid, []).append(v)
                self.open[pid] = v
                changed = True
            v["end"] = now
            v["frames"] += 1
            if score is not None:
                v["score_sum"] += score
                v["scored"] += 1
                v["best"] = score if v["best"] is None else max(v["best"], score)
            self._dirty = True
        for pid, v in list(self.open.items()):
            if now - v["end"] > self.gap_s:
                del self.open[pid]  # visit over
                changed = True
        if changed or (self._dirty and now - self._saved_at >= self.save_every_s):
            self.save(now)

    def remove(self, person_id: str) -> None:
        self.open.pop(person_id, None)
        if self.people.pop(person_id, None) is not None:
            self.save()

    def save(self, now: Optional[float] = None) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"gap_s": self.gap_s, "people": self.people}))
        tmp.replace(self.path)  # atomic: a crash never leaves a half-written file
        self._dirty = False
        self._saved_at = now if now is not None else time.time()

    @staticmethod
    def _public(v: dict, is_open: bool) -> dict:
        return {
            "start": round(v["start"], 3), "end": round(v["end"], 3),
            "duration_s": round(v["end"] - v["start"], 1),
            "frames": v["frames"],
            "mean_score": round(v["score_sum"] / v["scored"], 3) if v["scored"] else None,
            "best_score": round(v["best"], 3) if v["best"] is not None else None,
            "enrolled": v["enrolled"], "ongoing": is_open,
        }

    def summary(self, person_id: str) -> dict:
        visits = self.people.get(person_id, [])
        return {
            "visits": len(visits),
            "first_seen": round(visits[0]["start"], 3) if visits else None,
            "last_seen": round(visits[-1]["end"], 3) if visits else None,
            "in_view_now": person_id in self.open,
        }

    def history(self, person_id: str) -> dict:
        visits = self.people.get(person_id, [])
        cur = self.open.get(person_id)
        items = [self._public(v, v is cur) for v in visits]
        return {
            **self.summary(person_id),
            "total_in_view_s": round(sum(i["duration_s"] for i in items), 1),
            "history": list(reversed(items)),  # newest first
            "gap_s": self.gap_s,
        }
