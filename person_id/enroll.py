"""Enroll a person:  python -m person_id.enroll --id alice --source 0 --samples 10

Only enroll people who have agreed to it. Embeddings are stored in the
git-ignored gallery file; remove someone with --remove.
"""
from __future__ import annotations

import argparse
import sys

from .camera import open_source
from .config import DEFAULT_PATH, load_config
from .gallery import Gallery
from .identifier import build_identifier


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=DEFAULT_PATH)
    p.add_argument("--id", required=True, help="person_id to enroll or remove")
    p.add_argument("--source")
    p.add_argument("--identifier")
    p.add_argument("--samples", type=int, default=10)
    p.add_argument("--remove", action="store_true")
    a = p.parse_args()

    cfg = load_config(a.config, source=a.source, identifier=a.identifier)
    gallery = Gallery.load(cfg["gallery_path"])

    if a.remove:
        gallery.remove(a.id)
        gallery.save(cfg["gallery_path"])
        print(f"Removed '{a.id}'.")
        return

    identifier = build_identifier(cfg["identifier"])
    taken = 0
    for frame in open_source(cfg["source"], cfg["width"], cfg["height"], cfg["fps"]):
        detections = identifier.detect(frame)
        if len(detections) != 1:
            continue  # need exactly one person in view to know whose face it is
        gallery.add(a.id, detections[0].embedding)
        taken += 1
        if taken >= a.samples:
            break

    if taken == 0:
        sys.exit("No usable frames (need exactly one person in view). Nothing saved.")
    gallery.save(cfg["gallery_path"])
    print(f"Enrolled '{a.id}' with {taken} sample(s).")


if __name__ == "__main__":
    main()
