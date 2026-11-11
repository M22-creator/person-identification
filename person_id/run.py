"""Live identification:  python -m person_id.run --source video.mp4"""
from __future__ import annotations

import argparse

from .camera import open_source
from .config import DEFAULT_PATH, load_config
from .gallery import Gallery
from .identifier import build_identifier
from .pipeline import Pipeline
from .publishers import build_publisher


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=DEFAULT_PATH)
    p.add_argument("--source", help='"realsense", a device index, or a video file')
    p.add_argument("--identifier")
    p.add_argument("--publisher", choices=["stdout", "ros2"])
    p.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = no limit)")
    a = p.parse_args()

    cfg = load_config(a.config, source=a.source, identifier=a.identifier, publisher=a.publisher)
    pipeline = Pipeline(
        build_identifier(cfg["identifier"]),
        Gallery.load(cfg["gallery_path"]),
        cfg["recognize_threshold"],
        cfg["unknown_threshold"],
    )
    publisher = build_publisher(cfg["publisher"], cfg["ros2_topic"])
    try:
        for i, frame in enumerate(open_source(cfg["source"], cfg["width"], cfg["height"], cfg["fps"]), 1):
            publisher.publish(pipeline.process(frame))
            if a.max_frames and i >= a.max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        publisher.close()


if __name__ == "__main__":
    main()
