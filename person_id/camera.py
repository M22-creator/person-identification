"""Frame sources. All yield HxWx3 BGR uint8 frames."""
from __future__ import annotations

from typing import Iterator

import numpy as np


def opencv_frames(source: str, width: int, height: int, fps: int) -> Iterator[np.ndarray]:
    """source: device index ("0") or a video file path."""
    import cv2

    is_device = source.isdigit()
    cap = cv2.VideoCapture(int(source) if is_device else source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source '{source}'")
    if is_device:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def realsense_frames(width: int, height: int, fps: int) -> Iterator[np.ndarray]:
    """RGB stream of a RealSense camera (e.g. D435i). UNTESTED on hardware."""
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    pipeline.start(config)
    try:
        while True:
            color = pipeline.wait_for_frames().get_color_frame()
            if color:
                yield np.asanyarray(color.get_data())
    finally:
        pipeline.stop()


def open_source(source: str, width: int = 640, height: int = 480, fps: int = 30) -> Iterator[np.ndarray]:
    if source == "realsense":
        return realsense_frames(width, height, fps)
    return opencv_frames(source, width, height, fps)
