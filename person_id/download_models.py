"""Fetch the YuNet + SFace weights:  python -m person_id.download_models

Source: OpenCV model zoo (https://github.com/opencv/opencv_zoo).
Licenses: YuNet MIT, SFace Apache-2.0.
Files go to models/ (git-ignored) and are checked against pinned SHA-256s.
"""
from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

from .identifier import MODEL_DIR, SFACE_FILE, YUNET_FILE

ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"
MODELS = {
    YUNET_FILE: (
        f"{ZOO}/face_detection_yunet/{YUNET_FILE}",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    ),
    SFACE_FILE: (
        f"{ZOO}/face_recognition_sface/{SFACE_FILE}",
        "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    ),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", default=str(MODEL_DIR))
    a = p.parse_args()
    d = Path(a.dir)
    d.mkdir(parents=True, exist_ok=True)
    for name, (url, digest) in MODELS.items():
        path = d / name
        if path.exists() and sha256(path) == digest:
            print(f"ok       {path}")
            continue
        print(f"download {url}")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        if sha256(tmp) != digest:
            tmp.unlink()
            raise SystemExit(f"Checksum mismatch for {name}; file discarded.")
        tmp.replace(path)
        print(f"saved    {path}")


if __name__ == "__main__":
    main()
