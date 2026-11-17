"""Fetch model weights into models/ (git-ignored), checked against pinned SHA-256s.

    python -m person_id.download_models                     # sface (YuNet + SFace, ~39 MB)
    python -m person_id.download_models --model arcface_r50 # + ArcFace ResNet-50 (~289 MB download)
    python -m person_id.download_models --model all         # every compared backend

Sources and licenses:
  YuNet, SFace   OpenCV model zoo (https://github.com/opencv/opencv_zoo). MIT / Apache-2.0.
  ArcFace        InsightFace model packs buffalo_l / buffalo_s
                 (https://github.com/deepinsight/insightface). Non-commercial research only.
                 Only the recognition model is kept; the rest of the zip is discarded.
"""
from __future__ import annotations

import argparse
import hashlib
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional

from .identifier import ARCFACE_MBF_FILE, ARCFACE_R50_FILE, COMPARED, MODEL_DIR, SFACE_FILE, YUNET_FILE

ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"
INSIGHTFACE = "https://github.com/deepinsight/insightface/releases/download/v0.7"


class Weight(NamedTuple):
    url: str
    sha256: str                        # of the final file in models/
    member: Optional[str] = None       # file inside the zip at url
    zip_sha256: Optional[str] = None


WEIGHTS: Dict[str, Weight] = {
    YUNET_FILE: Weight(
        f"{ZOO}/face_detection_yunet/{YUNET_FILE}",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    ),
    SFACE_FILE: Weight(
        f"{ZOO}/face_recognition_sface/{SFACE_FILE}",
        "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    ),
    ARCFACE_R50_FILE: Weight(
        f"{INSIGHTFACE}/buffalo_l.zip",
        "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
        member="w600k_r50.onnx",
        zip_sha256="80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f",
    ),
    ARCFACE_MBF_FILE: Weight(
        f"{INSIGHTFACE}/buffalo_s.zip",
        "9cc6e4a75f0e2bf0b1aed94578f144d15175f357bdc05e815e5c4a02b319eb4f",
        member="w600k_mbf.onnx",
        zip_sha256="d85a87f503f691807cd8bb97128bdf7a0660326cd9cd02657127fa978bab8b5e",
    ),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def files_for(models: List[str]) -> List[str]:
    out: List[str] = []
    for m in models:
        for f in COMPARED[m]["files"]:
            if f not in out:
                out.append(f)
    return out


def fetch(name: str, w: Weight, d: Path) -> None:
    path = d / name
    if path.exists() and sha256(path) == w.sha256:
        print(f"ok       {path}")
        return
    print(f"download {w.url}", flush=True)
    tmp = d / (name + ".part")
    urllib.request.urlretrieve(w.url, tmp)
    try:
        if w.member:
            if sha256(tmp) != w.zip_sha256:
                raise SystemExit(f"Checksum mismatch for {w.url}; file discarded.")
            with zipfile.ZipFile(tmp) as z, z.open(w.member) as src, open(path.with_suffix(".tmp"), "wb") as dst:
                while chunk := src.read(1 << 20):
                    dst.write(chunk)
            tmp.unlink()
            tmp = path.with_suffix(".tmp")
        if sha256(tmp) != w.sha256:
            raise SystemExit(f"Checksum mismatch for {name}; file discarded.")
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()
    print(f"saved    {path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default=str(MODEL_DIR))
    p.add_argument("--model", default="sface", choices=sorted(COMPARED) + ["all"])
    a = p.parse_args()
    d = Path(a.dir)
    d.mkdir(parents=True, exist_ok=True)
    models = list(COMPARED) if a.model == "all" else [a.model]
    for name in files_for(models):
        fetch(name, WEIGHTS[name], d)


if __name__ == "__main__":
    main()
