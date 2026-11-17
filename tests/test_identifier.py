"""Tests for the sface backend.

Model tests are skipped when the weights are absent (python -m person_id.download_models).
The LFW test runs only if PERSON_ID_LFW points at an LFW folder (pairs.txt + lfw/);
face images are never stored in this repo.
"""
import os
from pathlib import Path

import numpy as np
import pytest

from person_id import Gallery, Pipeline, Status
from person_id.config import load_config
from person_id.identifier import (
    BACKENDS, MODEL_DIR, SFACE_FILE, YUNET_FILE, SFaceIdentifier, build_identifier,
)

cv2 = pytest.importorskip("cv2")
HAVE_MODELS = (MODEL_DIR / YUNET_FILE).exists() and (MODEL_DIR / SFACE_FILE).exists()
needs_models = pytest.mark.skipif(not HAVE_MODELS, reason="run python -m person_id.download_models")
LFW = Path(os.environ["PERSON_ID_LFW"]) if os.environ.get("PERSON_ID_LFW") else None


def test_registered_and_default():
    assert "sface" in BACKENDS
    assert load_config()["identifier"] == "sface"


def test_missing_weights_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_models"):
        SFaceIdentifier(model_dir=tmp_path)


@pytest.fixture(scope="module")
def ident():
    return build_identifier("sface")


@needs_models
@pytest.mark.parametrize("shape", [(480, 640, 3), (240, 320, 3), (720, 1280, 3)])
def test_blank_frames_find_nobody(ident, shape):
    # Also exercises re-sizing the detector between frame sizes.
    assert ident.detect(np.zeros(shape, np.uint8)) == []
    assert ident.detect(np.full(shape, 255, np.uint8)) == []


@needs_models
def test_noise_frame_pipeline_no_person(ident):
    frame = np.random.default_rng(0).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    r = Pipeline(ident, Gallery(), 0.40, 0.30).process(frame)
    assert [x.status for x in r] == [Status.NO_PERSON]


def _lfw(name, i):
    img = cv2.imread(str(LFW / "lfw" / name / f"{name}_{i:04d}.jpg"))
    assert img is not None
    return img


@needs_models
@pytest.mark.skipif(LFW is None, reason="set PERSON_ID_LFW to an LFW folder")
def test_lfw_end_to_end(ident):
    cfg = load_config()
    g = Gallery()
    for d in ident.detect(_lfw("George_W_Bush", 1)):
        g.add("bush", d.embedding)
    assert len(g) == 1

    dets = ident.detect(_lfw("George_W_Bush", 2))
    assert dets and dets[0].embedding.shape == (128,)
    x1, y1, x2, y2 = dets[0].bbox
    assert 0 <= x1 < x2 <= 250 and 0 <= y1 < y2 <= 250

    p = Pipeline(ident, g, cfg["recognize_threshold"], cfg["unknown_threshold"])
    same = p.process(_lfw("George_W_Bush", 2))
    assert any(r.status == Status.RECOGNIZED and r.person_id == "bush" for r in same)
    other = p.process(_lfw("Colin_Powell", 1))
    assert all(r.person_id is None for r in other)
