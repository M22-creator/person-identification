"""Model selection, per-model settings, ArcFace alignment and the weight downloader."""
import hashlib
import zipfile
from pathlib import Path

import numpy as np
import pytest

from person_id import Gallery
from person_id.config import load_config, model_settings
from person_id.identifier import (
    ARCFACE_TEMPLATE, BACKENDS, COMPARED, MODEL_DIR, ArcFaceMBFIdentifier, ArcFaceR50Identifier,
    SFaceIdentifier, missing_weights, similarity_transform,
)

cv2 = pytest.importorskip("cv2")


def test_compared_backends_are_registered_in_order():
    assert list(COMPARED) == ["sface", "arcface_mbf", "arcface_r50"]
    assert set(COMPARED) <= set(BACKENDS)


def test_each_model_has_its_own_gallery_and_thresholds():
    cfg = load_config()
    paths = {model_settings(cfg, m)["gallery_path"] for m in COMPARED}
    assert len(paths) == 3
    assert model_settings(cfg, "sface")["gallery_path"] == "data/gallery.npz"  # existing gallery kept
    for m in COMPARED:
        s = model_settings(cfg, m)
        assert s["unknown_threshold"] <= s["recognize_threshold"]


def test_backend_without_entry_uses_top_level_fallback(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("identifier: sface\ngallery_path: g.npz\nrecognize_threshold: 0.6\nunknown_threshold: 0.2\n"
                 "models:\n  sface: {gallery_path: s.npz, recognize_threshold: 0.4, unknown_threshold: 0.3}\n")
    cfg = load_config(str(p))
    assert (cfg["gallery_path"], cfg["recognize_threshold"]) == ("s.npz", 0.4)
    assert model_settings(cfg, "dummy") == {"gallery_path": "g.npz", "recognize_threshold": 0.6, "unknown_threshold": 0.2}


def test_gallery_rejects_embeddings_of_another_model():
    g = Gallery()
    g.add("a", np.ones(128))
    with pytest.raises(ValueError, match="different identifier"):
        g.match(np.ones(512))
    with pytest.raises(ValueError, match="different identifier"):
        g.add("b", np.ones(512))


def test_similarity_transform_recovers_a_known_transform():
    a, s, t = 0.3, 1.7, np.array([12.0, -5.0])
    r = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    src = ARCFACE_TEMPLATE.astype(np.float64)
    dst = (s * src @ r.T) + t
    m = similarity_transform(src, dst)
    np.testing.assert_allclose(m[:, :2], s * r, atol=1e-4)
    np.testing.assert_allclose(m[:, 2], t, atol=1e-3)


def test_missing_arcface_weights_message(tmp_path):
    pytest.importorskip("onnxruntime")
    with pytest.raises(FileNotFoundError, match="--model arcface_r50"):
        ArcFaceR50Identifier(model_dir=tmp_path)


HAVE_ARCFACE = not missing_weights(COMPARED["arcface_mbf"]["files"] + COMPARED["arcface_r50"]["files"][1:])


@pytest.mark.skipif(not HAVE_ARCFACE, reason="python -m person_id.download_models --model all")
def test_arcface_alignment_matches_sface_alignment_and_shapes():
    pytest.importorskip("onnxruntime")
    sf, mbf = SFaceIdentifier(), ArcFaceMBFIdentifier()
    rng = np.random.default_rng(1)
    frame = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
    face = np.zeros(15, np.float32)
    face[:4] = [100, 60, 90, 110]
    face[4:14] = [125, 100, 165, 100, 145, 125, 128, 145, 162, 145]
    a = sf.recognizer.alignCrop(frame, face).astype(float)
    b = mbf.align(frame, face).astype(float)
    assert np.abs(a - b).mean() < 1.0  # same 112x112 ArcFace template as OpenCV
    assert mbf.embed(frame, face).shape == (512,)
    assert mbf.detect(np.zeros((480, 640, 3), np.uint8)) == []
    assert set(mbf.last_timing) == {"detect_ms", "embed_ms", "faces"}


def test_downloader_extracts_and_verifies_a_zip_member(tmp_path, monkeypatch):
    from person_id import download_models as dm

    payload = b"fake onnx bytes"
    zpath = tmp_path / "pack.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr("w600k_mbf.onnx", payload)
        z.writestr("other.onnx", b"ignored")
    w = dm.Weight(zpath.as_uri(), hashlib.sha256(payload).hexdigest(), "w600k_mbf.onnx", dm.sha256(zpath))
    out = tmp_path / "models"
    out.mkdir()
    dm.fetch("arcface_w600k_mbf.onnx", w, out)
    assert (out / "arcface_w600k_mbf.onnx").read_bytes() == payload
    assert sorted(p.name for p in out.iterdir()) == ["arcface_w600k_mbf.onnx"]  # nothing else kept
    bad = w._replace(zip_sha256="0" * 64)
    (out / "arcface_w600k_mbf.onnx").unlink()
    with pytest.raises(SystemExit, match="Checksum mismatch"):
        dm.fetch("arcface_w600k_mbf.onnx", bad, out)
    assert list(out.iterdir()) == []
