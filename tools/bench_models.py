"""Compare face recognition backends: LFW accuracy + latency on this machine.

    python -m person_id.download_models --model all
    python tools/bench_models.py --lfw data/bench/lfw --out data/bench/results.json

--lfw must contain pairs.txt and the extracted lfw/ folder (see tools/bench_lfw.py
for download links). data/ is git-ignored; never commit LFW or the results' source images.

All compared backends share the YuNet detector, so each LFW image is detected
once and the same face crop goes to every recognition model: only the
recognizer differs. The web view shows the results at /compare.

Accuracy: standard LFW verification, 6000 pairs, 10 folds (threshold picked on 9
folds, tested on the 10th). A pair with a missed face counts as a non-match.
Latency: --frames camera-sized frames (an LFW face on a 640x480 canvas) per model,
after a warm-up; detection and embedding are timed separately. CPU only.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import socket
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from bench_lfw import best_threshold, center_face, read_pairs, threshold_at_far  # noqa: E402
from person_id.config import load_config, model_settings  # noqa: E402
from person_id.identifier import COMPARED, MODEL_DIR, build_identifier  # noqa: E402

ROC_FARS = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1, 0.2, 0.5, 1.0]


def rss_mb() -> float:
    """Current resident memory in MB (Linux), else peak RSS (macOS reports bytes)."""
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024 if sys.platform == "darwin" else 1024)


def pct(values, q) -> float:
    return round(float(np.percentile(values, q)), 2) if len(values) else float("nan")


def suggest(gen: np.ndarray, imp: np.ndarray, scores, labels) -> dict:
    """The rule used for config/default.yaml (same for every backend)."""
    grid = np.round(np.arange(0.0, 1.0001, 0.05), 2)
    rec = next((float(t) for t in grid if np.mean(imp >= t) <= 0.0005), 1.0)
    best_t, _ = best_threshold(scores, labels)
    unk = float(np.floor(best_t / 0.05) * 0.05)
    return {"recognize_threshold": round(rec, 2), "unknown_threshold": round(min(unk, rec), 2)}


def accuracy_report(scores, labels, fold, nfolds, settings) -> dict:
    accs = []
    for k in range(nfolds):
        tr, te = fold != k, fold == k
        t, _ = best_threshold(scores[tr], labels[tr])
        accs.append(np.mean((scores[te] >= t) == labels[te]))
    gen, imp = scores[labels], scores[~labels]
    roc = []
    for far in ROC_FARS:
        t = threshold_at_far(imp, far) if far < 1 else float(imp.min())
        roc.append({"far": far, "tar": round(float(np.mean(gen >= t)), 4), "threshold": round(t, 4)})
    rec, unk = settings["recognize_threshold"], settings["unknown_threshold"]
    return {
        "accuracy_10fold_mean": round(float(np.mean(accs)), 4),
        "accuracy_10fold_std": round(float(np.std(accs)), 4),
        "tar_at_far_1e-2": next(r["tar"] for r in roc if r["far"] == 1e-2),
        "tar_at_far_1e-3": next(r["tar"] for r in roc if r["far"] == 1e-3),
        "roc": roc,
        "genuine_score_pct": {str(q): pct(gen, q) for q in (1, 5, 50)},
        "impostor_score_pct": {str(q): pct(imp, q) for q in (50, 99, 99.9)},
        "configured": {
            "recognize_threshold": rec,
            "unknown_threshold": unk,
            # Per pair: how a single comparison would be classified at these thresholds.
            "same_recognized": round(float(np.mean(gen >= rec)), 4),
            "same_uncertain": round(float(np.mean((gen >= unk) & (gen < rec))), 4),
            "same_unknown": round(float(np.mean(gen < unk)), 4),
            "diff_recognized": round(float(np.mean(imp >= rec)), 5),
            "diff_uncertain": round(float(np.mean((imp >= unk) & (imp < rec))), 4),
        },
        "suggested": suggest(gen, imp, scores, labels),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lfw", required=True, type=Path)
    p.add_argument("--out", type=Path, default=ROOT / "data/bench/results.json")
    p.add_argument("--models", default=",".join(COMPARED))
    p.add_argument("--frames", type=int, default=200, help="latency frames per model")
    p.add_argument("--config", default=str(ROOT / "config/default.yaml"))
    a = p.parse_args()

    import cv2

    names = [m for m in a.models.split(",") if m]
    cfg = load_config(a.config)
    pairs, nfolds = read_pairs(a.lfw / "pairs.txt")
    started = time.time()

    # Pay the shared start-up costs (OpenCV DNN, onnxruntime, the YuNet detector)
    # before measuring, so they are not charged to whichever model loads first.
    import onnxruntime  # noqa: F401
    build_identifier(names[0]).detect(np.zeros((480, 640, 3), np.uint8))

    models, load = {}, {}
    for name in names:
        before, t0 = rss_mb(), time.perf_counter()
        models[name] = build_identifier(name)
        load[name] = {"load_s": round(time.perf_counter() - t0, 2), "rss_added_mb": round(rss_mb() - before, 1)}
        print(f"loaded {name} in {load[name]['load_s']} s", flush=True)
    det = models[names[0]]  # every compared backend uses the same YuNet settings

    # ---- accuracy: detect each LFW image once, embed with every model ----
    keys = sorted({k for x, y, _, _ in pairs for k in (x, y)})
    emb = {n: {} for n in names}
    crop_ms = {n: [] for n in names}
    missed = 0
    for i, (person, idx) in enumerate(keys):
        img = cv2.imread(str(a.lfw / "lfw" / person / f"{person}_{idx:04d}.jpg"))
        if img is None:
            raise FileNotFoundError(f"{person}_{idx:04d}.jpg")
        f = center_face(det.faces(img), img.shape[1], img.shape[0])
        if f is None:
            missed += 1
        for n in names:
            if f is None:
                emb[n][(person, idx)] = None
                continue
            t0 = time.perf_counter()
            v = models[n].embed(img, f)
            crop_ms[n].append((time.perf_counter() - t0) * 1000)
            emb[n][(person, idx)] = v / (np.linalg.norm(v) + 1e-12)
        if (i + 1) % 1000 == 0:
            print(f"  embedded {i + 1}/{len(keys)} images", flush=True)

    labels = np.array([same for _, _, same, _ in pairs])
    fold = np.array([k for _, _, _, k in pairs])
    results = {}
    for n in names:
        scores = np.array([
            -1.0 if emb[n][x] is None or emb[n][y] is None else float(emb[n][x] @ emb[n][y])
            for x, y, _, _ in pairs
        ])
        results[n] = {
            "label": COMPARED[n]["label"],
            "about": COMPARED[n]["about"],
            "dim": COMPARED[n]["dim"],
            "weights_mb": round(sum((MODEL_DIR / f).stat().st_size for f in COMPARED[n]["files"][1:]) / 1e6, 1),
            **load[n],
            "lfw": accuracy_report(scores, labels, fold, nfolds, model_settings(cfg, n)),
            "embed_ms_lfw": {"mean": round(float(np.mean(crop_ms[n])), 2), "p50": pct(crop_ms[n], 50), "p95": pct(crop_ms[n], 95)},
        }
        print(f"{n}: LFW 10-fold accuracy {results[n]['lfw']['accuracy_10fold_mean']:.4f}", flush=True)

    # ---- latency on camera-sized frames ----
    faces = [k for k in keys if emb[names[0]][k] is not None][: a.frames]
    frames = []
    for person, idx in faces:
        c = np.full((480, 640, 3), 40, np.uint8)
        c[115:365, 195:445] = cv2.imread(str(a.lfw / "lfw" / person / f"{person}_{idx:04d}.jpg"))
        frames.append(c)
    for n in names:
        m = models[n]
        for f in frames[:10]:
            m.detect(f)  # warm-up
        dms, ems, tot = [], [], []
        for f in frames:
            t0 = time.perf_counter()
            m.detect(f)
            tot.append((time.perf_counter() - t0) * 1000)
            dms.append(m.last_timing["detect_ms"])
            if m.last_timing["faces"]:
                ems.append(m.last_timing["embed_ms"] / m.last_timing["faces"])
        results[n]["latency_640x480"] = {
            "frames": len(frames),
            "detect_ms": {"p50": pct(dms, 50), "p95": pct(dms, 95)},
            "embed_ms_per_face": {"p50": pct(ems, 50), "p95": pct(ems, 95)},
            "frame_ms": {"p50": pct(tot, 50), "p95": pct(tot, 95), "mean": round(float(np.mean(tot)), 2)},
            "max_fps_one_face": round(1000.0 / float(np.median(tot)), 1),
        }
        print(f"{n}: frame p50 {results[n]['latency_640x480']['frame_ms']['p50']} ms", flush=True)

    import onnxruntime

    out = {
        "meta": {
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpus": os.cpu_count(),
            "load_avg_at_start": [round(x, 2) for x in os.getloadavg()],
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "onnxruntime": onnxruntime.__version__,
            "execution": "CPU (OpenCV DNN for YuNet/SFace, onnxruntime CPUExecutionProvider for ArcFace)",
            "started": time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(started)),
            "seconds": round(time.time() - started, 1),
            "lfw_pairs": len(pairs),
            "lfw_images": len(keys),
            "lfw_images_without_face": missed,
            "detector": "YuNet 2023mar, score >= 0.8, faces < 40 px skipped (shared by all models)",
            "order": names,
        },
        "models": results,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
