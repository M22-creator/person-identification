"""Benchmark an identifier on LFW face verification (6000 pairs, 10 folds).

    python -m person_id.download_models
    python tools/bench_lfw.py --lfw ~/.cache/person_id_bench/lfw

--lfw must contain pairs.txt and the extracted lfw/ image folder, from
http://vis-www.cs.umass.edu/lfw/ (mirror: https://ndownloader.figshare.com/files/5976018
for lfw.tgz, .../5976006 for pairs.txt). Keep the dataset OUTSIDE this repo:
it is face images of real people and must never be committed.

Prints 10-fold accuracy, TAR at fixed FAR, and suggested values for
recognize_threshold / unknown_threshold.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from person_id.identifier import SFaceIdentifier  # noqa: E402


def read_pairs(path: Path):
    lines = path.read_text().strip().splitlines()
    folds, per = map(int, lines[0].split())
    pairs = []
    for i, line in enumerate(lines[1:]):
        p = line.split()
        if len(p) == 3:
            a, b, same = (p[0], int(p[1])), (p[0], int(p[2])), True
        else:
            a, b, same = (p[0], int(p[1])), (p[2], int(p[3])), False
        pairs.append((a, b, same, i // (2 * per)))
    assert len(pairs) == folds * 2 * per, "unexpected pairs.txt layout"
    return pairs, folds


def center_face(faces: np.ndarray, w: int, h: int):
    """LFW images are centred on the subject: pick the face nearest the centre."""
    if len(faces) == 0:
        return None
    c = faces[:, :2] + faces[:, 2:4] / 2
    return faces[np.argmin(np.sum((c - [w / 2, h / 2]) ** 2, axis=1))]


def threshold_at_far(impostor: np.ndarray, far: float) -> float:
    s = np.sort(impostor)[::-1]
    k = int(np.floor(far * len(s)))
    return float(s[k]) + 1e-6  # strictly above the (k+1)-th highest impostor score


def best_threshold(scores, labels):
    cand = np.unique(scores)
    acc = [np.mean((scores >= t) == labels) for t in cand]
    i = int(np.argmax(acc))
    return float(cand[i]), float(acc[i])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lfw", required=True, type=Path)
    p.add_argument("--out", type=Path, help="optional .npz to save per-pair scores (outside the repo)")
    a = p.parse_args()

    import cv2

    pairs, nfolds = read_pairs(a.lfw / "pairs.txt")
    ident = SFaceIdentifier()
    cache, missed = {}, set()

    def emb(key):
        if key not in cache:
            name, idx = key
            img = cv2.imread(str(a.lfw / "lfw" / name / f"{name}_{idx:04d}.jpg"))
            if img is None:
                raise FileNotFoundError(f"{name}_{idx:04d}.jpg")
            f = center_face(ident.faces(img), img.shape[1], img.shape[0])
            if f is None:
                missed.add(key)
                cache[key] = None
            else:
                cache[key] = ident.embed(img, f)
        return cache[key]

    t0 = time.time()
    scores, labels, fold = [], [], []
    for x, y, same, k in pairs:
        ex, ey = emb(x), emb(y)
        # A missed face counts as a non-match (score -1): honest worst case.
        s = -1.0 if ex is None or ey is None else float(
            np.dot(ex, ey) / (np.linalg.norm(ex) * np.linalg.norm(ey))
        )
        scores.append(s); labels.append(same); fold.append(k)
    scores, labels, fold = np.array(scores), np.array(labels), np.array(fold)
    elapsed = time.time() - t0

    # Standard LFW protocol: pick the threshold on 9 folds, test on the 10th.
    accs = []
    for k in range(nfolds):
        tr, te = fold != k, fold == k
        t, _ = best_threshold(scores[tr], labels[tr])
        accs.append(np.mean((scores[te] >= t) == labels[te]))

    gen, imp = scores[labels], scores[~labels]
    best_t, best_acc = best_threshold(scores, labels)
    far_rows = {}
    for far in (0.01, 0.001):
        t = threshold_at_far(imp, far)
        far_rows[f"FAR={far:g}"] = {"threshold": round(t, 4), "TAR": round(float(np.mean(gen >= t)), 4)}

    t_rec = threshold_at_far(imp, 0.001)
    report = {
        "images": len(cache),
        "images_without_face": len(missed),
        "pairs": len(scores),
        "seconds": round(elapsed, 1),
        "accuracy_10fold_mean": round(float(np.mean(accs)), 4),
        "accuracy_10fold_std": round(float(np.std(accs)), 4),
        "best_threshold_all_pairs": round(best_t, 4),
        "best_accuracy_all_pairs": round(best_acc, 4),
        **far_rows,
        "genuine_score_pct": {q: round(float(np.percentile(gen, q)), 4) for q in (1, 5, 50)},
        "impostor_score_pct": {q: round(float(np.percentile(imp, q)), 4) for q in (50, 99, 99.9)},
        "suggested": {
            "recognize_threshold": round(t_rec, 2),   # FAR <= 0.1% per comparison
            "unknown_threshold": round(best_t, 2),    # best-accuracy split
        },
    }
    print(json.dumps(report, indent=2))
    if a.out:
        np.savez(a.out, scores=scores, labels=labels, fold=fold)


if __name__ == "__main__":
    main()
