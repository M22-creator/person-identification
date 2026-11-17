# person-identification

Person identification module for the AVA robot: camera frame in, "who is this" out.

**Status:** runs end to end with the `sface` backend (OpenCV YuNet face detector +
SFace embedding), benchmarked on LFW. RealSense and ROS 2 output are still untested
on the robot.

## Layout

| File | What it does | Need to edit? |
|---|---|---|
| `person_id/identifier.py` | Face model: frame -> boxes + embeddings (`sface`, `dummy`) | Only to add a backend |
| `person_id/download_models.py` | Fetches the YuNet + SFace weights into `models/` | No |
| `tools/bench_lfw.py` | LFW verification benchmark + threshold suggestions | No |
| `person_id/gallery.py` | Enrolled people and cosine matching | No |
| `person_id/pipeline.py` | recognized / unknown / uncertain / no_person logic | No |
| `person_id/camera.py` | RealSense, webcam, or video file input | Maybe (RealSense untested) |
| `person_id/web/` | Live web view: `server.py`, `auto_enroll.py` (saves new faces), `index.html` | No |
| `person_id/publishers.py` | Output as JSON lines or a ROS 2 topic | Maybe (ROS 2 untested) |
| `config/default.yaml` | Thresholds, camera, output settings | Tune thresholds |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m person_id.download_models   # ~39 MB into models/ (git-ignored), SHA-256 checked
pip install pytest && python -m pytest
```

## Use

```bash
# Enroll someone (exactly one person in view), then run
python -m person_id.enroll --id alice --source 0
python -m person_id.run --source 0              # webcam
python -m person_id.run --source clip.mp4       # offline video
python -m person_id.run --source realsense      # needs pyrealsense2
python -m person_id.enroll --id alice --remove  # delete someone
```

## Output

One JSON list per frame, one entry per person (or a single `no_person` entry):

```json
[{"status": "recognized", "person_id": "alice", "confidence": 0.94,
  "bbox": [120, 80, 260, 240], "timestamp": 1794360000.0}]
```

| status | meaning | person_id |
|---|---|---|
| `recognized` | matched an enrolled person | set |
| `unknown` | a person who is not enrolled | null |
| `uncertain` | match too weak to trust | null |
| `no_person` | nobody in frame | null |

Downstream modules should only act on `recognized`. `person_id.run` never
enrolls anyone. The live web view does (see below).

## Live web view

```bash
python -m person_id.web --source 4      # then open http://localhost:8000
```

Shows the camera with a box per face and the enrolled people. A face the
gallery does not know is saved automatically as `person_N` after a few seconds
(`auto_enroll` in the config, or the Pause button on the page). Every
participant must have agreed to this beforehand; pause it when anyone else can
walk into view. The page has no login and listens on localhost only: from a
laptop use `ssh -L 8000:localhost:8000 <user>@<robot>`. It owns the camera, so
do not run `person_id.run` at the same time; pass `--publisher ros2` instead.

## Comparing recognition models

The web view has a **Recognition model** selector and a **Compare models** page
(`/compare`). Three backends share the same YuNet detector, so only the
recognition model differs:

| backend | model | embedding | license |
|---|---|---|---|
| `sface` | OpenCV SFace | 128-D | Apache-2.0 |
| `arcface_mbf` | InsightFace buffalo_s, ArcFace MobileFaceNet | 512-D | non-commercial research only |
| `arcface_r50` | InsightFace buffalo_l, ArcFace ResNet-50 | 512-D | non-commercial research only |

Each backend has its own gallery file and thresholds (`models:` in the config),
because embeddings of different models cannot be compared. Switching models keeps
every model's enrolled people; they are just in separate galleries.

```bash
pip install onnxruntime                                   # ArcFace backends (CPU)
python -m person_id.download_models --model all           # ~420 MB download, ~190 MB kept
python tools/bench_models.py --lfw data/bench/lfw --out data/bench/results.json
```

`bench_models.py` runs the LFW 6000-pair, 10-fold benchmark for each model and
times detection and embedding on 640x480 frames on the current machine. The
`/compare` page shows its `results.json` (error rate, TAR at FAR 0.1%, ROC,
latency, memory) plus live latency from the camera stream. Run the benchmark
while nothing else uses the CPU heavily; the load average is recorded.

## Data

`data/` holds face embeddings, which are biometric data. It is git-ignored;
keep it that way, and enroll only people who have agreed.

## Model and benchmark

`sface` = [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) (MIT)
\+ [SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) (Apache-2.0),
run through `cv2.FaceDetectorYN` / `cv2.FaceRecognizerSF`, so no extra packages are needed.
Faces smaller than 40 px are skipped.

LFW verification (6000 pairs, standard 10-fold protocol, all 7701 images had a face detected):

| metric | value |
|---|---|
| 10-fold accuracy | 99.33% ± 0.39% |
| TAR @ FAR 1% / 0.1% | 99.27% / 98.80% |
| at `recognize_threshold` 0.40 | TAR 97.6%, FAR 0.03% |
| same-person pairs below `unknown_threshold` 0.30 | 0.8% |

`recognize_threshold` is set conservatively because a gallery match takes the max
over every enrolled sample, so the real false-match rate grows with gallery size.
LFW is web photos, not robot-camera footage: re-check the thresholds on recorded
robot video before relying on them. To reproduce (keep LFW outside the repo):

```bash
python tools/bench_lfw.py --lfw ~/.cache/person_id_bench/lfw
PERSON_ID_LFW=~/.cache/person_id_bench/lfw python -m pytest   # adds an LFW end-to-end test
```

## TODO

- [x] Add the real identifier backend and set `identifier:` in the config
- [x] Tune `recognize_threshold` / `unknown_threshold` (on LFW; re-check on robot footage)
- [ ] Test RealSense D435i input on the robot
- [ ] Agree on the output channel with the state machine (stdout JSON vs ROS 2 topic)
