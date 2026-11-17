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

Downstream modules should only act on `recognized`. Unknown people are never
enrolled automatically.

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
