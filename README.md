# person-identification

Person identification module for the AVA robot: camera frame in, "who is this" out.

**Status: skeleton.** Everything runs end to end, but the default `dummy`
identifier detects nobody. Plug your model into `person_id/identifier.py`.

## Layout

| File | What it does | Need to edit? |
|---|---|---|
| `person_id/identifier.py` | Face model: frame -> boxes + embeddings | **Yes, your model goes here** |
| `person_id/gallery.py` | Enrolled people and cosine matching | No |
| `person_id/pipeline.py` | recognized / unknown / uncertain / no_person logic | No |
| `person_id/camera.py` | RealSense, webcam, or video file input | Maybe (RealSense untested) |
| `person_id/publishers.py` | Output as JSON lines or a ROS 2 topic | Maybe (ROS 2 untested) |
| `config/default.yaml` | Thresholds, camera, output settings | Tune thresholds |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
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

## TODO

- [ ] Add the real identifier backend and set `identifier:` in the config
- [ ] Tune `recognize_threshold` / `unknown_threshold` (current values are placeholders)
- [ ] Test RealSense D435i input on the robot
- [ ] Agree on the output channel with the state machine (stdout JSON vs ROS 2 topic)
