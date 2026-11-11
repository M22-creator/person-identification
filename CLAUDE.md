# Instructions for Claude Code

This repo is the person identification module for the AVA robot (UT Austin).
Owner: Hongxuan Wu. It is a working skeleton; your job is to turn it into the
real system. Read README.md first.

## Ask Hongxuan before starting

1. Where is her existing, locally working person identification code? Reuse
   its model; do not pick a new face model on your own.
2. How does she SSH into the robot computer (`avanuc`), and is it her own account?
3. Is the RealSense D435i plugged into the NUC or the Thor yet?
4. Which output channel the team wants (stdout JSON or ROS 2 topic). If
   undecided, keep both.

## Tasks, in order

1. Wrap her existing model as an `Identifier` subclass in
   `person_id/identifier.py`, register it in `BACKENDS`, set it in
   `config/default.yaml`, add its packages to `requirements.txt`.
2. Run it on a recorded video on her laptop. Tune the two thresholds in the
   config on real data; the current values are placeholders.
3. Add tests for the new backend. `python -m pytest` must pass.
4. Deploy to `avanuc`: copy the repo to her home directory, create a venv
   inside the repo, install, run on a recorded video.
5. Switch to the live camera (`--source realsense`) once it is connected.
6. Integrate with the other modules through `person_id/publishers.py`.

## Rules

- `avanuc` is shared. Work only inside this repo's directory and its venv.
  No `sudo`, no system-wide installs, no edits to `~/ava_ws`, `~/social_ws`
  or other people's files, unless Hongxuan explicitly approves each one.
- Never send robot motion commands (e.g. anything under `/robot/drive/`).
  Read-only status calls are fine.
- Never commit anything in `data/`, face images, or videos of people.
- Never auto-enroll unknown people. `uncertain` results carry no `person_id`.
- Keep the output schema in README.md stable; other modules depend on it.
  If it must change, tell Hongxuan so she can inform the team.
- Show Hongxuan each command before running it on the robot computer.
