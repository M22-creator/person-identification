import json

from person_id.visits import VisitLog


def test_visits_split_on_gap_and_count_only_given_ids(tmp_path):
    log = VisitLog(tmp_path / "g.visits.json", gap_s=10)
    for t in (0, 1, 2, 9):                      # one visit, short gaps
        log.seen({"alice": 0.8}, t)
    log.seen({}, 25)                            # alice gone for 16 s: visit closes
    for t in (30, 31):
        log.seen({"alice": 0.6, "bob": 0.9}, t)  # alice's second visit, bob's first
    h = log.history("alice")
    assert h["visits"] == 2 and h["first_seen"] == 0 and h["last_seen"] == 31
    assert [v["duration_s"] for v in h["history"]] == [1.0, 9.0]          # newest first
    assert h["history"][1]["mean_score"] == 0.8 and h["history"][0]["frames"] == 2
    assert h["history"][0]["ongoing"] and not h["history"][1]["ongoing"]
    assert log.summary("bob")["visits"] == 1 and log.summary("nobody")["visits"] == 0


def test_enrollment_frame_opens_first_visit_without_a_score(tmp_path):
    log = VisitLog(tmp_path / "v.json")
    log.seen({"person_1": None}, 100, enrolled={"person_1"})
    log.seen({"person_1": 0.9}, 101)
    v = log.history("person_1")["history"][0]
    assert v["enrolled"] and v["frames"] == 2 and v["mean_score"] == 0.9 and v["best_score"] == 0.9


def test_history_survives_a_restart_and_remove_deletes_it(tmp_path):
    p = tmp_path / "gallery.visits.json"
    log = VisitLog(p, gap_s=10)
    log.seen({"alice": 0.7}, 0)
    log.seen({}, 20)                            # closes the visit -> saved
    again = VisitLog(p, gap_s=10)
    assert again.summary("alice")["visits"] == 1
    again.seen({"alice": 0.7}, 1000)
    assert again.summary("alice")["visits"] == 2  # a visit after the restart is a new one
    again.remove("alice")
    assert "alice" not in json.loads(p.read_text())["people"]
    assert VisitLog(p).summary("alice")["visits"] == 0


def test_path_next_to_gallery():
    assert str(VisitLog.path_for("data/gallery_arcface_r50.npz")) == "data/gallery_arcface_r50.visits.json"
