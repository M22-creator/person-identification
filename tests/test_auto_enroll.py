import numpy as np

from person_id import Detection, Gallery, Pipeline, Status
from person_id.web.auto_enroll import AutoEnroller
from person_id.identifier import Identifier

FRAME = np.zeros((4, 4, 3), dtype=np.uint8)
A = np.array([1.0, 0.0, 0.0])
B = np.array([0.0, 1.0, 0.0])
BIG = (0, 0, 100, 100)


class Scripted(Identifier):
    def __init__(self):
        self.now = []

    def detect(self, frame):
        return self.now


def setup(tmp_path, **kw):
    ident, gallery = Scripted(), Gallery()
    pipe = Pipeline(ident, gallery, 0.5, 0.3)
    auto = AutoEnroller(gallery, str(tmp_path / "g.npz"), samples=3, sample_gap_s=1.0, **kw)

    def step(t, *dets):
        ident.now = list(dets)
        return auto.update(ident.now, pipe.process(FRAME), t)

    return gallery, auto, step


def test_enrolls_after_enough_spaced_samples(tmp_path):
    gallery, _, step = setup(tmp_path)
    assert step(0.0, Detection(BIG, A))[0].taken == 1
    assert step(0.1, Detection(BIG, A))[0].taken == 1   # too soon after the last sample
    assert step(1.0, Detection(BIG, A))[0].taken == 2
    done = step(2.0, Detection(BIG, A))[0]
    assert done.enrolled_as == "person_1"
    assert gallery.person_ids == ["person_1"]
    assert (tmp_path / "g.npz").exists()
    assert step(3.0, Detection(BIG, A)) == [None]       # now recognized, nothing to learn


def test_two_people_get_two_ids(tmp_path):
    gallery, _, step = setup(tmp_path)
    for t in (0.0, 1.0, 2.0):
        step(t, Detection(BIG, A), Detection(BIG, B))
    assert gallery.person_ids == ["person_1", "person_2"]


def test_candidate_is_forgotten_when_face_leaves(tmp_path):
    gallery, auto, step = setup(tmp_path, forget_after_s=5.0)
    step(0.0, Detection(BIG, A))
    step(10.0)                                           # nobody in view
    assert auto.candidates == [] and len(gallery) == 0
    assert step(11.0, Detection(BIG, A))[0].taken == 1   # starts over


def test_small_faces_are_ignored(tmp_path):
    _, auto, step = setup(tmp_path, min_face_px=48)
    assert step(0.0, Detection((0, 0, 20, 20), A)) == [None]
    assert auto.candidates == []


def test_ids_continue_after_existing_ones(tmp_path):
    gallery, _, step = setup(tmp_path)
    gallery.add("person_7", B)
    gallery.add("alice", np.array([0.0, 0.0, 1.0]))
    for t in (0.0, 1.0, 2.0):
        step(t, Detection(BIG, A))
    assert "person_8" in gallery.person_ids


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def test_one_odd_sample_cannot_chain_a_second_person_into_a_candidate(tmp_path):
    # What happened on the robot: person B looked like ONE of A's samples (a2), so the old
    # closest-sample rule pulled B into A's candidate and both became person_1.
    gallery, auto, step = setup(tmp_path)
    a1, a2, b = unit([1, 0, 0]), unit([0.8, 0.6, 0]), unit([0.1, 0.995, 0])
    assert float(a2 @ b) > 0.5 > float(a1 @ b)       # b is close to a2 only
    step(0.0, Detection(BIG, a1))
    step(1.0, Detection(BIG, a2))
    p = step(2.0, Detection(BIG, b))[0]
    assert p.key != 1 and p.taken == 1                # b starts its own candidate
    assert len(auto.candidates) == 2


def test_samples_that_disagree_are_dropped_before_saving(tmp_path):
    from person_id.web.auto_enroll import Candidate

    gallery, auto, _ = setup(tmp_path)
    a, odd = unit([1, 0, 0]), unit([0.2, 0.98, 0])
    cand = Candidate(key=1, embeddings=[a, a, a, odd])
    auto.candidates.append(cand)
    assert "dropped 1 sample" in auto._check(cand)
    assert len(cand.embeddings) == 3 and auto._check(cand) is None


def test_candidate_matching_an_enrolled_person_is_not_saved_again(tmp_path):
    # e.g. the same person was enrolled by hand (person_id.enroll) while being learned
    gallery, auto, step = setup(tmp_path)
    step(0.0, Detection(BIG, A))
    step(1.0, Detection(BIG, A))
    gallery._people["alice"] = np.stack([A / np.linalg.norm(A)])  # appears meanwhile
    auto_status = Status.UNKNOWN  # the frame's result is still "unknown" for this test
    det = Detection(BIG, A)
    from person_id.types import IdentityResult
    out = auto.update([det], [IdentityResult(auto_status, None, 0.1, BIG)], 2.0)
    assert out == [None] and "already enrolled as alice" in auto.last_blocked[0]
    assert gallery.person_ids == ["alice"] and auto.candidates == []


def test_reasons_for_not_learning(tmp_path):
    gallery, auto, step = setup(tmp_path, min_face_px=48)
    step(0.0, Detection((0, 0, 30, 30), A))
    assert "too small" in auto.last_blocked[0]
    gallery.add("alice", A)
    step(1.0, Detection(BIG, unit([0.4, 0.9165, 0])))  # cosine 0.4 to alice: uncertain (0.3-0.5)
    assert "similar to alice (0.40)" in auto.last_blocked[0]


def test_enrolls_in_under_two_seconds_with_the_defaults(tmp_path):
    ident, gallery = Scripted(), Gallery()
    pipe = Pipeline(ident, gallery, 0.4, 0.3)
    auto = AutoEnroller(gallery, str(tmp_path / "g.npz"), same_threshold=0.4)  # defaults: 8 samples, 0.25 s
    ident.now = [Detection(BIG, A)]
    t, saved = 0.0, None
    while saved is None and t < 5:
        p = auto.update(ident.now, pipe.process(FRAME), t)[0]
        saved = p and p.enrolled_as
        t += 1 / 30
    assert saved == "person_1" and t < 2.0
