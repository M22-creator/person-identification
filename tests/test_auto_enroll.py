import numpy as np

from person_id import Detection, Gallery, Pipeline
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
