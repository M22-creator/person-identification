import numpy as np
import pytest

from person_id import Detection, Gallery, Pipeline, Status
from person_id.identifier import DummyIdentifier, Identifier

FRAME = np.zeros((4, 4, 3), dtype=np.uint8)
ALICE = np.array([1.0, 0.0, 0.0])
BOB = np.array([0.0, 1.0, 0.0])


class Fixed(Identifier):
    def __init__(self, *embeddings):
        self.embeddings = embeddings

    def detect(self, frame):
        return [Detection((0, 0, 1, 1), e) for e in self.embeddings]


def gallery():
    g = Gallery()
    g.add("alice", ALICE)
    g.add("bob", BOB)
    return g


def run(embedding, g=None):
    return Pipeline(Fixed(embedding), gallery() if g is None else g, 0.5, 0.3).process(FRAME)[0]


def test_no_person():
    r = Pipeline(DummyIdentifier(), gallery()).process(FRAME)
    assert [x.status for x in r] == [Status.NO_PERSON]


def test_recognized():
    r = run(np.array([0.9, 0.1, 0.0]))
    assert (r.status, r.person_id) == (Status.RECOGNIZED, "alice")


def test_unknown():
    r = run(np.array([0.0, 0.0, 1.0]))
    assert (r.status, r.person_id) == (Status.UNKNOWN, None)


def test_uncertain_gives_no_id():
    r = run(np.array([0.4, 0.0, 0.9]))  # cosine to alice ~0.41
    assert (r.status, r.person_id) == (Status.UNCERTAIN, None)


def test_empty_gallery_is_unknown():
    assert run(ALICE, Gallery()).status == Status.UNKNOWN


def test_multiple_people():
    r = Pipeline(Fixed(ALICE, BOB), gallery()).process(FRAME)
    assert [x.person_id for x in r] == ["alice", "bob"]


def test_gallery_roundtrip_and_remove(tmp_path):
    path = str(tmp_path / "g.npz")
    g = gallery()
    g.save(path)
    g2 = Gallery.load(path)
    assert g2.person_ids == ["alice", "bob"]
    g2.remove("bob")
    assert g2.match(BOB)[0] == "alice" and g2.match(BOB)[1] < 0.3


def test_bad_thresholds():
    with pytest.raises(ValueError):
        Pipeline(DummyIdentifier(), Gallery(), 0.3, 0.5)
