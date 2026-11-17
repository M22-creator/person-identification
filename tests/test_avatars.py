import re

from person_id.web.avatars import AVATARS, BACKGROUNDS, avatar_for, avatar_svg


def test_every_avatar_is_a_10x10_grid_with_a_full_palette():
    assert len(AVATARS) == 10
    assert len({name for name, _, _ in AVATARS}) == 10
    for name, rows, palette in AVATARS:
        assert len(rows) == 10, name
        for row in rows:
            assert len(row) == 10, (name, row)
            for c in row:
                assert c == "." or c in palette, (name, c)
        assert sum(c != "." for row in rows for c in row) >= 40, name  # not mostly empty
        for colour in palette.values():
            assert re.fullmatch(r"#[0-9a-f]{6}", colour), (name, colour)


def test_first_ten_people_get_ten_different_avatars_then_a_new_background():
    first = [avatar_for(f"person_{i}") for i in range(1, 11)]
    assert sorted(i for i, _ in first) == list(range(10))
    assert {bg for _, bg in first} == {BACKGROUNDS[0]}
    assert avatar_for("person_11") == (0, BACKGROUNDS[1])
    assert avatar_for("person_1")[0] == 0 and avatar_for("person_2")[0] == 1  # fox, bee


def test_other_ids_are_stable_and_svg_is_well_formed():
    assert avatar_for("alice") == avatar_for("alice")
    svg = avatar_svg("person_3")
    assert svg.startswith("<svg") and svg.endswith("</svg>") and "<title>cat</title>" in svg
