"""Ten original 10x10 pixel-art avatars for the web view's people lists.

Drawn for this project (not taken from any game), so they can ship with the code.
An avatar carries no information about the person's face: it is chosen from the
person_id only. person_1..person_10 get ten different animals in order; from
person_11 on the sequence repeats on a different background colour. Other IDs
(e.g. names enrolled by hand) are mapped by a stable hash.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

K, W = "#1d1d1f", "#f4f1ea"  # black, off-white (shared)

# name -> (10 rows of 10 cells, palette). "." is empty (background shows through).
AVATARS: List[Tuple[str, List[str], Dict[str, str]]] = [
    ("fox", [
        "O........O",
        "OO......OO",
        "OOOOOOOOOO",
        "OOOOOOOOOO",
        "OKKOOOOKKO",
        "OOOOOOOOOO",
        "WWOOOOOOWW",
        ".WWWWWWWW.",
        "..WWKKWW..",
        "...WWWW...",
    ], {"O": "#e8772e", "W": W, "K": K}),
    ("bee", [
        "..CC..CC..",
        "..CC..CC..",
        ".YYYYYYYY.",
        "YKKYYYYKKY",
        "YYYYYYYYYY",
        "KKKKKKKKKK",
        "YYYYYYYYYY",
        "KKKKKKKKKK",
        ".YYYYYYYY.",
        "....KK....",
    ], {"C": "#bfe3f5", "Y": "#f5c518", "K": K}),
    ("cat", [
        "G........G",
        "GG......GG",
        "GGGGGGGGGG",
        "GGGGGGGGGG",
        "GGEGGGGEGG",
        "GGKGGGGKGG",
        "WGGGPPGGGW",
        "GGGKGGKGGG",
        "GGGGGGGGGG",
        ".GGGGGGGG.",
    ], {"G": "#9aa0a6", "E": "#7ccf5a", "K": K, "P": "#f29bb0", "W": W}),
    ("frog", [
        ".GGG..GGG.",
        "GWKG..GWKG",
        "GGGGGGGGGG",
        "GGGGGGGGGG",
        "GGGGGGGGGG",
        "GDGGGGGGDG",
        "GGDDDDDDGG",
        "GGGGGGGGGG",
        "LLGGGGGGLL",
        ".LLLLLLLL.",
    ], {"G": "#4caf50", "W": W, "K": K, "D": "#1f5f22", "L": "#b7e08a"}),
    ("penguin", [
        "...KKKK...",
        "..KKKKKK..",
        ".KKWWWWKK.",
        ".KWKWWKWK.",
        ".KWWOOWWK.",
        "KKWWWWWWKK",
        "KKWWWWWWKK",
        "KKWWWWWWKK",
        ".KWWWWWWK.",
        "..OO..OO..",
    ], {"K": "#2b2f3a", "W": W, "O": "#f39c12"}),
    ("owl", [
        "B........B",
        "BB......BB",
        "BBBBBBBBBB",
        "BWWWBBWWWB",
        "BWKWBBWKWB",
        "BWWWYYWWWB",
        "BBBBYYBBBB",
        "BTBTBBTBTB",
        "BBTBTTBTBB",
        ".BBBBBBBB.",
    ], {"B": "#8d5a3b", "W": W, "K": K, "Y": "#f5b041", "T": "#d9b38c"}),
    ("panda", [
        "KK......KK",
        "KKWWWWWWKK",
        ".WWWWWWWW.",
        "WWWWWWWWWW",
        "WKKWWWWKKW",
        "WKKWWWWKKW",
        "WWWWKKWWWW",
        "WWWWWWWWWW",
        ".WWWWWWWW.",
        "..WWWWWW..",
    ], {"K": K, "W": W}),
    ("rabbit", [
        ".WW....WW.",
        ".WP....PW.",
        ".WP....PW.",
        ".WWWWWWWW.",
        "WWWWWWWWWW",
        "WKWWWWWWKW",
        "WWWWPPWWWW",
        "WWWKWWKWWW",
        ".WWWWWWWW.",
        "..WWWWWW..",
    ], {"W": "#e9e4dc", "P": "#f29bb0", "K": K}),
    ("robot", [
        "....RR....",
        "....SS....",
        ".SSSSSSSS.",
        ".SDDDDDDS.",
        ".SDCDDCDS.",
        ".SDDDDDDS.",
        ".SDCCCCDS.",
        ".SSSSSSSS.",
        "S.SSSSSS.S",
        "..SS..SS..",
    ], {"R": "#e74c3c", "S": "#aab4be", "D": "#26313d", "C": "#4dd0e1"}),
    ("explorer", [
        "..YYYYYY..",
        ".YYYYYYYY.",
        "YYYYYYYYYY",
        ".SSSSSSSS.",
        ".SKSSSSKS.",
        ".SSSSSSSS.",
        ".SSSMMSSS.",
        "..SSSSSS..",
        ".BBBBBBBB.",
        "BBBBBBBBBB",
    ], {"Y": "#f1c40f", "S": "#f2c9a0", "K": K, "M": "#c0392b", "B": "#3f7fd8"}),
]
BACKGROUNDS = ["#2a2f38", "#33283d", "#24363a", "#3a3324"]  # cycles every 10 people


def avatar_for(person_id: str) -> Tuple[int, str]:
    """(avatar index, background colour) for an ID."""
    m = re.fullmatch(r"person_(\d+)", person_id)
    if m and int(m.group(1)) >= 1:
        n = int(m.group(1)) - 1
        return n % len(AVATARS), BACKGROUNDS[(n // len(AVATARS)) % len(BACKGROUNDS)]
    h = 2166136261  # FNV-1a, stable across runs (Python's hash() is not)
    for ch in person_id.encode():
        h = ((h ^ ch) * 16777619) & 0xFFFFFFFF
    return h % len(AVATARS), BACKGROUNDS[(h // len(AVATARS)) % len(BACKGROUNDS)]


def avatar_svg(person_id: str) -> str:
    idx, bg = avatar_for(person_id)
    name, rows, palette = AVATARS[idx]
    cells = []
    for y, row in enumerate(rows):
        for x, c in enumerate(row):
            if c != ".":
                cells.append(f'<rect x="{x}" y="{y}" width="1" height="1" fill="{palette[c]}"/>')
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="-1 -1 12 12" shape-rendering="crispEdges">'
        f'<title>{name}</title><rect x="-1" y="-1" width="12" height="12" fill="{bg}"/>'
        + "".join(cells) + "</svg>"
    )
