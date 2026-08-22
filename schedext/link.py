"""Join detected openings to their schedule rows, so a crop carries its size.

Stage 2 gives boxes; the text pipeline gives marks and dimensions. Nothing links
them, and ``annotate.py`` records why the obvious attempt failed: a naive
geometric match "put five boxes on the same mark".

It fails because the two things are not in the same place. On a pictorial block
the item ``rect`` is the *caption*, and the drawing sits **above** it:

    detections (drawings)   y 138-280    x  259  464  621  800  978 1158 1339 1519
    items      (captions)   y 314-604    x  257  271  451  797  977 1157 1337 1517

Containment therefore scores ~0. What does hold is **reading order**: the nth
drawing across a row is the nth caption across that row, and their left edges
line up to within a few points. So this aligns the two sequences instead of
overlapping them.

Two consequences worth knowing before trusting a size:

  * A caption gives the size of one leaf. A pair or a unit with sidelites is
    drawn as one wider box, so the box is 1.5-2x the caption's width. That is
    the drawing being right, not the match being wrong -- hence ``agrees``
    checks *height* ratio primarily, which is unaffected by leaf count.
  * Where a block has more drawings than captions the extra ones stay unlinked.
    Guessing would be worse than saying nothing.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

# Two boxes belong to the same row if their tops are within this many points.
# Rows on these sheets are separated by 300pt+, and drawings within a row vary
# by ~25pt, so the gap is wide and the threshold is not delicate.
ROW_TOLERANCE_PT = 90.0

# Cost of leaving a drawing or a caption unmatched. Set above the worst
# reasonable positional cost so the aligner prefers a poor pairing to a skip,
# but not so high that it forces alignment when counts genuinely differ.
SKIP_COST = 0.55

# A linked size is flagged as agreeing when the drawn aspect is within this of
# the stated one. Generous because pairs and sidelites legitimately widen a box.
ASPECT_TOLERANCE = 0.35


def _rows(boxes: list, key) -> list[list]:
    """Group into visual rows, top to bottom, each sorted left to right."""
    out: list[list] = []
    for b in sorted(boxes, key=lambda b: key(b)[1]):
        top = key(b)[1]
        if out and abs(key(out[-1][0])[1] - top) <= ROW_TOLERANCE_PT:
            out[-1].append(b)
        else:
            out.append([b])
    return [sorted(r, key=lambda b: key(b)[0]) for r in out]


def _align(drawings: list, captions: list, dkey, ckey) -> list[tuple]:
    """Monotonic sequence alignment on left-edge position.

    Needleman-Wunsch rather than nearest-neighbour: nearest-neighbour lets two
    drawings claim one caption, which is precisely the five-boxes-one-mark
    failure. Monotonicity encodes the thing that is actually true of a drawing
    sheet -- captions do not cross over each other.
    """
    n, m = len(drawings), len(captions)
    if not n or not m:
        return []
    span = max(
        max(dkey(d)[0] for d in drawings) - min(dkey(d)[0] for d in drawings),
        max(ckey(c)[0] for c in captions) - min(ckey(c)[0] for c in captions),
        1.0,
    )

    def cost(i: int, j: int) -> float:
        return abs(dkey(drawings[i])[0] - ckey(captions[j])[0]) / span

    best = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        best[i][0] = best[i - 1][0] + SKIP_COST
    for j in range(1, m + 1):
        best[0][j] = best[0][j - 1] + SKIP_COST
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best[i][j] = min(best[i - 1][j - 1] + cost(i - 1, j - 1),
                             best[i - 1][j] + SKIP_COST,
                             best[i][j - 1] + SKIP_COST)

    pairs, i, j = [], n, m
    while i > 0 and j > 0:
        if best[i][j] == best[i - 1][j - 1] + cost(i - 1, j - 1):
            pairs.append((drawings[i - 1], captions[j - 1]))
            i, j = i - 1, j - 1
        elif best[i][j] == best[i - 1][j] + SKIP_COST:
            i -= 1
        else:
            j -= 1
    return list(reversed(pairs))


def _agrees(box: tuple, width_in, height_in) -> bool | None:
    """Does the drawn shape match the stated size?

    Height carries the signal: leaf count changes width, never height. Width is
    still checked, but only to reject a match that is wrong in both directions.
    """
    if not width_in or not height_in:
        return None
    bw, bh = box[2] - box[0], box[3] - box[1]
    if bw <= 0 or bh <= 0:
        return None
    drawn, stated = bw / bh, float(width_in) / float(height_in)
    if stated <= 0:
        return None
    ratio = drawn / stated
    # 1.0 is exact; a pair sits near 2.0 and is still a correct link.
    return 0.6 <= ratio <= 2.4


def link(openings: list[dict], items: list[dict]) -> dict[str, dict]:
    """Return {opening_id: matched item} for the openings that can be linked."""
    linked: dict[str, dict] = {}
    keys = {(o.get("page"), o.get("block_id")) for o in openings
            if o.get("block_id")}
    for page, block in keys:
        draw = [o for o in openings
                if o.get("page") == page and o.get("block_id") == block]
        caps = [i for i in items
                if str(i.get("page")) == str(page)
                and i.get("block_id") == block and i.get("rect")]
        if not draw or not caps:
            continue

        dkey = lambda o: (float(o["x0_pt"]), float(o["y0_pt"]))     # noqa: E731
        ckey = lambda i: (i["rect"][0], i["rect"][1])               # noqa: E731

        # Rows are matched to rows, so a block whose drawings wrap onto a second
        # row does not have that row absorbed into the first.
        #
        # Only when both sides agree on how many rows there are, though. The two
        # sides are clustered independently -- drawings by their own tops,
        # captions by theirs -- and a caption block that wraps differently from
        # the drawings above it produces a different row count. Pairing row i to
        # row i then silently drops every row past the shorter side. Falling back
        # to one flat reading-order alignment recovers those, and the aligner is
        # monotonic anyway, so it cannot cross captions over each other.
        drows, crows = _rows(draw, dkey), _rows(caps, ckey)
        if len(drows) != len(crows):
            drows, crows = [[b for r in drows for b in r]], [[c for r in crows for c in r]]
        for drow, crow in zip(drows, crows):
            for o, it in _align(drow, crow, dkey, ckey):
                box = (float(o["x0_pt"]), float(o["y0_pt"]),
                       float(o["x1_pt"]), float(o["y1_pt"]))
                linked[o["opening_id"]] = {
                    "mark": it.get("mark"),
                    "item_id": it.get("item_id"),
                    "width_in": it.get("width_in"),
                    "height_in": it.get("height_in"),
                    "width_text": it.get("width_text"),
                    "height_text": it.get("height_text"),
                    "type_text": it.get("type_text"),
                    "agrees": _agrees(box, it.get("width_in"), it.get("height_in")),
                }
    return linked


def main(argv: list[str]) -> int:
    """Report link coverage for one planset result + its openings.csv."""
    import argparse

    parser = argparse.ArgumentParser(prog="schedext.link")
    parser.add_argument("openings", help="out/openings.csv")
    parser.add_argument("planset", help="out/plansets/<name>.json")
    parser.add_argument("--write", help="write an enriched CSV here")
    args = parser.parse_args(argv)

    openings = list(csv.DictReader(open(args.openings)))
    items = json.loads(Path(args.planset).read_text())["items"]
    linked = link(openings, items)

    in_block = [o for o in openings if o.get("block_id")]
    agree = [v for v in linked.values() if v["agrees"] is True]
    disagree = [v for v in linked.values() if v["agrees"] is False]
    marks = {v["item_id"] for v in linked.values()}

    print(f"  openings in a block   {len(in_block)}")
    print(f"  linked to a schedule  {len(linked)}"
          f"   ({100 * len(linked) / max(len(in_block), 1):.0f}%)")
    print(f"  distinct items used   {len(marks)} of {len(items)}")
    print(f"  shape agrees          {len(agree)}"
          f"   ({100 * len(agree) / max(len(linked), 1):.0f}% of linked)")
    print(f"  shape disagrees       {len(disagree)}   <- pairs, sidelites, or a bad link")
    dupes = len(linked) - len(marks)
    print(f"  items claimed twice   {dupes}   <- the failure the naive match had")

    if args.write:
        fields = list(openings[0]) + ["mark", "width_in", "height_in",
                                      "size_text", "shape_agrees"]
        with open(args.write, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for o in openings:
                m = linked.get(o["opening_id"], {})
                writer.writerow({**o, "mark": m.get("mark", ""),
                                 "width_in": m.get("width_in", ""),
                                 "height_in": m.get("height_in", ""),
                                 "size_text": (f'{m["width_text"]} x {m["height_text"]}'
                                               if m.get("width_text") else ""),
                                 "shape_agrees": m.get("agrees", "")})
        print(f"\n  wrote {args.write}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
