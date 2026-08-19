"""Detection gate: recall and no-nesting on hand-counted regions.

Exact-count equality was the wrong test. The detector boxes *leaves* -- a
triple-wide mulled window comes back as three sashes 11px apart, where I had
counted one assembly. No dedup threshold reconciles those two definitions, and
pretending otherwise just tunes the thresholds until they overfit.

What actually matters for "where is an opening":
  * recall  -- every real opening is covered by at least one box;
  * no nesting -- no box survives inside another (that would be double counting).
The leaf-to-entry ratio is printed, not enforced.

Run: .venv/bin/python -m schedext.eval.test_detect
"""

from __future__ import annotations

import sys
from pathlib import Path

from schedext import detect

WEIGHTS = "/var/www/html/WindowsDoorsClassification/best.pt"
PAGES = Path("viewer/pages")

# Regions were marked at zoom 2.0; the rendered sheets are zoom 1.8.
SCALE = detect.ZOOM / 2.0

# (image, region name, region box at zoom 2.0, hand-counted assemblies)
REGIONS = [
    ("559_p1", "door row", (340, 150, 3320, 650), 8),
    ("559_p1", "window row", (340, 1820, 1800, 2290), 4),
    ("486_p18", "window strip", (330, 700, 3300, 1400), 6),
    ("486_p18", "DOOR TYPES legend", (300, 3150, 3200, 4230), 11),
]


def _boxes_in(model, image_name: str, region) -> list:
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    sheet = Image.open(PAGES / f"{image_name}.jpg").convert("RGB")
    raw = [(b, c, "sheet") for b, c in detect._predict(
        model, sheet, detect.SHEET_IMGSZ, detect.CONF)]
    x0, y0, x1, y1 = [v * SCALE for v in region]
    inside = [t for t in raw
              if t[0][0] >= x0 - 30 and t[0][2] <= x1 + 30
              and t[0][1] >= y0 - 30 and t[0][3] <= y1 + 30]
    return detect.dedup(inside)


def main() -> int:
    if not PAGES.exists():
        print("viewer/pages missing — run `cli viewer` first")
        return 1

    model = detect._load(WEIGHTS)
    problems: list[str] = []

    print(f"{'region':26s} {'raw':>4} {'kept':>5} {'entries':>8} {'leaf/entry':>11}  nesting")
    for image, name, region, entries in REGIONS:
        kept = _boxes_in(model, image, region)

        # no survivor may sit inside another
        nested = 0
        for i, (box, _, _) in enumerate(kept):
            for j, (other, _, _) in enumerate(kept):
                if i == j or detect._area(box) >= detect._area(other):
                    continue
                share = detect._intersection(box, other) / max(detect._area(box), 1e-9)
                if share >= detect.CONTAINMENT:
                    nested += 1
                    break
        if nested:
            problems.append(f"{name}: {nested} nested box(es) survived dedup")

        # recall: never fewer boxes than hand-counted assemblies
        if len(kept) < entries:
            problems.append(f"{name}: {len(kept)} boxes < {entries} known openings — misses")

        ratio = len(kept) / entries if entries else 0
        print(f"{name:26s} {'-':>4} {len(kept):5d} {entries:8d} {ratio:11.2f}  "
              f"{'ok' if not nested else str(nested)}")

    for problem in problems:
        print(f"FAIL {problem}")
    print(f"\n{len(REGIONS) - len(problems)}/{len(REGIONS)} regions pass recall + no-nesting")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
