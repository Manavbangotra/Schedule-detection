"""Score stage 1 by what stage 2 actually needs.

Stage 1's only consumer is stage 2, and stage 2 needs one thing: the openings
have to fall *inside* a region carrying the *right class*. An IoU against a
human-drawn rectangle does not measure that. A region can score a mediocre IoU
and still contain every opening, or score a respectable one and miss half of
them off its right-hand edge -- which is precisely what happens on pictorial
legends, where the word-derived rect collapses onto the dimension labels.

So the headline number here is **opening coverage**: of the openings a human
accepted and classed, what share land at least ``CONTAINMENT`` inside a proposed
region of the same class?

Two reference points make the number readable:

    41.7%   the text pipeline, measured when this module was written
    96.5%   the human's own corrected areas -- the ceiling, not 100% because
            `mixed` and `unknown` areas hand down no class at all

Run:
    .venv/bin/python -m schedext.eval.coverage              # text regions
    .venv/bin/python -m schedext.eval.coverage --ceiling    # human areas
"""

from __future__ import annotations

import collections
import glob
import json
import os
import sys
from pathlib import Path

from ..annotate import CONTAINMENT, containment

ANN_DIR = Path("annotations")
PLANSETS_DIR = Path("out/plansets")

# Block category -> the class an opening inherits. Mirrors annotate.SEED_AREA_TYPE
# collapsed to classes; `both`/`unknown` deliberately hand down nothing, which is
# why they show up as wrong-class rather than silently counting as correct.
CATEGORY_CLASS = {
    "window": "window",
    "door": "door",
    "storefront": "door",
    "garage_door": "garage_door",
}
AREA_CLASS = {"window_area": "window", "door_area": "door",
              "garage_area": "garage_door"}


def text_regions(plansets_dir: Path = PLANSETS_DIR) -> dict:
    """(project_id, page) -> [(rect, class, genre)] from the text pipeline."""
    out: dict[tuple[str, int], list] = collections.defaultdict(list)
    for path in glob.glob(str(plansets_dir / "*.json")):
        # Filenames are project_<id>_<sha8>.json; the id is the only stable key
        # shared with the annotation store.
        project_id = os.path.basename(path).split("_")[1]
        data = json.loads(Path(path).read_text())
        for sheet in data.get("sheets", []):
            for block in sheet.get("blocks", []):
                out[(project_id, sheet["page"])].append(
                    (block["rect"],
                     CATEGORY_CLASS.get(block.get("category")),
                     block.get("genre", "?"))
                )
    return out


def human_regions(sheet: dict) -> list:
    """The same shape, from a human-verified sheet -- used for the ceiling."""
    return [(a["rect"], AREA_CLASS.get(a["type"]), (a.get("seed") or {}).get("genre", "?"))
            for a in sheet.get("areas", []) if a["type"] != "exclude"]


def score_sheet(sheet: dict, regions: list, threshold: float = CONTAINMENT) -> dict:
    """Bucket every accepted, classed opening on one sheet."""
    areas = {a["id"]: a for a in sheet.get("areas", [])}
    buckets = collections.Counter()
    by_genre = collections.defaultdict(collections.Counter)

    for opening in sheet.get("openings", []):
        if opening.get("state") != "accepted" or not opening.get("cls"):
            continue
        # The genre of the block this opening really belongs to, so a loss can be
        # attributed to pictorial-vs-tabular rather than just counted.
        genre = (areas.get(opening.get("area_id"), {}).get("seed") or {}).get("genre", "?")

        shares = [(containment(opening["rect"], rect), cls) for rect, cls, _ in regions]
        best = max((s for s, _ in shares), default=0.0)
        hit_classes = {cls for share, cls in shares if share >= threshold}

        if best >= threshold and opening["cls"] in hit_classes and len(hit_classes) > 1:
            # Covered by the right class AND a wrong one at the same time. The
            # opening inherits its class from the region it sits in, so two
            # regions of different classes covering it is not a success -- it is
            # a coin flip. Without this bucket the metric is trivially gamed:
            # emitting the whole sheet once per class present scores 97.7%,
            # because every opening is inside its own class's region and the
            # wrong one is never counted.
            key = "ambiguous_class"
        elif best >= threshold and opening["cls"] in hit_classes:
            key = "right_class"
        elif best >= threshold:
            key = "wrong_class"
        elif best > 0.2:
            key = "region_too_small"
        elif regions:
            key = "outside"
        else:
            key = "no_region_on_page"
        buckets[key] += 1
        by_genre[genre][key] += 1

    return {"buckets": buckets, "by_genre": by_genre}


def run(ann_dir: Path = ANN_DIR, ceiling: bool = False,
        threshold: float = CONTAINMENT) -> dict:
    regions_by_page = {} if ceiling else text_regions()

    totals = collections.Counter()
    per_planset = collections.defaultdict(collections.Counter)
    per_genre = collections.defaultdict(collections.Counter)
    sheets = 0

    for path in sorted(glob.glob(str(ann_dir / "*.json"))):
        sheet = json.loads(Path(path).read_text())
        if sheet.get("status") != "verified":
            continue
        sheets += 1
        regions = (human_regions(sheet) if ceiling
                   else regions_by_page.get((sheet["project_id"], sheet["page"]), []))
        result = score_sheet(sheet, regions, threshold)
        totals.update(result["buckets"])
        per_planset[sheet["planset_key"]].update(result["buckets"])
        for genre, counts in result["by_genre"].items():
            per_genre[genre].update(counts)

    total = sum(totals.values())
    return {
        "source": "human areas (ceiling)" if ceiling else "text pipeline",
        "threshold": threshold,
        "sheets": sheets,
        "openings": total,
        "coverage": round(totals["right_class"] / total, 4) if total else 0.0,
        "buckets": dict(totals),
        "per_planset": {k: dict(v) for k, v in sorted(per_planset.items())},
        "per_genre": {k: dict(v) for k, v in sorted(per_genre.items())},
    }


ORDER = ["right_class", "outside", "wrong_class", "ambiguous_class",
         "region_too_small", "no_region_on_page"]
LABEL = {
    "right_class": "inside, right class",
    "outside": "outside every region",
    "wrong_class": "inside, wrong class",
    "ambiguous_class": "inside BOTH a right and a wrong region — class is a coin flip",
    "region_too_small": "partly inside — region too small",
    "no_region_on_page": "no region proposed on the page",
}


def render(report: dict) -> str:
    total = report["openings"] or 1
    lines = [
        f"source: {report['source']}   containment >= {report['threshold']}",
        f"{report['sheets']} verified sheets, {report['openings']} accepted classed openings",
        "",
        f"OPENING COVERAGE   {report['coverage']:.1%}"
        "     <- share reaching stage 2 correctly classed",
        "",
    ]
    for key in ORDER:
        n = report["buckets"].get(key, 0)
        if n or key == "right_class":
            lines.append(f"  {n:5d}  {n / total:5.1%}  {LABEL[key]}")

    lines += ["", "by block genre   (pictorial is where word-based extent fails)"]
    for genre, counts in report["per_genre"].items():
        got = counts.get("right_class", 0)
        tot = sum(counts.values())
        name = {"?": "no seed block"}.get(genre, genre)
        lines.append(f"  {name:16s} {got:4d}/{tot:<4d} {got / tot:6.1%}" if tot else "")

    lines += ["", "by planset   (a single weak planset is the expected shape)"]
    for planset, counts in report["per_planset"].items():
        got = counts.get("right_class", 0)
        tot = sum(counts.values())
        if tot:
            lines.append(f"  {planset:22s} {got:4d}/{tot:<4d} {got / tot:6.1%}")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="schedext.eval.coverage")
    parser.add_argument("--ceiling", action="store_true",
                        help="score the human's own areas instead — the upper bound")
    parser.add_argument("--containment", type=float, default=CONTAINMENT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    report = run(ceiling=args.ceiling, threshold=args.containment)
    print(json.dumps(report, indent=2) if args.json else render(report))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
