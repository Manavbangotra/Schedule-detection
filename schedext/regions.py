"""Export a stage-1 region dataset: whole pages, region boxes.

Stage 2 detects openings *inside* a region; this detects the regions. The unit
is therefore a whole page, not a crop, and the boxes are large -- median 434x159
px at imgsz 1280, p10 166x101 -- so nothing here is in stage 2's 23-51px regime.

Three decisions worth stating, because each was measured rather than assumed.

**Gold only, never silver.** The text pipeline's own block rects (307 of them)
are exactly the rects the human corrected: training on them teaches the model to
reproduce the error being removed. Worse, 102 of those 307 are
``kind=schedule, genre=tabular``, which the text pipeline labels a *positive
region* and the human labels ``exclude``. A third of silver is sign-flipped
against gold. That is a label contradiction, not a domain gap, and no amount of
fine-tuning removes it from a backbone.

**`exclude` is a class, not background.** Those 38 human-verified regions are the
corpus's only written statement of "this is a real schedule and it holds no drawn
openings". As background they say nothing; as a class they compete in NMS and
suppress the tabular-schedule false positives that cost 1,254 junk detections
when stage 2 was first seeded.

**`mixed` and `unknown` are dropped.** The human was shown these and declined to
class them. A box whose class a human refused to give is not a training example;
at inference they belong in a review queue instead.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from . import annotate, geom, manifest as manifest_mod
from .dataset import DUPLICATE_JACCARD, VAL_BUCKET, is_val

CLASSES = ["window_region", "door_region", "exclude_region"]
CLASS_INDEX = {name: index for index, name in enumerate(CLASSES)}

# garage_area folds into door for the same reason dataset.FOLD does it: three
# examples cannot train a class, and as a third one its AP is noise.
REGION_FOLD = {
    "window_area": "window_region",
    "door_area": "door_region",
    "garage_area": "door_region",
    "exclude": "exclude_region",
    # mixed / unknown deliberately absent -- see the module docstring.
}

# Pages render to their long side, then the dataloader downsamples to imgsz.
# 1600 rather than dataset.EXPORT_ZOOM's 4.0: a 3024x2160pt sheet at zoom 4 is
# 104 megapixels, and the target here is the whole page rather than a small crop.
LONG_SIDE = 1600
JPEG_QUALITY = 88

# Non-schedule pages carried as pure background. The dangerous ones dominate this
# corpus -- 47% of pages mention DOOR or WINDOW, 35% carry some discipline's
# SCHEDULE, 33% say ELEVATION -- so a model that has never seen one will fire on
# all three. Capped per planset or the two 140-page sets supply a third of the
# pool and the model learns a drafting style instead of a region.
NEGATIVE_RATIO = 1.5
MAX_NEGATIVES_PER_PLANSET = 4


@dataclass
class PageExample:
    name: str
    planset_key: str
    project_id: str
    file: str
    page: int
    width: int
    height: int
    zoom: float
    boxes: list[tuple[int, float, float, float, float]] = field(default_factory=list)
    kind: str = "positive"          # positive | negative


def _zoom_for(page: pymupdf.Page, long_side: int = LONG_SIDE) -> float:
    return long_side / max(page.rect.width, page.rect.height)


def _render(page: pymupdf.Page, out_path: Path, long_side: int = LONG_SIDE):
    """Render a whole page. No clip, so this is display space throughout."""
    zoom = _zoom_for(page, long_side)
    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
    pixmap.save(out_path, jpg_quality=JPEG_QUALITY)
    return pixmap.width, pixmap.height, zoom


def _to_yolo(rect, width_pt: float, height_pt: float):
    """Region rect in display points -> normalised cx cy w h, clipped to page."""
    x0 = max(0.0, min(rect[0], width_pt))
    y0 = max(0.0, min(rect[1], height_pt))
    x1 = max(0.0, min(rect[2], width_pt))
    y1 = max(0.0, min(rect[3], height_pt))
    if x1 - x0 < 1.0 or y1 - y0 < 1.0:
        return None
    return ((x0 + x1) / 2 / width_pt, (y0 + y1) / 2 / height_pt,
            (x1 - x0) / width_pt, (y1 - y0) / height_pt)


def _page_words(record: dict, page_number: int) -> frozenset:
    """Word fingerprint for duplicate detection -- same rule as dataset.py."""
    try:
        with pymupdf.open(record["path"]) as doc:
            words = geom.words(doc[page_number - 1])
    except Exception:
        return frozenset()
    return frozenset(w.text.upper() for w in words if len(w.text) > 2)


def _dedupe(sheets: list[dict], records: dict) -> tuple[list[dict], list]:
    """Drop pages that are the same drawing as one already kept.

    Matters more here than in dataset.py: the unit *is* the page, so a duplicate
    pair split across train and val leaks the whole example rather than a crop.
    """
    prints = {}
    for sheet in sheets:
        record = records.get(sheet["project_id"])
        prints[sheet["sheet_key"]] = _page_words(record, sheet["page"]) if record else frozenset()

    kept, groups = [], []
    for sheet in sheets:
        words = prints[sheet["sheet_key"]]
        if not words:
            kept.append(sheet)
            continue
        for other in kept:
            other_words = prints[other["sheet_key"]]
            if not other_words:
                continue
            union = len(words | other_words)
            if union and len(words & other_words) / union >= DUPLICATE_JACCARD:
                groups.append([other["sheet_key"], sheet["sheet_key"]])
                break
        else:
            kept.append(sheet)
    return kept, groups


def _negative_pages(records: dict, positives: set, want: int,
                    seed: int = 0) -> list[tuple[str, int]]:
    """Sample non-schedule pages, capped per planset."""
    rng = random.Random(seed)
    pool: list[tuple[str, int]] = []
    for project_id, record in sorted(records.items()):
        if record.get("health") in {"truncated", "unopenable"}:
            continue
        pages = [p for p in range(1, record.get("page_count", 0) + 1)
                 if (project_id, p) not in positives]
        rng.shuffle(pages)
        pool.extend((project_id, p) for p in pages[:MAX_NEGATIVES_PER_PLANSET])
    rng.shuffle(pool)
    return pool[:want]


def build(ann_dir: Path, manifest_path: Path, out_root: Path, tag: str = "r1",
          negatives: bool = True) -> dict:
    records = {r["project_id"]: r for r in manifest_mod.load(manifest_path)}
    root = out_root / tag
    images, labels = root / "images", root / "labels"
    for directory in (images, labels):
        directory.mkdir(parents=True, exist_ok=True)

    verified = []
    for path in sorted(ann_dir.glob("*.json")):
        sheet = json.loads(path.read_text())
        if sheet.get("status") == "verified":
            verified.append(sheet)

    kept, duplicate_groups = _dedupe(verified, records)

    examples: list[PageExample] = []
    dropped = Counter()
    positives_seen: set = set()

    for sheet in kept:
        record = records.get(sheet["project_id"])
        if not record:
            continue
        boxes = []
        for area in sheet.get("areas", []):
            # A sheet's "verified" stamp only covers the areas that existed when
            # the human pressed the button. `merge` appends later proposals with
            # new_since_seed, and those carry a gold badge they did not earn --
            # measured at 10 of 155 exportable areas after one merge. Excluding
            # them is the difference between a gold set and a mixed one.
            if (area.get("seed") or {}).get("new_since_seed"):
                dropped["unverified_new_since_seed"] += 1
                continue
            cls = REGION_FOLD.get(area["type"])
            if cls is None:
                dropped[area["type"]] += 1
                continue
            boxes.append((cls, area["rect"]))
        if not boxes:
            continue

        name = sheet["sheet_key"]
        try:
            with pymupdf.open(record["path"]) as doc:
                page = doc[sheet["page"] - 1]
                width_pt, height_pt = page.rect.width, page.rect.height
                pixel_w, pixel_h, zoom = _render(page, images / f"{name}.jpg")
        except Exception:
            dropped["render_failed"] += 1
            continue

        rows = []
        for cls, rect in boxes:
            norm = _to_yolo(rect, width_pt, height_pt)
            if norm is None:
                dropped["degenerate_box"] += 1
                continue
            rows.append((CLASS_INDEX[cls], *norm))
        if not rows:
            (images / f"{name}.jpg").unlink(missing_ok=True)
            continue

        positives_seen.add((sheet["project_id"], sheet["page"]))
        examples.append(PageExample(
            name=name, planset_key=sheet["planset_key"],
            project_id=sheet["project_id"], file=sheet["file"], page=sheet["page"],
            width=pixel_w, height=pixel_h, zoom=zoom, boxes=rows))

    if negatives:
        want = int(len(examples) * NEGATIVE_RATIO)
        for project_id, page_number in _negative_pages(records, positives_seen, want):
            record = records[project_id]
            name = f"neg_{project_id}_{record['sha256'][:8]}_p{page_number}"
            try:
                with pymupdf.open(record["path"]) as doc:
                    pixel_w, pixel_h, zoom = _render(doc[page_number - 1],
                                                     images / f"{name}.jpg")
            except Exception:
                continue
            examples.append(PageExample(
                name=name, planset_key=f"{project_id}_{record['sha256'][:8]}",
                project_id=project_id, file=Path(record["path"]).name,
                page=page_number, width=pixel_w, height=pixel_h, zoom=zoom,
                kind="negative"))

    for example in examples:
        (labels / f"{example.name}.txt").write_text(
            "".join(f"{c} {a:.6f} {b:.6f} {w:.6f} {h:.6f}\n"
                    for c, a, b, w, h in example.boxes))

    return _write(root, examples, duplicate_groups, dropped)


def _write(root: Path, examples: list[PageExample], duplicate_groups: list,
           dropped: Counter) -> dict:
    train = [e for e in examples if not is_val(e.planset_key)]
    val = [e for e in examples if is_val(e.planset_key)]

    for split, items in (("train", train), ("val", val)):
        # "./" is required: ultralytics expands it to the txt file's parent and
        # then swaps "/images/" for "/labels/". Same trap as dataset.py.
        (root / f"{split}.txt").write_text(
            "".join(f"./images/{e.name}.jpg\n" for e in items))

    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASSES))
    (root / "dataset.yaml").write_text(
        "# Stage-1 regions. Whole pages, human-verified boxes only.\n"
        "# Split by planset, hash-stable, shared with the stage-2 dataset so a\n"
        "# planset never lands in train for one stage and val for the other.\n"
        f"path: {root.resolve()}\n"
        f"train: train.txt\nval: val.txt\nnames:\n{names}\n"
    )

    def summarise(items):
        counts = Counter()
        for e in items:
            for row in e.boxes:
                counts[CLASSES[row[0]]] += 1
        return {"images": len(items),
                "positives": sum(1 for e in items if e.kind == "positive"),
                "negatives": sum(1 for e in items if e.kind == "negative"),
                "boxes": sum(len(e.boxes) for e in items),
                "by_class": dict(counts)}

    report = {
        "tag": root.name,
        "images": len(examples),
        "train": summarise(train),
        "val": summarise(val),
        "plansets": {"train": len({e.planset_key for e in train}),
                     "val": len({e.planset_key for e in val})},
        "duplicates_dropped": duplicate_groups,
        "dropped_areas": dict(dropped),
    }
    (root / "report.json").write_text(json.dumps(report, indent=2))
    return report
