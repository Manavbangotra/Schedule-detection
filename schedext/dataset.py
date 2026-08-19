"""Export verified annotations as a YOLO dataset (stage E).

Reads only sheets a human marked ``verified``. Everything else is ignored --
a half-checked sheet silently entering training data is worse than a smaller
dataset.

Three rules here are load-bearing, and each exists because the data made it
necessary rather than because it is conventional:

  * **Deduplicate by page content before splitting.** Five pairs of verified
    sheets are the same drawing under different project ids (543 / 558 / 559,
    and 585 twice). Splitting by planset would put one copy in train and its
    twin in val, and the score would come out inflated.
  * **Fold garage_door into door.** Three areas and four boxes cannot train a
    class; as a third class its AP is noise that drags reported mAP by a third
    for reasons unrelated to the model.
  * **Emit hard negatives.** Without crops of the excluded regions the model
    has never seen a title block or a notes column, and fires on both the first
    time it meets a full sheet.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from . import annotate

# Render high and let the dataloader downsample. Rendering near the target size
# aliases 1px CAD strokes into dropouts; downsampling from 4x antialiases them.
EXPORT_ZOOM = 4.0

# Two classes. garage_door folds into door at export -- see the module docstring.
CLASSES = ["window", "door"]
CLASS_INDEX = {name: i for i, name in enumerate(CLASSES)}
FOLD = {"window": "window", "door": "door", "garage_door": "door"}

# Page-content Jaccard at or above this is the same drawing.
DUPLICATE_JACCARD = 0.95

# Share of the exported set that should be background-only crops.
HARD_NEGATIVE_SHARE = 0.15

# A box must survive this much of itself after clipping to the crop.
MIN_BOX_RETAINED = 0.6
VAL_BUCKET = 4          # sha256(planset)[0] % 5 == 4  -> ~20% val


@dataclass
class Example:
    """One training image and its boxes."""

    name: str
    sheet_key: str
    planset_key: str
    project_id: str
    area_id: str
    area_type: str
    width: int
    height: int
    boxes: list[tuple[int, float, float, float, float]] = field(default_factory=list)
    is_negative: bool = False


# --- deduplication ----------------------------------------------------------

def _page_words(record: dict, page: int) -> frozenset:
    try:
        with pymupdf.open(record["path"]) as doc:
            return frozenset(
                w[4].upper() for w in doc[page - 1].get_text("words") if len(w[4]) > 2
            )
    except Exception:
        return frozenset()


def deduplicate(sheets: list, records: dict) -> tuple[list, list[list[str]]]:
    """Drop sheets that are the same drawing as one already kept.

    Compared on page content, not on project id -- the damaging duplicates in
    this corpus carry *different* project ids.
    """
    fingerprints = {s.sheet_key: _page_words(records[s.pdf_sha256[:8]], s.page)
                    for s in sheets if s.pdf_sha256[:8] in records}
    kept: list = []
    groups: list[list[str]] = []
    for sheet in sheets:
        words = fingerprints.get(sheet.sheet_key)
        if not words:
            kept.append(sheet)
            continue
        for index, other in enumerate(kept):
            other_words = fingerprints.get(other.sheet_key)
            if not other_words:
                continue
            union = len(words | other_words)
            if union and len(words & other_words) / union >= DUPLICATE_JACCARD:
                groups.append([other.sheet_key, sheet.sheet_key])
                break
        else:
            kept.append(sheet)
    return kept, groups


# --- split ------------------------------------------------------------------

def is_val(planset_key: str) -> bool:
    """Hash-stable split, so adding plansets never reshuffles the old ones."""
    return hashlib.sha256(planset_key.encode()).digest()[0] % 5 == VAL_BUCKET


# --- rendering --------------------------------------------------------------

def _render_area(page: pymupdf.Page, rect: list[float], out_path: Path) -> tuple[int, int]:
    """Render one area from the PDF.

    The clip goes in **display** space. get_pixmap and get_text disagree about
    coordinates, and 36 of 59 plansets are rotated -- passing an extraction rect
    here returns a transposed crop of the wrong region.
    """
    clip = pymupdf.Rect(*rect) & page.rect
    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(EXPORT_ZOOM, EXPORT_ZOOM), clip=clip)
    pixmap.save(out_path)
    return pixmap.width, pixmap.height


def _to_yolo(box: list[float], area: list[float]) -> tuple[float, float, float, float] | None:
    """Box in display points -> YOLO cx,cy,w,h normalised to the area crop."""
    aw, ah = area[2] - area[0], area[3] - area[1]
    if aw <= 0 or ah <= 0:
        return None
    x0 = max(box[0], area[0]); y0 = max(box[1], area[1])
    x1 = min(box[2], area[2]); y1 = min(box[3], area[3])
    if x1 <= x0 or y1 <= y0:
        return None
    # A box mostly outside its crop is a labelling artefact, not an example.
    clipped = (x1 - x0) * (y1 - y0)
    original = max((box[2] - box[0]) * (box[3] - box[1]), 1e-9)
    if clipped / original < MIN_BOX_RETAINED:
        return None
    return (
        ((x0 + x1) / 2 - area[0]) / aw,
        ((y0 + y1) / 2 - area[1]) / ah,
        (x1 - x0) / aw,
        (y1 - y0) / ah,
    )


# --- build ------------------------------------------------------------------

def build(ann_dir: Path, manifest_path: Path, out_root: Path,
          tag: str = "v1", negatives: bool = True) -> dict:
    """Export every verified sheet to a YOLO dataset."""
    from . import manifest as manifest_mod

    records = {r["sha256"][:8]: r for r in manifest_mod.load(manifest_path)}
    sheets = [s for s in annotate.load_all(ann_dir).values() if s.status == "verified"]
    sheets.sort(key=lambda s: (s.project_id, s.page))

    kept, dupe_groups = deduplicate(sheets, records)

    root = out_root / tag
    images, labels = root / "images", root / "labels"
    for d in (images, labels, root / "crops", root / "preview"):
        d.mkdir(parents=True, exist_ok=True)

    examples: list[Example] = []
    negatives_made = 0
    dropped_boxes = 0
    class_counts: Counter = Counter()

    for sheet in kept:
        record = records.get(sheet.pdf_sha256[:8])
        if not record:
            continue
        by_area: dict[str, list] = defaultdict(list)
        for opening in sheet.openings:
            if opening.state == "accepted" and opening.area_id and opening.cls:
                by_area[opening.area_id].append(opening)

        with pymupdf.open(record["path"]) as doc:
            page = doc[sheet.page - 1]
            for area in sheet.areas:
                is_neg = area.type == "exclude"
                if is_neg and not negatives:
                    continue
                if not is_neg and area.type not in {"window_area", "door_area", "garage_area"}:
                    continue

                name = f"{sheet.sheet_key}_{area.id}"
                try:
                    width, height = _render_area(page, area.rect, images / f"{name}.png")
                except Exception:
                    continue
                if width < 64 or height < 64:
                    (images / f"{name}.png").unlink(missing_ok=True)
                    continue

                rows: list[tuple[int, float, float, float, float]] = []
                for opening in by_area.get(area.id, []):
                    folded = FOLD.get(opening.cls)
                    if folded is None:
                        continue
                    yolo = _to_yolo(opening.rect, area.rect)
                    if yolo is None:
                        dropped_boxes += 1
                        continue
                    rows.append((CLASS_INDEX[folded], *yolo))
                    class_counts[folded] += 1

                if is_neg:
                    rows = []          # background image: an empty label file
                    negatives_made += 1
                elif not rows:
                    (images / f"{name}.png").unlink(missing_ok=True)
                    continue

                (labels / f"{name}.txt").write_text(
                    "".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in rows)
                )
                examples.append(Example(
                    name=name, sheet_key=sheet.sheet_key, planset_key=sheet.planset_key,
                    project_id=sheet.project_id, area_id=area.id, area_type=area.type,
                    width=width, height=height, boxes=rows, is_negative=is_neg,
                ))

    _trim_negatives(examples, images, labels)
    report = _write_split(root, examples, class_counts, dupe_groups, dropped_boxes)
    _write_coco(root, examples)
    return report


def _trim_negatives(examples: list[Example], images: Path, labels: Path) -> None:
    """Keep background crops to a sensible share of the set.

    Too few and the model never learns to reject a title block; too many and the
    loss is dominated by empty images.
    """
    positives = [e for e in examples if not e.is_negative]
    negatives = [e for e in examples if e.is_negative]
    allowed = int(len(positives) * HARD_NEGATIVE_SHARE / (1 - HARD_NEGATIVE_SHARE))
    if len(negatives) <= allowed:
        return
    # Drop the smallest first -- a large excluded region shows more of the sheet
    # furniture the model needs to learn to ignore.
    negatives.sort(key=lambda e: e.width * e.height)
    for example in negatives[:len(negatives) - allowed]:
        (images / f"{example.name}.png").unlink(missing_ok=True)
        (labels / f"{example.name}.txt").unlink(missing_ok=True)
        examples.remove(example)


def _write_split(root: Path, examples: list[Example], class_counts: Counter,
                 dupe_groups: list[list[str]], dropped_boxes: int) -> dict:
    """Write train/val lists, dataset.yaml and the balance report."""
    train, val = [], []
    for example in examples:
        (val if is_val(example.planset_key) else train).append(example)

    # A project uploaded twice must not straddle the split -- the two files are
    # the same building.
    by_project: dict[str, set[str]] = defaultdict(set)
    for example in examples:
        by_project[example.project_id].add("val" if is_val(example.planset_key) else "train")
    straddling = [p for p, sides in by_project.items() if len(sides) > 1]

    for split, items in (("train", train), ("val", val)):
        # The leading "./" is required: ultralytics expands it to the txt file's
        # parent, and then derives the label path by swapping "/images/" for
        # "/labels/". Without it the separator is missing and no labels are found.
        (root / f"{split}.txt").write_text(
            "".join(f"./images/{e.name}.png\n" for e in items)
        )

    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASSES))
    (root / "dataset.yaml").write_text(
        f"# Exported from human-verified annotations.\n"
        f"# Split by planset (hash-stable), never by page: plansets repeat\n"
        f"# near-identical sheets, and a page-level split leaks.\n"
        f"path: {root.resolve()}\ntrain: train.txt\nval: val.txt\nnames:\n{names}\n"
    )

    def count(items):
        c = Counter()
        for e in items:
            for box in e.boxes:
                c[CLASSES[box[0]]] += 1
        return c

    train_c, val_c = count(train), count(val)
    warnings = []
    for name in CLASSES:
        if val_c[name] == 0:
            warnings.append(f"class '{name}' has no boxes in val — mAP for it is meaningless")
        if train_c[name] + val_c[name] < 50:
            warnings.append(f"class '{name}' has only {train_c[name] + val_c[name]} boxes — too rare to train")
    if straddling:
        warnings.append(f"project(s) {straddling} span both splits — leakage")
    if dropped_boxes:
        warnings.append(f"{dropped_boxes} box(es) dropped for falling outside their area crop")

    report = {
        "tag": root.name,
        "images": len(examples),
        "positives": sum(1 for e in examples if not e.is_negative),
        "negatives": sum(1 for e in examples if e.is_negative),
        "boxes": sum(len(e.boxes) for e in examples),
        "classes": dict(class_counts),
        "train": {"images": len(train), "boxes": sum(train_c.values()), "by_class": dict(train_c)},
        "val": {"images": len(val), "boxes": sum(val_c.values()), "by_class": dict(val_c)},
        "plansets": {
            "train": len({e.planset_key for e in train}),
            "val": len({e.planset_key for e in val}),
        },
        "duplicates_dropped": dupe_groups,
        "warnings": warnings,
    }
    (root / "report.json").write_text(json.dumps(report, indent=2))

    with (root / "index.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "split", "planset", "project", "area_type",
                         "width", "height", "boxes", "is_negative"])
        for e in examples:
            writer.writerow([f"images/{e.name}.png",
                             "val" if is_val(e.planset_key) else "train",
                             e.planset_key, e.project_id, e.area_type,
                             e.width, e.height, len(e.boxes), int(e.is_negative)])
    return report


def _write_coco(root: Path, examples: list[Example]) -> None:
    """COCO alongside YOLO, for tools that prefer it."""
    images, annos = [], []
    anno_id = 1
    for image_id, e in enumerate(examples, start=1):
        images.append({"id": image_id, "file_name": f"images/{e.name}.png",
                       "width": e.width, "height": e.height})
        for cls, cx, cy, w, h in e.boxes:
            annos.append({
                "id": anno_id, "image_id": image_id, "category_id": cls + 1,
                "bbox": [round((cx - w / 2) * e.width, 2), round((cy - h / 2) * e.height, 2),
                         round(w * e.width, 2), round(h * e.height, 2)],
                "area": round(w * e.width * h * e.height, 2), "iscrowd": 0,
            })
            anno_id += 1
    (root / "coco.json").write_text(json.dumps({
        "images": images, "annotations": annos,
        "categories": [{"id": i + 1, "name": n} for i, n in enumerate(CLASSES)],
    }))
