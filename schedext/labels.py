"""Generate YOLO detection labels from the text pipeline (stage D1).

The point of this module is that no one has to annotate anything. The text
layer already tells us, for every schedule sheet:

  * where each schedule block sits (segment.py), and
  * which category that block is, from its title (lexicon.category_of).

So we run the existing detector to get boxes -- its localisation is good, and
after merging, box counts matched the true item counts exactly on every region
tested -- then take the *class* from the block title rather than from the model.
That is precisely the thing the model gets wrong (it labelled a block titled
"DOOR TYPES" as windows), and precisely the thing the sheet states in plain
text. Boxes that fall outside a schedule block are dropped, which removes the
false positives it fires on unit-entry elevations.

The result is a 3-class dataset -- window / door / garage_door -- built to
finetune the existing weights for detection only.

Splitting is by planset, never by page: a set repeats near-identical sheets
across building types, so a page-level split would leak and inflate mAP.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from . import geom, genre, locate, manifest as manifest_mod, segment

# Classes for the finetune. Deliberately not the original 17: the type belongs
# to the text pipeline, and a flat class list cannot express "vinyl single
# hung" anyway.
CLASSES = ["window", "door", "garage_door"]
CLASS_INDEX = {name: index for index, name in enumerate(CLASSES)}

# Categories that are not one of the three trainable classes get mapped here,
# or dropped when there is no sensible home.
CATEGORY_ALIASES = {
    "window": "window",
    "door": "door",
    "garage_door": "garage_door",
    "storefront": "door",
    "both": None,      # ambiguous block; boxes would be unlabelled
    "louver": None,
    "shutter": None,
    "grille": None,
    "gate": None,
    "unknown": None,
}

# Render scale for the training crops. Schedule pictograms are illegible at the
# 640px the original model trained on, so crops are rendered generously and the
# finetune runs at a larger imgsz.
CROP_ZOOM = 2.0

# A detection must overlap the block by at least this much to be kept.
MIN_CONTAINMENT = 0.80


@dataclass
class LabelledCrop:
    """One training image: a rendered block plus its boxes."""

    project_id: str
    page: int
    block_id: str
    category: str
    image_path: Path
    label_path: Path
    boxes: list[tuple[int, float, float, float, float]]


def _load_detector(weights: str):
    from ultralytics import YOLO

    return YOLO(weights)


def _detect(model, image_path: Path, imgsz: int, conf: float) -> list[tuple]:
    result = model.predict(str(image_path), imgsz=imgsz, conf=conf, verbose=False)[0]
    return [tuple(float(v) for v in box) for box in result.boxes.xyxy]


def _containment(box: pymupdf.Rect, block: pymupdf.Rect) -> float:
    overlap = box & block
    area = box.width * box.height
    if area <= 0 or overlap.is_empty:
        return 0.0
    return (overlap.width * overlap.height) / area


def crops_for_page(page: pymupdf.Page, blocks: list[segment.Block],
                   out_dir: Path, project_id: str, page_no: int) -> list[tuple]:
    """Render one image per schedule block. Returns (block, path, origin)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for block in blocks:
        category = CATEGORY_ALIASES.get(block.category)
        if category is None or not block.is_parseable:
            continue
        # Display space, not extraction: get_pixmap and get_text take clips in
        # different coordinate systems, and 36 of 59 plansets are rotated.
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(CROP_ZOOM, CROP_ZOOM),
                                 clip=block.rect)
        if pixmap.width < 64 or pixmap.height < 64:
            continue
        path = out_dir / f"{project_id}_p{page_no}_{block.block_id}.png"
        pixmap.save(path)
        made.append((block, path, category))
    return made


def build(manifest_path: Path, candidates_path: Path, out_root: Path,
          weights: str, imgsz: int = 1600, conf: float = 0.20,
          limit: int | None = None) -> dict:
    """Produce a YOLO dataset from every located schedule sheet."""
    records = {Path(r["path"]).name: r for r in manifest_mod.load(manifest_path)}
    with candidates_path.open() as handle:
        found = {
            json.loads(line)["file"]: json.loads(line)["candidates"]
            for line in handle if line.strip()
        }

    model = _load_detector(weights)
    images_dir = out_root / "images"
    labels_dir = out_root / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    stats = {"crops": 0, "boxes": 0, "dropped_outside": 0, "per_class": {}}
    index: list[dict] = []
    processed = 0

    for name, record in records.items():
        candidates = found.get(name, [])
        if not candidates or record["health"] in {"truncated", "unopenable"}:
            continue
        if limit is not None and processed >= limit:
            break
        processed += 1

        with pymupdf.open(record["path"]) as doc:
            for candidate in candidates:
                page = doc[candidate["page_index"]]
                page_no = candidate["page_index"] + 1
                try:
                    blocks = genre.annotate(page, segment.segment(
                        page,
                        locate.PageCandidate(
                            page_index=candidate["page_index"],
                            score=candidate["score"],
                            signals=candidate["signals"],
                            titles=[locate.TitleHit(**t) for t in candidate["titles"]],
                        ),
                    ))
                except Exception:
                    continue

                for block, image_path, category in crops_for_page(
                    page, blocks, images_dir, record["project_id"], page_no
                ):
                    detections = _detect(model, image_path, imgsz, conf)
                    if not detections:
                        image_path.unlink(missing_ok=True)
                        continue

                    # Collapse the whole-unit-plus-every-sash duplicates the
                    # detector emits. Without this a single window contributes
                    # five boxes.
                    merged = geom.merge_overlapping(detections)

                    width = block.rect.width * CROP_ZOOM
                    height = block.rect.height * CROP_ZOOM
                    crop_rect = pymupdf.Rect(0, 0, width, height)
                    lines: list[str] = []
                    for box in merged:
                        if _containment(box, crop_rect) < MIN_CONTAINMENT:
                            stats["dropped_outside"] += 1
                            continue
                        cx = ((box.x0 + box.x1) / 2) / width
                        cy = ((box.y0 + box.y1) / 2) / height
                        bw = box.width / width
                        bh = box.height / height
                        if not (0 < bw <= 1 and 0 < bh <= 1):
                            continue
                        lines.append(
                            f"{CLASS_INDEX[category]} "
                            f"{cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
                        )

                    if not lines:
                        image_path.unlink(missing_ok=True)
                        continue

                    label_path = labels_dir / f"{image_path.stem}.txt"
                    label_path.write_text("\n".join(lines) + "\n")
                    stats["crops"] += 1
                    stats["boxes"] += len(lines)
                    stats["per_class"][category] = (
                        stats["per_class"].get(category, 0) + len(lines)
                    )
                    index.append(
                        {
                            "image": str(image_path),
                            "label": str(label_path),
                            "project_id": record["project_id"],
                            "page": page_no,
                            "block_id": block.block_id,
                            "block_title": block.title,
                            "category": category,
                            "boxes": len(lines),
                        }
                    )

    (out_root / "index.json").write_text(json.dumps(index, indent=2))
    _write_dataset_yaml(out_root, index)
    stats["plansets"] = len({e["project_id"] for e in index})
    return stats


def _write_dataset_yaml(out_root: Path, index: list[dict]) -> None:
    """Split by planset and write the ultralytics dataset descriptor."""
    projects = sorted({entry["project_id"] for entry in index})
    # Deterministic, and roughly 80/20 by planset rather than by page.
    holdout = {p for i, p in enumerate(projects) if i % 5 == 4}

    for split in ("train", "val"):
        (out_root / split).mkdir(parents=True, exist_ok=True)

    train_list, val_list = [], []
    for entry in index:
        target = val_list if entry["project_id"] in holdout else train_list
        target.append(entry["image"])

    (out_root / "train.txt").write_text("\n".join(train_list) + "\n")
    (out_root / "val.txt").write_text("\n".join(val_list) + "\n")

    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASSES))
    (out_root / "dataset.yaml").write_text(
        f"# Auto-generated from the text pipeline -- no manual annotation.\n"
        f"# Split by planset ({len(projects) - len(holdout)} train / "
        f"{len(holdout)} val) so near-identical sheets cannot leak.\n"
        f"path: {out_root.resolve()}\n"
        f"train: train.txt\n"
        f"val: val.txt\n"
        f"names:\n{names}\n"
    )
