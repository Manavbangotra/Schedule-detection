"""Locate every window and door opening drawn on a schedule sheet (stage D).

This is `best.pt` used for the one thing it is genuinely good at. Measured on
hand-counted sheets, its localisation is reliable; its class labels are not, so
they are discarded here and the category comes from the schedule block's title
instead.

Two passes, because one is not enough and tiling everything is not affordable
on CPU (36s per 1600px tile -- a naive 12-tile sweep of 154 sheets is 7+ hours):

  A. one inference over the whole sheet, so openings outside any schedule block
     are still found and can be tagged as such;
  B. one inference per schedule block, which recovers the small pictograms the
     whole-sheet downsample loses. Blocks measure 800-1600px, so each fits a
     single tile and no tiling is needed at all.

Output is **leaf level**: one box per physical unit. A triple-wide mulled window
stays three boxes. Nothing is merged away unless ``group_mullions`` is asked for.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import pymupdf

from . import genre, locate, manifest as manifest_mod, segment

# Matches viewer.ZOOM so the rendered sheets under viewer/pages/ are reused
# rather than re-rendered. image_px = display_pt * ZOOM.
ZOOM = 1.8

SHEET_IMGSZ = 1600
BLOCK_IMGSZ = 1280
CONF = 0.20

# Two views of the same opening overlap heavily and have similar area.
DEDUP_IOU = 0.55
# A sash has small IoU with its parent but sits almost entirely inside it.
CONTAINMENT = 0.85

# A box belongs to a block when most of it lies inside.
IN_BLOCK = 0.70

# Mulled leaves sit a mullion apart; separate schedule entries sit far apart.
# Measured on project_486: 11px between leaves, 88-164px between entries.
MULLION_GAP_RATIO = 0.15
MULLION_Y_TOL = 0.10

CROP_ZOOM = 4.0
CROP_PAD_PT = 2.0


@dataclass
class Opening:
    opening_id: str
    project_id: str
    file: str
    page: int
    sheet_name: str
    rotation: int
    x0_px: float
    y0_px: float
    x1_px: float
    y1_px: float
    width_px: float
    height_px: float
    x0_pt: float
    y0_pt: float
    x1_pt: float
    y1_pt: float
    confidence: float
    source: str          # "sheet" | "block"
    in_block: bool
    block_id: str
    block_title: str
    category: str
    group_id: str
    crop: str

    def to_dict(self) -> dict:
        return asdict(self)


# --- geometry ---------------------------------------------------------------

def _area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _intersection(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _iou(a, b) -> float:
    i = _intersection(a, b)
    union = _area(a) + _area(b) - i
    return i / union if union > 0 else 0.0


def dedup(boxes: list[tuple[list[float], float, str]],
          iou_threshold: float = DEDUP_IOU,
          containment: float = CONTAINMENT) -> list[tuple[list[float], float, str]]:
    """Collapse duplicate and nested detections. Two mechanisms, two problems.

    IoU-NMS removes the same opening seen by both passes. Containment then drops
    a box that sits inside a larger one -- the whole unit plus its own lites.
    Adjacent leaves survive both, deliberately: they are separate units.
    """
    kept: list[tuple[list[float], float, str]] = []
    for box, conf, source in sorted(boxes, key=lambda x: -x[1]):
        if all(_iou(box, k[0]) < iou_threshold for k in kept):
            kept.append((box, conf, source))

    out = []
    for index, (box, conf, source) in enumerate(kept):
        nested = any(
            _area(box) < _area(other[0])
            and _intersection(box, other[0]) / max(_area(box), 1e-9) >= containment
            for j, other in enumerate(kept) if j != index
        )
        if not nested:
            out.append((box, conf, source))
    return out


def group_mullions(boxes: list[list[float]]) -> list[int]:
    """Assign a group id per box, joining leaves of one mulled assembly.

    Only ever used for reporting unless ``--group-mullions`` is passed; the
    default output stays leaf level.
    """
    order = sorted(range(len(boxes)), key=lambda i: boxes[i][0])
    group = [-1] * len(boxes)
    current = 0
    for position, index in enumerate(order):
        if group[index] == -1:
            group[index] = current
            current += 1
        if position + 1 >= len(order):
            continue
        nxt = order[position + 1]
        a, b = boxes[index], boxes[nxt]
        gap = b[0] - a[2]
        width = max(a[2] - a[0], b[2] - b[0], 1.0)
        height_a, height_b = a[3] - a[1], b[3] - b[1]
        aligned = abs(height_a - height_b) <= MULLION_Y_TOL * max(height_a, height_b, 1.0)
        overlap_y = _intersection([a[0], a[1], a[0] + 1, a[3]],
                                  [a[0], b[1], a[0] + 1, b[3]]) > 0
        if 0 <= gap <= MULLION_GAP_RATIO * width and aligned and overlap_y:
            group[nxt] = group[index]
    return group


# --- detection --------------------------------------------------------------

def _load(weights: str):
    from ultralytics import YOLO

    return YOLO(weights)


def _predict(model, image, imgsz: int, conf: float) -> list[tuple[list[float], float]]:
    result = model.predict(image, imgsz=imgsz, conf=conf, verbose=False)[0]
    return [
        ([float(v) for v in box], float(score))
        for box, score in zip(result.boxes.xyxy, result.boxes.conf)
    ]


def detect_sheet(model, page: pymupdf.Page, blocks: list[segment.Block],
                 zoom: float, conf: float) -> list[tuple[list[float], float, str]]:
    """Run both passes over one sheet; return boxes in sheet-pixel space."""
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
    sheet = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)

    found: list[tuple[list[float], float, str]] = [
        (box, conf_, "sheet") for box, conf_ in _predict(model, sheet, SHEET_IMGSZ, conf)
    ]

    for block in blocks:
        rect = block.rect
        crop_box = (
            max(0, int(rect.x0 * zoom)), max(0, int(rect.y0 * zoom)),
            min(pixmap.width, int(rect.x1 * zoom)), min(pixmap.height, int(rect.y1 * zoom)),
        )
        if crop_box[2] - crop_box[0] < 64 or crop_box[3] - crop_box[1] < 64:
            continue
        crop = sheet.crop(crop_box)
        for box, score in _predict(model, crop, BLOCK_IMGSZ, conf):
            found.append((
                [box[0] + crop_box[0], box[1] + crop_box[1],
                 box[2] + crop_box[0], box[3] + crop_box[1]],
                score, "block",
            ))

    return dedup(found)


def _tag(box: list[float], blocks: list[segment.Block], zoom: float):
    """Which schedule block owns this box. Smallest containing block wins."""
    best = None
    for block in blocks:
        rect = [block.rect.x0 * zoom, block.rect.y0 * zoom,
                block.rect.x1 * zoom, block.rect.y1 * zoom]
        share = _intersection(box, rect) / max(_area(box), 1e-9)
        if share >= IN_BLOCK and (best is None or _area(rect) < _area(best[1])):
            best = (block, rect)
    return best[0] if best else None


# --- driver -----------------------------------------------------------------

def run(manifest_path: Path, candidates_path: Path, out_dir: Path,
        weights: str, limit: int | None = None, conf: float = CONF,
        zoom: float = ZOOM, crops: bool = True,
        do_group: bool = False) -> dict:
    """Detect openings across the located schedule sheets."""
    records = {Path(r["path"]).name: r for r in manifest_mod.load(manifest_path)}
    with candidates_path.open() as handle:
        located = [json.loads(line) for line in handle if line.strip()]

    model = _load(weights)
    crop_dir = out_dir / "crops"
    if crops:
        crop_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    openings: list[Opening] = []
    per_sheet: list[dict] = []
    sheets_done = 0

    for entry in located:
        record = records.get(entry["file"])
        if not record or not entry["candidates"]:
            continue
        if record["health"] in {"truncated", "unopenable"}:
            continue

        with pymupdf.open(record["path"]) as doc:
            for candidate in entry["candidates"]:
                if limit is not None and sheets_done >= limit:
                    break
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
                    blocks = []

                try:
                    boxes = detect_sheet(model, page, blocks, zoom, conf)
                except Exception as exc:
                    per_sheet.append({"file": entry["file"], "page": page_no,
                                      "error": f"{type(exc).__name__}: {exc}"[:160]})
                    sheets_done += 1
                    continue

                groups = group_mullions([b[0] for b in boxes]) if boxes else []
                sheet_rows: list[Opening] = []

                for n, (box, score, source) in enumerate(sorted(boxes, key=lambda b: (b[0][1], b[0][0]))):
                    block = _tag(box, blocks, zoom)
                    oid = f"{record['project_id']}_p{page_no}_{n:03d}"
                    crop_path = ""
                    if crops:
                        crop_path = _write_crop(page, box, zoom, crop_dir / f"{oid}.png")

                    sheet_rows.append(Opening(
                        opening_id=oid,
                        project_id=record["project_id"],
                        file=entry["file"],
                        page=page_no,
                        sheet_name=candidate.get("sheet_name", ""),
                        rotation=candidate.get("rotation", 0),
                        x0_px=round(box[0], 1), y0_px=round(box[1], 1),
                        x1_px=round(box[2], 1), y1_px=round(box[3], 1),
                        width_px=round(box[2] - box[0], 1),
                        height_px=round(box[3] - box[1], 1),
                        x0_pt=round(box[0] / zoom, 2), y0_pt=round(box[1] / zoom, 2),
                        x1_pt=round(box[2] / zoom, 2), y1_pt=round(box[3] / zoom, 2),
                        confidence=round(score, 3),
                        source=source,
                        in_block=block is not None,
                        block_id=block.block_id if block else "",
                        block_title=block.title if block else "",
                        category=block.category if block else "unclassified",
                        group_id=f"{oid[:-4]}_g{groups[n]}" if groups else "",
                        crop=str(crop_path),
                    ))

                openings.extend(sheet_rows)
                per_sheet.append({
                    "file": entry["file"], "project_id": record["project_id"],
                    "page": page_no, "blocks": len(blocks),
                    "openings": len(sheet_rows),
                    "in_block": sum(1 for r in sheet_rows if r.in_block),
                })
                sheets_done += 1
        if limit is not None and sheets_done >= limit:
            break

    _write(out_dir, openings, per_sheet)
    return {
        "sheets": sheets_done,
        "openings": len(openings),
        "in_block": sum(1 for o in openings if o.in_block),
        "outside": sum(1 for o in openings if not o.in_block),
        "by_source": {
            "sheet": sum(1 for o in openings if o.source == "sheet"),
            "block": sum(1 for o in openings if o.source == "block"),
        },
    }


def _write_crop(page: pymupdf.Page, box: list[float], zoom: float, path: Path) -> str:
    """Render the opening from the PDF, not upscaled from the sheet JPEG.

    The clip goes in **display** space. Rendering and text extraction disagree
    about coordinates: get_pixmap(clip=) wants display, get_text(clip=) wants
    unrotated. Verified on a 270-degree page -- passing an extraction rect here
    returns a transposed crop of the wrong region.
    """
    display = pymupdf.Rect(box[0] / zoom, box[1] / zoom, box[2] / zoom, box[3] / zoom)
    display = pymupdf.Rect(display.x0 - CROP_PAD_PT, display.y0 - CROP_PAD_PT,
                           display.x1 + CROP_PAD_PT, display.y1 + CROP_PAD_PT) & page.rect
    if display.is_empty:
        return ""
    try:
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(CROP_ZOOM, CROP_ZOOM),
                                 clip=display)
        pixmap.save(path)
    except Exception:
        return ""
    return str(path)


def _write(out_dir: Path, openings: list[Opening], per_sheet: list[dict]) -> None:
    fields = list(Opening.__dataclass_fields__)
    with (out_dir / "openings.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for opening in openings:
            writer.writerow(opening.to_dict())

    by_sheet: dict[tuple[str, int], list[dict]] = {}
    for opening in openings:
        by_sheet.setdefault((opening.file, opening.page), []).append(opening.to_dict())
    with (out_dir / "openings.jsonl").open("w") as handle:
        for (file, page), rows in by_sheet.items():
            handle.write(json.dumps({"file": file, "page": page,
                                     "openings": rows}) + "\n")

    (out_dir / "detect_summary.json").write_text(json.dumps(per_sheet, indent=1))


# --- parallel driver --------------------------------------------------------
#
# Measured on this box: one torch thread runs a 1280px inference in 7.4s and
# four threads take 8.0s. YOLOv12's attention blocks simply do not scale across
# cores here, which makes process-level parallelism nearly free -- three
# single-threaded workers give ~3x throughput, turning a 2.5 hour corpus run
# into about 35 minutes.

_WORKER: dict = {}


def _init_worker(weights: str, zoom: float, conf: float, crops: bool,
                 crop_dir: str) -> None:
    """One model per process, loaded once, pinned to a single thread."""
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)

    _WORKER["model"] = _load(weights)
    _WORKER["zoom"] = zoom
    _WORKER["conf"] = conf
    _WORKER["crops"] = crops
    _WORKER["crop_dir"] = Path(crop_dir)


def _process_sheet(task: dict) -> dict:
    """Detect one sheet. Runs in a worker process, so it returns plain dicts."""
    model = _WORKER["model"]
    zoom, conf = _WORKER["zoom"], _WORKER["conf"]

    try:
        with pymupdf.open(task["path"]) as doc:
            page = doc[task["page_index"]]
            page_no = task["page_index"] + 1

            try:
                blocks = genre.annotate(page, segment.segment(
                    page,
                    locate.PageCandidate(
                        page_index=task["page_index"],
                        score=task["score"],
                        signals=task["signals"],
                        titles=[locate.TitleHit(**t) for t in task["titles"]],
                    ),
                ))
            except Exception:
                blocks = []

            boxes = detect_sheet(model, page, blocks, zoom, conf)
            groups = group_mullions([b[0] for b in boxes]) if boxes else []

            rows = []
            ordered = sorted(boxes, key=lambda b: (b[0][1], b[0][0]))
            for n, (box, score, source) in enumerate(ordered):
                block = _tag(box, blocks, zoom)
                oid = f"{task['project_id']}_p{page_no}_{n:03d}"
                crop_path = ""
                if _WORKER["crops"]:
                    crop_path = _write_crop(
                        page, box, zoom, _WORKER["crop_dir"] / f"{oid}.png")
                rows.append(Opening(
                    opening_id=oid, project_id=task["project_id"],
                    file=task["file"], page=page_no,
                    sheet_name=task.get("sheet_name", ""),
                    rotation=task.get("rotation", 0),
                    x0_px=round(box[0], 1), y0_px=round(box[1], 1),
                    x1_px=round(box[2], 1), y1_px=round(box[3], 1),
                    width_px=round(box[2] - box[0], 1),
                    height_px=round(box[3] - box[1], 1),
                    x0_pt=round(box[0] / zoom, 2), y0_pt=round(box[1] / zoom, 2),
                    x1_pt=round(box[2] / zoom, 2), y1_pt=round(box[3] / zoom, 2),
                    confidence=round(score, 3), source=source,
                    in_block=block is not None,
                    block_id=block.block_id if block else "",
                    block_title=block.title if block else "",
                    category=block.category if block else "unclassified",
                    group_id=f"{oid[:-4]}_g{groups[n]}" if groups else "",
                    crop=str(crop_path),
                ).to_dict())

            return {"file": task["file"], "project_id": task["project_id"],
                    "page": page_no, "blocks": len(blocks),
                    "openings": len(rows),
                    "in_block": sum(1 for r in rows if r["in_block"]),
                    "rows": rows}
    except Exception as exc:
        return {"file": task["file"], "page": task["page_index"] + 1,
                "error": f"{type(exc).__name__}: {exc}"[:160], "rows": []}


def run_parallel(manifest_path: Path, candidates_path: Path, out_dir: Path,
                 weights: str, workers: int = 3, limit: int | None = None,
                 conf: float = CONF, zoom: float = ZOOM,
                 crops: bool = True) -> dict:
    """Same as :func:`run`, spread across worker processes."""
    import multiprocessing as mp

    records = {Path(r["path"]).name: r for r in manifest_mod.load(manifest_path)}
    with candidates_path.open() as handle:
        located = [json.loads(line) for line in handle if line.strip()]

    tasks: list[dict] = []
    for entry in located:
        record = records.get(entry["file"])
        if not record or record["health"] in {"truncated", "unopenable"}:
            continue
        for candidate in entry["candidates"]:
            tasks.append({
                "path": record["path"], "file": entry["file"],
                "project_id": record["project_id"],
                "page_index": candidate["page_index"],
                "score": candidate["score"], "signals": candidate["signals"],
                "titles": candidate["titles"],
                "sheet_name": candidate.get("sheet_name", ""),
                "rotation": candidate.get("rotation", 0),
            })
    if limit is not None:
        tasks = tasks[:limit]

    crop_dir = out_dir / "crops"
    out_dir.mkdir(parents=True, exist_ok=True)
    if crops:
        crop_dir.mkdir(parents=True, exist_ok=True)

    context = mp.get_context("spawn")
    openings: list[dict] = []
    per_sheet: list[dict] = []
    done = 0

    with context.Pool(workers, initializer=_init_worker,
                      initargs=(weights, zoom, conf, crops, str(crop_dir))) as pool:
        for result in pool.imap_unordered(_process_sheet, tasks, chunksize=1):
            rows = result.pop("rows", [])
            openings.extend(rows)
            per_sheet.append(result)
            done += 1
            if done % 10 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} sheets · {len(openings)} openings",
                      flush=True)

    openings.sort(key=lambda r: (r["project_id"], r["page"], r["opening_id"]))
    _write_rows(out_dir, openings, per_sheet)
    return {
        "sheets": len(per_sheet),
        "openings": len(openings),
        "in_block": sum(1 for o in openings if o["in_block"]),
        "outside": sum(1 for o in openings if not o["in_block"]),
        "errors": sum(1 for s in per_sheet if s.get("error")),
        "by_source": {
            "sheet": sum(1 for o in openings if o["source"] == "sheet"),
            "block": sum(1 for o in openings if o["source"] == "block"),
        },
    }


def _write_rows(out_dir: Path, rows: list[dict], per_sheet: list[dict]) -> None:
    """Write CSV/JSONL from plain dicts (what the workers hand back)."""
    fields = list(Opening.__dataclass_fields__)
    with (out_dir / "openings.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    by_sheet: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        by_sheet.setdefault((row["file"], row["page"]), []).append(row)
    with (out_dir / "openings.jsonl").open("w") as handle:
        for (file, page), group in by_sheet.items():
            handle.write(json.dumps({"file": file, "page": page,
                                     "openings": group}) + "\n")

    (out_dir / "detect_summary.json").write_text(json.dumps(per_sheet, indent=1))
