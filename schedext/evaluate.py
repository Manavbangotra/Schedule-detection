"""Score a trained opening detector against the held-out plansets.

A 5-planset validation set is small enough that a single mAP number is a
coin flip, so this reports the *shape* of the performance instead:

  * class-agnostic mAP50 -- what the pipeline actually consumes, because
    stage 1 supplies the class from the area title, not the model
  * per-class AP with the val box count printed beside it, so a thin class
    is visibly untrustworthy rather than quietly averaged in
  * per-planset AP -- reveals "works on 4 plansets, fails on 1", which is the
    expected failure shape when a drafting style is unseen
  * exact-count rate per area -- the number takeoff actually needs, and far
    more stable than mAP at this sample size
  * false positives on the hard negatives -- a model with good mAP that fires
    on title blocks and notes columns is useless in the pipeline

AP is computed here rather than read off ultralytics because none of the
per-planset or class-agnostic breakdowns are available from its validator.
Standard VOC all-point interpolation at IoU 0.5, greedy highest-confidence
matching, one detection per ground-truth box.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

CLASSES = ["window", "door"]
IOU = 0.50
CONF = 0.25          # what the pipeline would actually run at
PREVIEW_LIMIT = 24


def class_map(names: dict) -> dict:
    """Fold a model's own class list onto ``window`` / ``door``.

    Lets any model be scored on the same footing: the 2-class detectors trained
    here, and the 17-class ``best.pt`` whose labels are ``Windows-casement``,
    ``Doors-flush``, ``Garage-door-metal`` and so on. Anything that names
    neither is dropped rather than guessed at.
    """
    out = {}
    for index, raw in names.items():
        name = str(raw).lower()
        if "window" in name or "sash" in name or "glazing" in name:
            out[int(index)] = 0
        elif "door" in name or "garage" in name:
            out[int(index)] = 1
    return out


# --- geometry ---------------------------------------------------------------

def _iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0 else 0.0


def _load_labels(path: Path, width: int, height: int):
    """YOLO label file -> [(cls, x0, y0, x1, y1)] in pixels."""
    out = []
    if not path.exists():
        return out
    for line in path.read_text().split("\n"):
        parts = line.split()
        if len(parts) != 5:
            continue
        c, cx, cy, w, h = int(parts[0]), *(float(v) for v in parts[1:])
        out.append((c, (cx - w / 2) * width, (cy - h / 2) * height,
                    (cx + w / 2) * width, (cy + h / 2) * height))
    return out


def merge_predictions(preds, expansion: float = 0.07):
    """Collapse the whole-unit-plus-every-sash boxes a raw detector emits.

    ``best.pt`` fires once for the window and again for each sash, lite or
    panel -- 57 boxes over 15 openings on one val area. It is only ever used
    behind this step, so scoring it without one measures a configuration
    nobody runs. Same union-find and same 7% expansion as
    ``geom.merge_overlapping``; confidence is the group max and the class is
    the highest-confidence member.
    """
    n = len(preds)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    grown = []
    for _, _, (x0, y0, x1, y1) in preds:
        dx, dy = (x1 - x0) * expansion, (y1 - y0) * expansion
        grown.append((x0 - dx, y0 - dy, x1 + dx, y1 + dy))
    for i in range(n):
        for j in range(i + 1, n):
            a, b = grown[i], grown[j]
            if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                parent[find(i)] = find(j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(preds[i])
    out = []
    for members in groups.values():
        conf, cls, _ = max(members, key=lambda m: m[0])
        boxes = [m[2] for m in members]
        out.append((conf, cls,
                    (min(b[0] for b in boxes), min(b[1] for b in boxes),
                     max(b[2] for b in boxes), max(b[3] for b in boxes))))
    return out


# --- average precision ------------------------------------------------------

def average_precision(records, n_truth: int) -> float:
    """VOC all-point AP from (confidence, is_true_positive) records."""
    if n_truth == 0:
        return float("nan")
    if not records:
        return 0.0
    records = sorted(records, key=lambda r: -r[0])
    tp = fp = 0
    points = []
    for _, hit in records:
        tp, fp = tp + bool(hit), fp + (not hit)
        points.append((tp / n_truth, tp / (tp + fp)))
    # Make precision monotonically non-increasing, then integrate over recall.
    best = 0.0
    smooth = []
    for recall, precision in reversed(points):
        best = max(best, precision)
        smooth.append((recall, best))
    smooth.reverse()
    ap = prev_recall = 0.0
    for recall, precision in smooth:
        ap += (recall - prev_recall) * precision
        prev_recall = recall
    return ap


def match(preds, truths, class_agnostic: bool):
    """Greedy highest-confidence matching. Returns (records, unmatched_truth)."""
    used = set()
    records = []
    for conf, cls, box in sorted(preds, key=lambda p: -p[0]):
        best_iou, best_j = 0.0, -1
        for j, (tcls, *tbox) in enumerate(truths):
            if j in used or (not class_agnostic and tcls != cls):
                continue
            value = _iou(box, tbox)
            if value > best_iou:
                best_iou, best_j = value, j
        if best_iou >= IOU and best_j >= 0:
            used.add(best_j)
            records.append((conf, True))
        else:
            records.append((conf, False))
    return records, len(truths) - len(used)


# --- main -------------------------------------------------------------------

def evaluate(weights: str, root: Path, conf: float = CONF,
             previews: bool = True, merge: bool = False) -> dict:
    from ultralytics import YOLO

    meta = {}
    with (root / "index.csv").open() as handle:
        for row in csv.DictReader(handle):
            meta[row["image"]] = row

    entries = [l for l in (root / "val.txt").read_text().split() if l]
    model = YOLO(weights)
    folded = class_map(model.names)
    if sorted(model.names.values()) != sorted(CLASSES):
        print(f"  folding {len(model.names)} model classes onto "
              f"{CLASSES} ({len(model.names) - len(folded)} dropped)\n")

    agnostic_records, agnostic_truth = [], 0
    per_class = {c: {"records": [], "truth": 0} for c in CLASSES}
    per_planset = defaultdict(lambda: {"records": [], "truth": 0})
    counts = []                    # (planset, area_type, n_truth, n_pred)
    negatives = {"areas": 0, "fired": 0, "boxes": 0}
    preview_dir = root / "eval_preview"
    if previews:
        preview_dir.mkdir(exist_ok=True)
    made = 0

    for rel in entries:
        rel = rel.lstrip("./")
        image = root / rel
        row = meta.get(rel, {})
        planset = row.get("planset", "?")
        width, height = int(row.get("width", 0)), int(row.get("height", 0))
        truths = _load_labels(root / "labels" / f"{image.stem}.txt", width, height)

        result = model.predict(str(image), imgsz=1024, conf=conf, verbose=False)[0]
        preds = [
            (float(c), folded[int(k)], tuple(float(v) for v in b))
            for c, k, b in zip(result.boxes.conf, result.boxes.cls, result.boxes.xyxy)
            if int(k) in folded
        ]
        if merge:
            preds = merge_predictions(preds)

        if row.get("is_negative") == "1":
            negatives["areas"] += 1
            negatives["fired"] += bool(preds)
            negatives["boxes"] += len(preds)
            continue

        counts.append((planset, row.get("area_type", "?"), len(truths), len(preds)))

        records, _ = match(preds, truths, class_agnostic=True)
        agnostic_records += records
        agnostic_truth += len(truths)
        per_planset[planset]["records"] += records
        per_planset[planset]["truth"] += len(truths)

        for index, name in enumerate(CLASSES):
            cls_truth = [t for t in truths if t[0] == index]
            cls_pred = [p for p in preds if p[1] == index]
            recs, _ = match(cls_pred, cls_truth, class_agnostic=False)
            per_class[name]["records"] += recs
            per_class[name]["truth"] += len(cls_truth)

        if previews and made < PREVIEW_LIMIT:
            _burn(image, preview_dir / image.name, truths, preds)
            made += 1

    tp = sum(1 for _, hit in agnostic_records if hit)
    report = {
        "weights": weights,
        "conf": conf,
        "merged": merge,
        "val_images": len(entries),
        "map50_class_agnostic": round(average_precision(agnostic_records, agnostic_truth), 4),
        "precision": round(tp / len(agnostic_records), 4) if agnostic_records else 0.0,
        "recall": round(tp / agnostic_truth, 4) if agnostic_truth else 0.0,
        "per_class": {
            name: {"ap50": round(average_precision(d["records"], d["truth"]), 4),
                   "val_boxes": d["truth"]}
            for name, d in per_class.items()
        },
        "per_planset": {
            key: {"ap50": round(average_precision(d["records"], d["truth"]), 4),
                  "val_boxes": d["truth"]}
            for key, d in sorted(per_planset.items())
        },
        "exact_count_rate": round(
            sum(1 for *_, t, p in counts if t == p) / len(counts), 4) if counts else 0.0,
        "within_one_rate": round(
            sum(1 for *_, t, p in counts if abs(t - p) <= 1) / len(counts), 4) if counts else 0.0,
        "hard_negatives": negatives,
        "counts": [{"planset": p, "area": a, "truth": t, "pred": n}
                   for p, a, t, n in counts],
    }
    return report


def _burn(src: Path, dst: Path, truths, preds) -> None:
    """Ground truth in green, predictions in the class colour, onto one image.

    PIL rather than pymupdf: these crops are PNGs, and pymupdf can only draw
    onto a PDF page.
    """
    from PIL import Image, ImageDraw

    image = Image.open(src).convert("RGB")
    scale = min(1.0, 2200 / max(image.size))
    if scale < 1.0:
        image = image.resize((int(image.width * scale), int(image.height * scale)))
    draw = ImageDraw.Draw(image)
    for _, x0, y0, x1, y1 in truths:
        draw.rectangle([x0 * scale, y0 * scale, x1 * scale, y1 * scale],
                       outline=(0, 170, 0), width=4)
    for conf, cls, box in preds:
        colour = (30, 110, 255) if cls == 0 else (230, 40, 30)
        x0, y0, x1, y1 = (v * scale for v in box)
        draw.rectangle([x0, y0, x1, y1], outline=colour, width=2)
        draw.text((x0 + 2, max(0, y0 - 11)), f"{CLASSES[cls]} {conf:.2f}", fill=colour)
    image.save(dst)


def render(report: dict) -> str:
    lines = [
        f"weights: {report['weights']}   conf: {report['conf']}   "
        f"{report['val_images']} val images"
        + ("   [sash/lite duplicates merged]" if report.get("merged") else ""),
        "",
        f"class-agnostic mAP50   {report['map50_class_agnostic']:.3f}"
        "     <- what the pipeline consumes; stage 1 supplies the class",
        f"  precision {report['precision']:.3f}   recall {report['recall']:.3f}",
        "",
        "per class",
    ]
    for name, d in report["per_class"].items():
        warn = "   (too few boxes to trust)" if d["val_boxes"] < 30 else ""
        lines.append(f"  {name:8s} AP50 {d['ap50']:.3f}   "
                     f"{d['val_boxes']} val boxes{warn}")
    lines += ["", "per planset   (one weak planset is the expected failure shape)"]
    for key, d in report["per_planset"].items():
        lines.append(f"  {key:22s} AP50 {d['ap50']:.3f}   {d['val_boxes']} boxes")
    neg = report["hard_negatives"]
    lines += [
        "",
        f"exact opening count   {report['exact_count_rate']:.1%} of areas"
        "     <- the number takeoff needs",
        f"within +/- 1          {report['within_one_rate']:.1%} of areas",
        "",
        f"hard negatives        {neg['fired']}/{neg['areas']} areas fired, "
        f"{neg['boxes']} false boxes",
    ]
    worst = sorted(report["counts"], key=lambda c: -abs(c["truth"] - c["pred"]))[:5]
    if worst and abs(worst[0]["truth"] - worst[0]["pred"]) > 0:
        lines += ["", "worst count errors"]
        for c in worst:
            lines.append(f"  {c['planset']:22s} {c['area']:12s} "
                         f"truth {c['truth']:3d}  pred {c['pred']:3d}")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="schedext.evaluate")
    parser.add_argument("weights")
    parser.add_argument("--tag", default="v1")
    parser.add_argument("--conf", type=float, default=CONF)
    parser.add_argument("--no-previews", action="store_true")
    parser.add_argument("--merge", action="store_true",
                        help="collapse sash/lite duplicates first — how a raw "
                             "multi-class detector like best.pt is actually used")
    args = parser.parse_args(argv)

    root = Path("out/dataset") / args.tag
    if not (root / "val.txt").exists():
        print(f"no dataset at {root} — run `cli export-dataset` first")
        return 1

    report = evaluate(args.weights, root, args.conf,
                      not args.no_previews, args.merge)
    name = Path(args.weights).stem + ("_merged" if args.merge else "")
    (root / f"eval_{name}.json").write_text(json.dumps(report, indent=2))
    print(render(report))
    print(f"\nwrote {root / f'eval_{name}.json'}")
    if not args.no_previews:
        print(f"previews in {root / 'eval_preview'}/  (green = truth, "
              f"blue = window, red = door)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
