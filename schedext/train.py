"""Train the stage-2 opening detector.

Portable and device-agnostic -- a plain script, not a notebook, because the GPU
is on another machine. `torch.cuda.is_available()` picks the device.

Three arms, identical data and split, because the obvious question ("finetune
best.pt or not?") confounds two variables:

    A1  best.pt         26.4M   the incumbent
    A2  yolo12l COCO    26.4M   isolates the facade training
    A3  yolo11s COCO     9.4M   isolates the capacity

best.pt was itself trained from yolo12l.pt (COCO) for 100 epochs on facade
photographs, so it is COCO plus photographic drift rather than a separate
lineage. A1 vs A2 tells you whether that drift helped; A2 vs A3 tells you
whether the size hurts. A1 vs A3 alone would tell you neither.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _find_best_pt() -> str:
    """Locate the incumbent checkpoint on whichever box this is.

    It lives outside the repo because it is 53MB of gitignored weights, so its
    path is machine-specific: /var/www on the original Linux box, a sibling
    checkout of WindowsDoorsClassification anywhere else. Checked in that order
    after an explicit override, so neither box has to edit this file.
    """
    linux = "/var/www/html/WindowsDoorsClassification/best.pt"
    sibling = Path(__file__).resolve().parents[2] / "WindowsDoorsClassification" / "best.pt"
    for candidate in (os.environ.get("SCHEDEXT_BEST_PT"), linux, sibling):
        if candidate and Path(candidate).exists():
            return str(candidate)
    return linux            # nothing found: name the canonical path in the error


BEST_PT = _find_best_pt()

ARMS = {
    "best": {"weights": BEST_PT, "note": "incumbent, YOLOv12-L 26.4M"},
    "coco-l": {"weights": "yolo12l.pt", "note": "COCO at the same size as best.pt"},
    "coco-s": {"weights": "yolo11s.pt", "note": "COCO small — recommended"},
}

# Departures from ultralytics defaults, each for a measured reason.
RECIPE = dict(
    imgsz=1024,        # median box is 5.7% of its crop -> 58px here, 36px at 640
    epochs=400,
    patience=60,
    batch=8,
    # THE IMPORTANT ONE. Default nbs=64 with batch=8 gives accumulate=8, i.e.
    # ~2 optimizer steps per epoch on this dataset. Training would look like it
    # ran and learn almost nothing.
    nbs=8,
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    cos_lr=True,
    warmup_epochs=5,
    weight_decay=0.0005,
    # Line art is black on white: hue and saturation jitter augment nothing.
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.2,         # stroke darkness does vary by PDF producer
    # Drawings are axis-aligned. A rotation re-fits the axis-aligned box around
    # the rotated shape, which inflates it -- it corrupts the labels.
    degrees=0.0,
    shear=0.0,
    perspective=0.0,
    # Door pictograms are chiral: swing arcs, hinge marks, sliding arrows.
    fliplr=0.0,
    flipud=0.0,
    scale=0.4,         # crops render at fixed zoom, so scale carries signal
    translate=0.1,
    mosaic=0.5,        # strongest regulariser available, but it halves object size
    close_mosaic=40,
    mixup=0.0,
    cutmix=0.0,
    copy_paste=0.0,
    deterministic=True,
    seed=0,
    plots=True,
    val=True,
    # Crops run up to 9117px wide. Each dataloader worker holds decoded copies,
    # and the ultralytics default of 8 drove this box into swap before the
    # trainer had even started. Raise it on a machine with the RAM for it.
    workers=2,
)

# This box is a 4-core i5-4590 with no CUDA and no AVX-512. Measured on the real
# dataset, 62 train images: yolo11s at imgsz 1024 costs 150s/epoch, at 768 76s,
# at 640 53s; yolo11n at 1024 costs 71s. So full-quality yolo11s is an overnight
# run and the 26M-parameter arms (~4x the compute) are not worth starting.
#
# The schedule is shorter and runs to completion rather than longer with early
# stopping. `close_mosaic` is scheduled against the *planned* epoch count, so a
# 400-epoch plan closes mosaic at 360 -- and a run that early-stops at ~200,
# which 62 images will, never reaches the clean phase at all and finishes with
# mosaic still corrupting object scale. Better a schedule that always lands.
CPU_OVERRIDES = dict(
    epochs=300,
    patience=300,      # effectively off: completion is what guarantees the
                       # 40 clean epochs below actually happen
    close_mosaic=40,
)


def _repath(data: Path) -> Path:
    """Point the yaml's ``path:`` at wherever the dataset actually is now.

    The exporter writes an absolute root, which is correct on the machine that
    exported it and wrong on the GPU box the dataset gets copied to. Rewriting
    it here means the bundle is portable and nobody has to remember to edit a
    yaml before training.
    """
    data = Path(data).resolve()
    root = str(data.parent)
    lines = data.read_text().split("\n")
    fixed = [f"path: {root}" if l.startswith("path:") else l for l in lines]
    if fixed != lines:
        data.write_text("\n".join(fixed))
        print(f"  repointed dataset root -> {root}")
    return data


def _device() -> str:
    import torch

    return "0" if torch.cuda.is_available() else "cpu"


def overfit_gate(data: Path, weights: str = "yolo11s.pt") -> bool:
    """Train on a handful of images and check the loss collapses.

    If a detector cannot memorise eight images, the data pipeline is wrong --
    coordinates, normalisation, class indices -- and no amount of real training
    will reveal which. Minutes to run, and it fails loudly instead of quietly.
    """
    from ultralytics import YOLO

    root = data.parent
    tiny = root / "_overfit"
    tiny.mkdir(exist_ok=True)
    # Absolute paths here: the overfit lists live one directory deeper, so a
    # "./" prefix would resolve against _overfit/ instead of the dataset root.
    lines = [str((root / l.lstrip("./")).resolve())
             for l in (root / "train.txt").read_text().split() if l][:8]
    (tiny / "train.txt").write_text("\n".join(lines) + "\n")
    (tiny / "val.txt").write_text("\n".join(lines) + "\n")
    # Paths stay RELATIVE to `path`. Ultralytics joins the two, so an absolute
    # train/val here produces <root>/<root>/... and the dataset "disappears".
    (tiny / "dataset.yaml").write_text(
        (root / "dataset.yaml").read_text()
        .replace("train: train.txt", f"train: {tiny.name}/train.txt")
        .replace("val: val.txt", f"val: {tiny.name}/val.txt")
    )

    model = YOLO(weights)
    result = model.train(
        data=str((tiny / "dataset.yaml").resolve()), epochs=100, imgsz=640, batch=4, nbs=4,
        workers=0,
        optimizer="AdamW", lr0=0.001, mosaic=0.0, hsv_h=0, hsv_s=0, hsv_v=0,
        fliplr=0.0, degrees=0.0, scale=0.0, translate=0.0, val=True, plots=False,
        project=str((root / "runs").resolve()), name="overfit", exist_ok=True,
        device=_device(), verbose=False,
    )
    m = result.results_dict if hasattr(result, "results_dict") else {}
    mAP = float(m.get("metrics/mAP50(B)", 0.0))
    print(f"  overfit gate: mAP50 on its own 8 training images = {mAP:.3f}")
    if mAP < 0.80:
        print("  FAIL — cannot memorise 8 images. The data pipeline is wrong,")
        print("         not the model. Check coordinates and class indices before training.")
        return False
    print("  PASS")
    return True


def train_arm(arm: str, data: Path, out: Path, **overrides) -> dict:
    from ultralytics import YOLO

    spec = ARMS[arm]
    print(f"\n=== arm '{arm}' — {spec['note']} ===")
    # Fail here, not four hours in: a bare name like "yolo11s.pt" gets
    # downloaded by ultralytics, but a path that does not exist is a dead
    # machine-specific checkpoint and every later error would be about
    # something else.
    src = spec["weights"]
    if ("/" in src or "\\" in src) and not Path(src).exists():
        raise FileNotFoundError(f"arm '{arm}' needs weights at {src}")
    model = YOLO(src)
    cfg = dict(RECIPE)
    if _device() == "cpu":
        cfg.update(CPU_OVERRIDES)
    cfg.update(overrides)
    hours = cfg["epochs"] * (150 if cfg["imgsz"] >= 1024 else 76) / 3600
    if _device() == "cpu":
        print(f"  CPU schedule: {cfg['epochs']} epochs at imgsz {cfg['imgsz']}"
              f"  ~{hours:.0f}h (measured on this machine)")
    result = model.train(
        data=str(Path(data).resolve()), project=str(Path(out).resolve()),
        name=arm, exist_ok=True,
        device=_device(), verbose=True, **cfg,
    )
    metrics = getattr(result, "results_dict", {}) or {}
    summary = {
        "arm": arm,
        "weights": spec["weights"],
        "mAP50": round(float(metrics.get("metrics/mAP50(B)", 0)), 4),
        "mAP50_95": round(float(metrics.get("metrics/mAP50-95(B)", 0)), 4),
        "precision": round(float(metrics.get("metrics/precision(B)", 0)), 4),
        "recall": round(float(metrics.get("metrics/recall(B)", 0)), 4),
        "best": str(Path(out) / arm / "weights" / "best.pt"),
    }
    print(f"  {summary}")
    return summary


def main(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="schedext.train")
    parser.add_argument("--tag", default="v1")
    parser.add_argument("--arms", default="coco-s",
                        help="comma-separated: best,coco-l,coco-s  (or 'all')")
    parser.add_argument("--skip-gate", action="store_true")
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    # Pass-throughs. All default to None so RECIPE stays the documented recipe
    # for anyone who passes nothing; these only exist so a run can be adapted to
    # the machine without editing the recipe and invalidating the comparison.
    parser.add_argument("--patience", type=int, default=None,
                        help="close_mosaic is scheduled against the PLANNED "
                             "epoch count, so a run that early-stops never "
                             "reaches the clean phase. Set == epochs to land it")
    parser.add_argument("--batch", type=int, default=None,
                        help="lower it if VRAM is short; leave nbs alone")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--cache", action="store_true",
                        help="hold images in RAM after the resize to imgsz — "
                             "these crops run to 38MP and decoding dominates")
    args = parser.parse_args(argv)

    root = Path("out/dataset") / args.tag
    data = root / "dataset.yaml"
    if not data.exists():
        print(f"no dataset at {data} — run `cli export-dataset` first")
        return 1

    data = _repath(data)
    print(f"device: {_device()}   dataset: {data}")
    if not args.skip_gate and not overfit_gate(data):
        return 1

    overrides = {}
    if args.imgsz:
        overrides["imgsz"] = args.imgsz
    if args.epochs:
        overrides["epochs"] = args.epochs
    if args.patience:
        overrides["patience"] = args.patience
    if args.batch:
        overrides["batch"] = args.batch
    if args.workers is not None:
        overrides["workers"] = args.workers
    if args.cache:
        overrides["cache"] = True

    arms = list(ARMS) if args.arms == "all" else [a.strip() for a in args.arms.split(",")]
    results = [train_arm(a, data, root / "runs", **overrides) for a in arms if a in ARMS]

    (root / "runs" / "comparison.json").write_text(json.dumps(results, indent=2))
    print("\n=== comparison ===")
    print(f"{'arm':10s}{'mAP50':>9}{'mAP50-95':>10}{'P':>8}{'R':>8}")
    for r in sorted(results, key=lambda r: -r["mAP50"]):
        print(f"{r['arm']:10s}{r['mAP50']:9.3f}{r['mAP50_95']:10.3f}"
              f"{r['precision']:8.3f}{r['recall']:8.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
