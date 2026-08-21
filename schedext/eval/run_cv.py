"""Cross-validate one arm over the folds written by ``make_folds``.

Why this exists rather than a flag on ``schedext.train``: ``train.main`` derives
its dataset from ``out/dataset/<tag>/dataset.yaml`` and cannot be pointed at a
fold. ``train.train_arm`` takes ``data`` as a parameter, so the folds are driven
through it directly and the recipe stays untouched.

Each fold trains from the same COCO init -- never from the previous fold's
weights, which would leak fold k-1's val set into fold k's training.

    python -m schedext.eval.run_cv coco-s --tag v2 --imgsz 1024 --batch 2

Prints per-fold class-agnostic mAP50 / window AP50 / exact-count and their mean
and spread. The spread is the point: on 195 boxes a 0.01 difference is noise,
and only the fold-to-fold range shows you that.
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

FOLDS = 5


def main(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="schedext.eval.run_cv")
    parser.add_argument("arm", help="an entry in train.ARMS, e.g. coco-s")
    parser.add_argument("--tag", default="v2")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--folds", type=int, default=FOLDS)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args(argv)

    from .. import evaluate as ev
    from .. import train as tr

    root = Path("out/dataset") / args.tag
    cv = root / "cv"
    if not (cv / "fold0" / "dataset.yaml").exists():
        print(f"no folds at {cv} — run `python -m schedext.eval.make_folds {args.tag}` first")
        return 1

    rows = []
    for k in range(args.folds):
        data = cv / f"fold{k}" / "dataset.yaml"
        print(f"\n########## {args.arm} fold {k}/{args.folds - 1} ##########")
        weights = cv / f"fold{k}" / "runs" / args.arm / "weights" / "best.pt"
        # Resume, but only past a fold that actually FINISHED. Long runs on this
        # box get killed, and a killed run still leaves a best.pt -- of a model
        # that was still improving. Folding a truncated run into the mean would
        # quietly understate it, which is worse than retraining.
        if weights.exists() and _converged(weights, args.epochs, args.patience):
            print(f"  fold {k}: converged run present, skipping training")
        else:
            tr.train_arm(args.arm, data, cv / f"fold{k}" / "runs",
                         imgsz=args.imgsz, batch=args.batch, epochs=args.epochs,
                         patience=args.patience, workers=4, cache=True)
        if not weights.exists():
            print(f"  fold {k}: no weights produced, skipping")
            continue

        # Score this fold's own held-out plansets. `root` supplies index.csv and
        # labels/; the fold's val.txt is read via the list file below.
        report = _evaluate_fold(ev, str(weights), root, cv / f"fold{k}",
                               args.conf, args.imgsz)
        rows.append(report)
        print(f"  fold {k}: mAP50 {report['map50_class_agnostic']:.4f}  "
              f"window {report['per_class']['window']['ap50']:.4f}  "
              f"exact {report['exact_count_rate']:.3f}")

    if not rows:
        print("no folds completed")
        return 1

    print(f"\n=== {args.arm} @ imgsz {args.imgsz}: {len(rows)}-fold cross-validation ===")
    for label, get in (
        ("class-agnostic mAP50", lambda r: r["map50_class_agnostic"]),
        ("window AP50", lambda r: r["per_class"]["window"]["ap50"]),
        ("door AP50", lambda r: r["per_class"]["door"]["ap50"]),
        ("exact opening count", lambda r: r["exact_count_rate"]),
        ("within +/- 1", lambda r: r["within_one_rate"]),
    ):
        vals = [get(r) for r in rows]
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"  {label:22} {statistics.mean(vals):.4f} +/- {sd:.4f}"
              f"   (min {min(vals):.4f}, max {max(vals):.4f})")
    fp = sum(r["hard_negatives"]["boxes"] for r in rows)
    print(f"  {'false boxes total':22} {fp}  across all folds")

    out = cv / f"cv_{args.arm}_{args.imgsz}.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")
    return 0


def _converged(weights: Path, epochs: int, patience: int) -> bool:
    """Did this run stop on its own terms, or was it killed?

    Converged means either it ran the full schedule, or early stopping had cause
    to fire -- ``patience`` epochs elapsed with no improvement. A run cut short
    while still improving is neither, and its best.pt understates the fold.
    """
    results = weights.parent.parent / "results.csv"
    if not results.exists():
        return False
    best_epoch, last, best = 0, 0, -1.0
    with results.open() as handle:
        for row in csv.DictReader(handle):
            row = {k.strip(): v for k, v in row.items()}
            last = int(float(row["epoch"]))
            score = float(row["metrics/mAP50(B)"])
            if score > best:
                best, best_epoch = score, last
    return last >= epochs or (last - best_epoch) >= patience


def _evaluate_fold(ev, weights: str, root: Path, fold: Path,
                   conf: float, imgsz: int) -> dict:
    """Run the normal evaluator against a fold's val list.

    ``ev.evaluate`` reads ``root/val.txt``. Rather than duplicate its ~120 lines
    of AP and counting logic, point it at the fold by swapping that one file for
    the duration -- and always put the original back.
    """
    live = root / "val.txt"
    saved = live.read_text()
    # The fold lists hold ABSOLUTE paths, which ultralytics needs. evaluate.py
    # cannot take them: it parses with .split(), so "C:\Users\Manav Bangotra\..."
    # breaks at the space, and it also looks each entry up in index.csv by its
    # RELATIVE key. So convert back to "./images/name.png" here -- the exact form
    # the committed val.txt uses.
    rel = [f"./images/{Path(l).name}"
           for l in (fold / "val.txt").read_text().splitlines() if l.strip()]
    try:
        live.write_text("\n".join(rel) + "\n")
        return ev.evaluate(weights, root, conf, previews=False,
                           merge=False, augment=False, imgsz=imgsz)
    finally:
        live.write_text(saved)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
