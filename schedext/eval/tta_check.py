"""Does test-time augmentation help? Paired test over the CV folds.

Comparing two independent CV means cannot answer this: the fold-to-fold spread
on this dataset is +/- 0.085 mAP50, and TTA is worth low single digits at best.
A paired test can, because it scores the SAME weights on the SAME held-out
plansets twice and looks at the per-fold difference. The fold-to-fold variance
cancels; only the effect of augmentation remains.

    python -m schedext.eval.tta_check coco-s --tag v2

Reports the mean paired delta and how many folds moved in each direction. TTA is
only worth keeping if it wins on exact-count as well as mAP -- multi-scale passes
tend to emit more boxes, and more boxes is exactly how the opening count goes
wrong.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

FOLDS = 5


def main(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="schedext.eval.tta_check")
    parser.add_argument("arm", help="an entry in train.ARMS, e.g. coco-s")
    parser.add_argument("--tag", default="v2")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--folds", type=int, default=FOLDS)
    args = parser.parse_args(argv)

    from .. import evaluate as ev
    from .run_cv import _evaluate_fold

    root = Path("out/dataset") / args.tag
    cv = root / "cv"
    pairs = []

    for k in range(args.folds):
        weights = cv / f"fold{k}" / "runs" / args.arm / "weights" / "best.pt"
        if not weights.exists():
            print(f"  fold {k}: no weights, skipping")
            continue
        plain = _evaluate_fold(ev, str(weights), root, cv / f"fold{k}",
                               args.conf, args.imgsz)
        tta = _evaluate_tta(ev, str(weights), root, cv / f"fold{k}",
                            args.conf, args.imgsz)
        pairs.append((plain, tta))
        print(f"  fold {k}:  mAP50 {plain['map50_class_agnostic']:.4f} -> "
              f"{tta['map50_class_agnostic']:.4f}   "
              f"exact {plain['exact_count_rate']:.3f} -> "
              f"{tta['exact_count_rate']:.3f}")

    if not pairs:
        print("no folds to compare")
        return 1

    print(f"\n=== {args.arm}: TTA paired delta over {len(pairs)} folds ===")
    verdict_ok = True
    for label, key in (("class-agnostic mAP50", "map50_class_agnostic"),
                       ("exact opening count", "exact_count_rate"),
                       ("within +/- 1", "within_one_rate"),
                       ("precision", "precision"),
                       ("recall", "recall")):
        deltas = [t[key] - p[key] for p, t in pairs]
        mean = statistics.mean(deltas)
        sd = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
        wins = sum(1 for d in deltas if d > 0)
        print(f"  {label:22} {mean:+.4f} +/- {sd:.4f}   "
              f"({wins}/{len(deltas)} folds improved)")
        if key in ("map50_class_agnostic", "exact_count_rate") and mean <= 0:
            verdict_ok = False

    fp_delta = sum(t["hard_negatives"]["boxes"] - p["hard_negatives"]["boxes"]
                   for p, t in pairs)
    print(f"  {'false boxes':22} {fp_delta:+d}  across all folds")

    print("\n  verdict: " + (
        "KEEP TTA -- it wins on both mAP and exact-count"
        if verdict_ok and fp_delta <= 0 else
        "DROP TTA -- it does not win on both metrics that matter"))
    return 0


def _evaluate_tta(ev, weights: str, root: Path, fold: Path,
                  conf: float, imgsz: int) -> dict:
    """Same swap-and-restore as the plain path, with augment on."""
    live = root / "val.txt"
    saved = live.read_text()
    rel = [f"./images/{Path(l).name}"
           for l in (fold / "val.txt").read_text().splitlines() if l.strip()]
    try:
        live.write_text("\n".join(rel) + "\n")
        return ev.evaluate(weights, root, conf, previews=False,
                           merge=False, augment=True, imgsz=imgsz)
    finally:
        live.write_text(saved)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
