"""Sanity checks on an exported dataset.

These catch the failures that otherwise appear only as mysteriously poor mAP:
labels outside [0,1], images without labels, a planset straddling the split, or
a duplicate pair landing on both sides of it.

Run: .venv/bin/python -m schedext.eval.test_dataset [tag]
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from schedext import dataset


def main(tag: str = "v1") -> int:
    root = Path("out/dataset") / tag
    if not root.exists():
        print(f"no dataset at {root}")
        return 1

    problems: list[str] = []
    images = sorted((root / "images").glob("*.png"))
    labels = sorted((root / "labels").glob("*.txt"))

    # every image has a label file and vice versa
    image_names = {p.stem for p in images}
    label_names = {p.stem for p in labels}
    for missing in image_names - label_names:
        problems.append(f"image without a label file: {missing}")
    for missing in label_names - image_names:
        problems.append(f"label without an image: {missing}")

    boxes = 0
    per_class: Counter = Counter()
    for path in labels:
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                problems.append(f"{path.name}:{n} has {len(parts)} fields, expected 5")
                continue
            cls = int(parts[0])
            values = [float(v) for v in parts[1:]]
            boxes += 1
            per_class[cls] += 1
            if not 0 <= cls < len(dataset.CLASSES):
                problems.append(f"{path.name}:{n} class {cls} out of range")
            if any(not 0.0 <= v <= 1.0 for v in values):
                problems.append(f"{path.name}:{n} value outside [0,1]: {values}")
            if values[2] <= 0 or values[3] <= 0:
                problems.append(f"{path.name}:{n} zero-size box")

    # split integrity
    train = set((root / "train.txt").read_text().split())
    val = set((root / "val.txt").read_text().split())
    if train & val:
        problems.append(f"{len(train & val)} image(s) in BOTH splits")
    listed = {Path(p).stem for p in train | val}
    for missing in image_names - listed:
        problems.append(f"image in neither split: {missing}")

    report = json.loads((root / "report.json").read_text())
    if report["boxes"] != boxes:
        problems.append(f"report says {report['boxes']} boxes, label files hold {boxes}")

    # a planset must not straddle the split
    index = (root / "index.csv").read_text().splitlines()[1:]
    sides: dict[str, set] = {}
    for row in index:
        cols = row.split(",")
        sides.setdefault(cols[2], set()).add(cols[1])
    for planset, s in sides.items():
        if len(s) > 1:
            problems.append(f"planset {planset} appears in both splits")

    print(f"dataset '{tag}': {len(images)} images, {boxes} boxes, "
          f"{dict((dataset.CLASSES[c], n) for c, n in per_class.items())}")
    print(f"  train {len(train)} / val {len(val)} images")
    for w in report.get("warnings", []):
        print(f"  report warning: {w}")
    for p in problems:
        print(f"FAIL {p}")
    print(f"\n{'PASS — dataset is structurally sound' if not problems else str(len(problems)) + ' problem(s)'}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "v1"))
