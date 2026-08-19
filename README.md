# Window & door extraction from architectural schedule sheets

Finds every window and door type on a planset's schedule sheets and returns
their coordinates, so a takeoff can be counted rather than read by hand.

## How it works

Two stages, split because they are genuinely different problems.

**Stage 1 — find the regions.** A schedule sheet says `WINDOW TYPES` and
`DOOR SCHEDULE` in plain text. We read those titles and grow a region under
each one. This is deterministic: no model, and exact when it fires. It works on
54 of 59 plansets; the 5 misses are scanned sheets with no text layer.

**Stage 2 — find the openings inside a region.** This needs a model, because
individual openings are line drawings with no text attached.

The leverage is that stage 1 answers the hard question for stage 2. The model
never has to decide *window or door* — it only finds shapes, and the region
they sit in supplies the class. Everything under `WINDOW TYPES` is a window.
That is also the one thing the incumbent detector reliably got wrong: it
labelled a block titled `DOOR TYPES` as windows.

## Repository layout

| Path | What |
|---|---|
| `schedext/` | the pipeline — locate, segment, detect, annotate, export, train, evaluate |
| `schedext/geom.py` | rotation-safe geometry. **Everything** goes through it |
| `dataset-creator/` | standalone web tool: drop in a PDF, get regions and openings to verify |
| `annotations/` | human-verified labels. The source of truth |
| `out/dataset/v1/` | the exported training set |
| `TRAINING.md` | how to train and what the numbers mean |

## Current state

The dataset is exported and verified; the detector is not yet trained.

| | |
|---|---|
| Verified sheets | 41 / 41, across 25 plansets |
| Training images | 90 — 77 with boxes, 13 deliberate background |
| Boxes | 579 — 327 door, 252 window |
| Split | 62 train / 28 val, **by planset**, hash-stable |
| Overfit gate | **passed** — mAP50 0.995 on 8 memorised images |

The incumbent `best.pt` on held-out plansets: class-agnostic mAP50 **0.407**,
precision 0.563, recall 0.571, exact opening count 18.5%. Per class it is
lopsided — door 0.457, window 0.160. That is the number to beat, and the full
measurement is in `TRAINING.md`.

## Setup

    python3.12 -m venv .venv
    .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu121
    .venv/bin/pip install -r requirements.txt
    .venv/bin/python -c "import torch; print(torch.cuda.is_available())"   # must be True

## Train

    .venv/bin/python -m schedext.cli train --arms all
    .venv/bin/python -m schedext.cli evaluate out/dataset/v1/runs/coco-s/weights/best.pt

`torch.cuda.is_available()` picks the device and the recipe adapts, so the same
command is correct on CPU and GPU. On a GPU all three arms take roughly two
hours; read `TRAINING.md` before changing any setting, since most of the
departures from ultralytics defaults are there for a measured reason.

## Re-exporting after more annotation

    .venv/bin/python -m schedext.cli export-dataset --tag v2
    .venv/bin/python -m schedext.eval.test_dataset v2

The annotator accumulates: `dataset-creator/run.sh` serves a UI that takes a
PDF, proposes regions and openings, and writes into the same `annotations/`
store the pipeline exports from.

## Note on data

`out/dataset/v1/` contains crops of client architectural drawings, committed
because they are the training set. The source plansets are **not** committed.
**Keep this repository private.**
