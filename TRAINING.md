# Stage-2 opening detector — how to train it

Everything below runs from the repo root. The dataset is already exported and
verified; nothing here regenerates it unless you ask it to.

## What the dataset is

One image per verified schedule *area*, rendered from the PDF at zoom 4.0.
Every box inside a window area is `window`, every box inside a door area is
`door` — the class comes from the area title, which is the one thing the sheet
states in plain text and the one thing a detector gets wrong.

| | |
|---|---|
| Images | **90** — 77 with boxes, 13 background |
| Boxes | **579** — 327 door, 252 window |
| Train | 62 images / 383 boxes / 18 plansets |
| Val | 28 images / 196 boxes / 5 plansets |
| Duplicate sheets dropped | 4 |

Split is by **planset**, hash-stable (`sha256(key)[0] % 5 == 4`), never by page:
a set repeats near-identical sheets and a page split would leak. Four sheets
were dropped because the same drawing appears under three different project ids
(`543_p3 = 558_p1 = 559_p1`), which would otherwise put a drawing in train and
its twin in val.

The 13 background images are the **hard negatives** — title blocks, notes
columns, tabular schedules. Without them the model fires on all three the first
time it sees a full sheet.

## Regenerating (only if annotations changed)

    .venv/bin/python -m schedext.cli export-dataset --tag v1
    .venv/bin/python -m schedext.eval.test_dataset v1     # structural check

## Training

    .venv/bin/python -m schedext.cli train --arms all

`torch.cuda.is_available()` picks the device — the same command is correct on
CPU and GPU. Three arms run on identical data and split:

| Arm | Init | Params | What it isolates | Result |
|---|---|---|---|---|
| `best` | `best.pt` | 26.4M | the incumbent | not run |
| `coco-l` | `yolo12l.pt` | 26.4M | whether the facade training helped or hurt | **lost** |
| `coco-s` | `yolo11s.pt` | 9.4M | whether the capacity hurts | **WINNER** |

`best.pt` was itself trained from `yolo12l.pt` for 100 epochs on facade
photographs, so it is COCO *plus photographic drift*, not a separate lineage.
`best` vs `coco-l` tells you whether that drift helped; `coco-l` vs `coco-s`
tells you whether the size hurts. `best` vs `coco-s` alone would tell you
neither, because it moves both variables at once.

An **overfit gate** runs first: 8 images, 100 epochs, and it must reach
mAP50 ≥ 0.80 on its own training images. If a detector cannot memorise eight
images the data pipeline is wrong — coordinates, normalisation, class indices —
and no amount of real training will tell you which. Pass `--skip-gate` once it
has passed on this dataset.

### Settings that are not defaults, and why

| Setting | Value | Reason |
|---|---|---|
| `imgsz` | 1024 | measured: box short side is p10 **23px**, median **51px** at 1024. At 640 those become 14px and 32px |
| `nbs` | **8** | the important one. Default 64 with `batch=8` gives `accumulate=8` → ~2 optimizer steps per epoch. Training looks like it ran and learns almost nothing |
| `hsv_h`, `hsv_s` | 0 | achromatic line art; hue/saturation jitter augments nothing |
| `degrees`, `shear`, `perspective` | 0 | axis-aligned drawings. A rotation re-fits the axis-aligned box around the rotated shape, which **inflates it** — it corrupts the labels |
| `fliplr` | 0 | door pictograms are chiral: swing arcs, hinge marks, sliding arrows |
| `mosaic` | 0.5, close at 40 | strongest regulariser available at 90 images, but it halves object size |
| `workers` | 2 | crops run to 9117px wide; the default 8 workers each hold decoded copies |
| `freeze` | none | the domain gap is *in* the early layers. Freezing them freezes the wrong thing |

Raise `workers` and `batch` on a machine with the RAM and VRAM for it — nothing
else in the recipe should change.

## Evaluating

    .venv/bin/python -m schedext.cli evaluate out/dataset/v1/runs/coco-s/weights/best.pt

A 5-planset val set is too small for a single number, so this reports the shape:

- **class-agnostic mAP50** — what the pipeline actually consumes, since stage 1
  supplies the class from the area title
- **per-class AP with val box counts beside them**, so a thin class is visibly
  untrustworthy rather than quietly averaged in
- **per-planset AP** — reveals "works on 4 of 5, fails on 1", the expected shape
- **exact opening count per area** — the number takeoff needs, and far more
  stable than mAP at this size
- **false positives on the hard negatives** — good mAP while firing on title
  blocks is useless
- predictions burned onto the val crops in `eval_preview/` (green = truth,
  blue = window, red = door)

### Measured result — trained 2026-08-20

Both arms at `imgsz=1024`, 300 epochs, `batch=2` (see `GPU-SETUP.md`: 26.4M at
batch=4 does not fit a 12GB card and spills to system RAM). `nbs=8` unchanged,
so accumulate rises to 4 and the effective batch stays 8.

| | incumbent (merged) | **`coco-s`** 9.4M | `coco-l` 26.4M |
|---|---|---|---|
| class-agnostic mAP50 | 0.407 | **0.696** | 0.658 |
| precision | 0.563 | **0.789** | 0.588 |
| recall | 0.571 | **0.725** | 0.699 |
| exact opening count | 18.5% | **33.3%** | 14.8% |
| within +/- 1 | — | **48.1%** | 29.6% |
| door AP50 (123 boxes) | 0.457 | 0.778 | **0.791** |
| window AP50 (73 boxes) | 0.160 | **0.377** | 0.365 |
| false boxes on hard negatives | 5 | **1** | **6** |

**`coco-s` wins; `coco-l` is disqualified.** 6 false boxes on the hard negatives
is worse than the incumbent, and a detector that fires on title blocks is
unusable no matter its mAP. `coco-l` also lands *below* the incumbent on exact
count because it over-predicts — 21 real openings in one door area drew 44 boxes.

The mAP gap (0.696 vs 0.658) is inside the noise band for a 28-image val set, but
the tie-breakers are not: exact count is 2.3x better and precision 34% better.

**Capacity hurts at this dataset size, as predicted.** `coco-l` peaked at epoch
**87** and never improved across 173 further epochs. `coco-s` peaked at 186.

Per planset, `coco-s` ranges 0.516 to 1.000 — still the "works on 4 of 5" shape.
`658_38ee4786` predicts **zero** openings on two areas: a drafting style absent
from 25 plansets. That is a data gap, not a model defect.

### Two findings that contradict the recipe above

**1. The `close_mosaic` clean phase did not pay off, and deadlocks on Windows.**
`coco-s` peaked at epoch 186, during mosaic; its clean phase (260-300) topped out
at 0.5952 and never recovered the 0.6262. `coco-l` peaked at 87. So the
`patience=300` "always lands" schedule bought nothing on either arm and cost
~8h on `coco-l`. Worse, `coco-l` **hung indefinitely** at `Closing dataloader
mosaic` (epoch 260) — a DataLoader worker deadlock. Nothing was lost, because
`best.pt` is written continuously, but budget for it: either accept the default
`patience=60`, or run `close_mosaic` with `--workers 0`.

**2. Data is the binding constraint, not the recipe.** 62 training images is very
small for detection. Stage 1 locates schedule sheets in **197 of 230 plansets —
654 sheets**, of which only 25 plansets / 51 sheets are annotated. Annotating
more, seeded by `coco-s` via `cli detect --weights` (0.696 vs the 0.407 the first
round was seeded with), is the highest-leverage move available.

### Baseline to beat — `best.pt` on the same held-out plansets

Measured, not estimated. `--merge` first collapses the whole-unit-plus-every-sash
boxes a raw multi-class detector emits, because that is the only configuration
`best.pt` is ever run in:

| | raw | merged (conf 0.25) | merged (conf 0.10) |
|---|---|---|---|
| class-agnostic mAP50 | 0.319 | **0.407** | 0.418 |
| precision | 0.364 | **0.563** | 0.534 |
| recall | 0.582 | **0.571** | 0.597 |
| exact opening count | 7.4% | **18.5%** | 22.2% |
| false boxes on hard negatives | 6 | 5 | 5 |

Per class it is lopsided: door AP50 **0.457**, window AP50 **0.160**. Per planset
it ranges 0.06 to 0.87 — it works on some drafting styles and not at all on
others, which is exactly the shape a 25-planset training set is meant to fix.

**This recall is lower than the 90% quoted earlier in the project, and the earlier
number was the wrong measurement.** That 90% was computed against the seed set —
boxes `best.pt` itself produced and a human then kept — so it was scoring the
model against its own output and could only ever look good. The table above is
held-out plansets the model has never influenced. Precision (~55%) is the one
figure that survives the correction unchanged.

So the honest read on the incumbent is worse than previously stated: it is a
usable *seeder* that a human corrects, not a model that is 90% of the way there.

## Gotchas that have already bitten

- **`project=` is resolved against ultralytics' global `runs_dir`** when
  relative. On this machine that setting still points at
  `/var/www/html/WindowsDoorsClassification/runs`, so a relative project path
  silently writes there. `train.py` passes absolute paths.
- **Image lists need `./images/...`**, not `images/...`. Ultralytics expands the
  leading `./` to the txt file's parent and then derives the label path by
  swapping `/images/` for `/labels/`. Without the prefix there is no separator
  and it reports "no labels found".
- **`train:`/`val:` in a dataset yaml are joined onto `path:`.** An absolute
  path there produces `<root>/<root>/...`.
- **`get_pixmap(clip=)` takes display space; `get_text(clip=)` takes extraction
  space.** 36 of 59 plansets have rotated pages. Everything goes through
  `schedext/geom.py`.
