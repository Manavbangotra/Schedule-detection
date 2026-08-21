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

| Arm | Init | Params | What it isolates |
|---|---|---|---|
| `best` | `best.pt` | 26.4M | the incumbent |
| `coco-l` | `yolo12l.pt` | 26.4M | whether the facade training helped or hurt |
| `coco-s` | `yolo11s.pt` | 9.4M | whether the capacity hurts — **the pick** |

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

### Measured result — trained on v2, cross-validated

The single 5-planset split is too small to rank models. 5-fold CV over balanced
planset folds (`schedext/eval/make_folds.py` + `run_cv.py`), every planset held
out exactly once:

    coco-s @ imgsz 1024, 5 folds, all 23 plansets / 678 boxes

      class-agnostic mAP50   0.7368 +/- 0.0853   (0.659 - 0.852)
      window AP50            0.5136 +/- 0.1582   (0.322 - 0.746)
      door AP50              0.7636 +/- 0.0711   (0.643 - 0.826)
      exact opening count    29.7%  +/- 15.3pp   (10%   - 50%)
      within +/- 1           52.3%  +/- 11.3pp   (40%   - 62%)

**The noise floor is +/- 0.085 mAP50.** Measured against it, nearly every arm
comparison below is a tie:

| comparison | gap | verdict |
|---|---|---|
| v1 vs v2 data | 0.002 | noise |
| imgsz 1024 vs 1280 | 0.011 | noise |
| coco-n 2.6M vs coco-s 9.4M | 0.013 | noise |
| coco-s vs best | 0.033 | noise |
| coco-s vs coco-m 20.1M | 0.041 | noise |
| **trained vs incumbent** | **0.287** | **real, 3.4 sigma** |

`coco-s @ 1024` is the pick because nothing beats it and it is the cheapest and
smallest, **not** because it measurably wins. Do not re-rank these on one split.

Two corrections this forced on claims made elsewhere in this file:

- **The hard-negative veto is noisier than it looks.** `coco-s` itself averages
  8.6 false boxes per fold (43 over 5). Disqualifying an arm for firing 6 or 9 on
  a single split was overconfident. Keep the veto as a smell test, not a gate.
- **The single split is pessimistic.** CV mean 0.737 vs single-split 0.694,
  because each fold trains on ~83 images against the split's 70. More training
  data is the one lever that clearly clears the noise.

### The capacity curve — size mattered, pretraining did not

| params | arm | mAP50 | exact count | window AP50 |
|---|---|---|---|---|
| 2.6M | coco-n | 0.681 | 31.0% | 0.435 |
| **9.4M** | **coco-s** | **0.694** | **34.5%** | **0.455** |
| 20.1M | coco-m | 0.653 | 31.0% | 0.374 |
| 26.4M | coco-l (v1) | 0.658 | 14.8% | 0.365 |
| 26.4M | best | 0.661 | 20.7% | 0.413 |

An inverted U peaking at 9.4M, though only the 26.4M arms fall outside the noise
band. `best` vs `coco-l` -- the facade-pretraining question the three-arm design
was built to answer -- is **0.661 vs 0.658: no effect.** Size is what mattered.

### imgsz sweep — where the headline metric actively misleads

| imgsz | mAP50 | window AP50 | exact count | false boxes |
|---|---|---|---|---|
| **1024** | 0.694 | 0.455 | **34.5%** | **4** |
| 1280 | 0.705 | 0.484 | 31.0% | 4 |
| 1536 | **0.738** | **0.540** | 24.1% | **15** |

Resolution buys detection and costs counting, monotonically. **1536 scores
highest on mAP and is the worst model for the job**: it gets the count right on a
quarter of areas and fires 15 times on title blocks. Ranking on mAP alone would
have shipped it. Never lower imgsz below 1024 (p10 box short side is 23px there,
14px at 640) -- and do not raise it either.

### Test-time augmentation — measured, rejected

Paired over the same folds (`schedext/eval/tta_check.py`), which cancels the
fold-to-fold variance that would otherwise swamp an effect this small:

      class-agnostic mAP50   +0.0206 +/- 0.0258   (4/5 folds improved)
      recall                 +0.0455 +/- 0.0217   (5/5 improved)
      precision              -0.1690 +/- 0.0271   (0/5 improved)
      exact opening count    -0.1541 +/- 0.0971   (0/5 improved)
      within +/- 1           -0.2679 +/- 0.1235   (0/5 improved)
      false boxes            +43 across all folds

TTA raises mAP and ruins the count -- it finds more openings and far more junk,
doubling false boxes. **Rejected.** `--augment` exists in `evaluate.py` so the
result stays reproducible, not because it should be used.

### The shipped model

`out/dataset/v2/cv/full/runs/coco-s/weights/best.pt` -- same recipe, trained on
**all 101 images** so the 5 val plansets (29% of boxes) finally contribute.
It has **no held-out score by construction**; its honest expected accuracy is the
CV mean above, 0.737 +/- 0.085. `arm-coco-s.pt` is kept beside it as the
measured fallback.

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
