# schedext — window & door schedule extraction

Pulls every window and door out of the schedule sheets in a plan set: mark,
size, and a canonical type, with the page and bounding box it came from.

## Why it is built this way

The starting plan was to train a YOLO detector. Testing the existing weights
(`/var/www/html/WindowsDoorsClassification/best.pt`, YOLOv12-L, 17 classes)
against real schedule sheets showed a clean split:

* **Localisation is good.** After the merge step the service already applies,
  box counts matched the true item counts exactly on all four regions tested
  (8/8 doors, 4/4 windows, 11/11 door types, 6/6 windows).
* **Classification is not.** It called windows printed `SINGLE HUNG` "fixed"
  and "casement", a glass shower door "flush", an aluminium storefront
  "square shaker", and labelled a block titled **`DOOR TYPES`** as windows.

Meanwhile 53 of the 59 plan sets are vector CAD exports whose text layer
already states the type. So the type comes from text, and the detector is kept
for the job it is actually good at. `labels.py` builds a 3-class detection
dataset from the text pipeline so that finetune needs no hand annotation.

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install pymupdf opencv-python-headless pydantic rapidfuzz pyyaml \
                      python-dotenv pandas
.venv/bin/pip install ultralytics    # only needed for labels.py
```

No GPU is required for extraction.

## Run

```bash
.venv/bin/python -m schedext.cli all           # manifest -> locate -> extract
.venv/bin/python -m schedext.cli one all_plansets/project_559_*.pdf 1
```

Outputs land in `out/`:

| File | Contents |
|---|---|
| `manifest.jsonl` | one record per plan set: pages, rotations, producer, health |
| `candidates.jsonl` | schedule sheets found, with the signals that found them |
| `plansets/*.json` | full result per plan set — sheets, blocks, items, warnings |
| `schedule_items.csv` | every item, flat, for takeoff |
| `type_summary.json` | distinct types with counts |

## Stages

```
pdfio/manifest  open + health triage
locate          which sheets carry a schedule   (TOC + title font + header row)
segment         cut a sheet into titled blocks
genre           tabular vs pictorial, per block
table           parse a table          + recover unruled columns
pictorial       parse drawn elevations + bullet specs
legend          decode TYPE codes (F1 -> FLUSH) from the sheet's own legend
normalize       free text -> canonical faceted type
pipeline        orchestrate; export    JSON + CSV
labels          build a YOLO dataset from all of the above
```

## Things that will bite you

**Page rotation.** 36 of 59 plan sets contain rotated pages. `page.rect` is the
rotated view but `get_text()` returns *unrotated* coordinates, and on a
270° page "below the title" is no longer +y. Everything goes through `geom.py`
for this reason; nothing else should touch raw coordinates.

**`find_tables` drops unruled columns.** On project_486's window schedule
`strategy="lines_strict"` returns a clean table and silently omits `COMMENTS` —
the column holding `VINYL SINGLE HUNG`, `DECORATIVE SHUTTER`, `LOUVER`,
`GRILL`. `strategy="text"` sees it but shatters the dimensions. `table.py`
takes the ruled skeleton and then re-reads the tail per row band.

**Clip `find_tables` to a block.** Full page: 7 tables, 17.2s. Clipped to the
block: 1 table, 1.1s.

**Two-tier headers.** `DOOR SIZE` spans `WIDTH`/`HEIGHT`/`THICK`. Group tiers
forward-fill; the leaf tier must not, or `FRAME.MATERIAL` bleeds into
`FIRE RTG..MATERIAL` and `COMMENTS.MATERIAL`.

**Titles are centred over their tables, not left-aligned.** Splitting adjacent
blocks on title *left edges* puts the boundary in the wrong place, and a table
then reads its neighbour's rows. `segment._resolve_overlaps` splits on title
centres.

**Sub-bullets are not columns.** Nested specs use a doubled glyph (`·· XXX
OUTSIDE`). Treating them as column anchors turns four windows into eight.

**Nothing is guessed.** Text no rule matches is emitted with
`needs_review: true` rather than assigned a plausible type. The review rate is
the honest coverage number for the lexicon.

## Tests

```bash
.venv/bin/python -m schedext.eval.test_lexicon   # title scoring
.venv/bin/python -m schedext.eval.test_blocks    # locate -> segment -> genre
.venv/bin/python -m schedext.eval.test_extract   # row-level ground truth
```

`test_extract` pins the two cases the detector got wrong — project_559's
windows B and C (`SINGLE HUNG`) and project_486's `LOUVER` / `DECORATIVE
SHUTTER` / `GRILL` rows.

## Known gaps

* `project_725_20260709_203652_3b2458d7.pdf` is truncated at exactly 8 MiB
  (opens, reports 0 pages). **Re-download from S3** — the zip copy is truncated
  identically.
* Four plan sets have no text layer; two of those (`project_646`,
  `project_731`) are raster scans, and `project_646` is ~36 DPI and unlikely to
  be recoverable. The other two are vector-outlined text that renders sharply
  and suits a VLM pass. That path is designed but not built.
* Pictorial schedules that use neither bullets nor `TYPE x` captions are not
  yet parsed; they show up as `pictorial parse found nothing` warnings.
