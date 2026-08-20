"""Human-verified annotation of openings (stage A).

Two levels, and one rule that makes the whole thing tractable:

  Level 1  areas    -- regions labelled "window area" or "door area"
  Level 2  openings -- the individual openings drawn inside them
  The rule -- every opening inside a window area is a window; inside a door
              area, a door.

That rule is why this works. Matching each detected box to its own schedule row
is genuinely hard (a naive geometric match put five boxes on the same mark);
inheriting the class from the enclosing area sidesteps it entirely, and the area
label comes from the block *title*, which is the most reliable signal on a sheet.

Coordinates are stored in **display points**, never pixels. Pixels are an
artifact of the render zoom; re-render at a different zoom and every stored
pixel silently shifts. Points survive that, and `get_pixmap(clip=)` takes
display points anyway, so exporting a crop needs no rotation maths.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1

# Share of an opening that must lie inside an area to belong to it. Same value
# as detect.IN_BLOCK so the editor and the detector never disagree.
CONTAINMENT = 0.70

# Seeded state thresholds. A filter may only ever defer a decision to the human
# by marking a box "pending" -- it must never reject or drop one.
AUTO_ACCEPT_CONF = 0.50
MIN_SIDE_PT = 13.0          # ~24 px at the viewer's 1.8 zoom
LOW_CONF = 0.35
EXTREME_ASPECT = (0.15, 6.0)

AREA_TYPES = ("window_area", "door_area", "garage_area", "mixed", "unknown", "exclude")
CLASSES = ("window", "door", "garage_door")

# Block category -> seeded area type. Deliberately not labels.CATEGORY_ALIASES,
# which drops `both` and `unknown`; here a human resolves them, so they have to
# survive seeding.
SEED_AREA_TYPE = {
    "window": "window_area",
    "door": "door_area",
    "storefront": "door_area",
    "garage_door": "garage_area",
    "both": "mixed",
    "unknown": "unknown",
}
AREA_CLASS = {"window_area": "window", "door_area": "door", "garage_area": "garage_door"}


@dataclass
class Area:
    id: str
    type: str
    rect: list[float]                      # display points
    origin: str = "model"                  # model | human
    edited: bool = False
    type_source: str = "seed"              # seed | human
    seed: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Opening:
    id: str
    rect: list[float]                      # display points
    cls: str = ""
    cls_source: str = "inherited"          # inherited | human | required
    area_id: str = ""
    containment: float = 0.0
    origin: str = "model"
    edited: bool = False
    state: str = "pending"                 # pending | accepted | rejected
    flags: list[str] = field(default_factory=list)
    confidence: float = 0.0
    detector_source: str = ""
    group_id: str = ""
    seed: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SheetAnnotation:
    sheet_key: str
    project_id: str
    planset_key: str
    file: str
    pdf_sha256: str
    page: int
    rotation: int
    page_rect_pt: list[float]
    render: dict
    areas: list[Area] = field(default_factory=list)
    openings: list[Opening] = field(default_factory=list)
    rev: int = 1
    status: str = "untouched"              # untouched|in_progress|verified|needs_recheck|skipped
    skip_reason: str = ""
    time_spent_s: int = 0
    notes: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict:
        data = asdict(self)
        data["schema_version"] = SCHEMA_VERSION
        return data

    @staticmethod
    def from_dict(data: dict) -> "SheetAnnotation":
        areas = [Area(**a) for a in data.get("areas", [])]
        openings = [Opening(**o) for o in data.get("openings", [])]
        fields = {k: v for k, v in data.items()
                  if k in SheetAnnotation.__dataclass_fields__}
        fields["areas"] = areas
        fields["openings"] = openings
        return SheetAnnotation(**fields)


# --- geometry ---------------------------------------------------------------

def _area_of(r) -> float:
    return max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])


def _intersection(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def containment(box, area) -> float:
    """Share of ``box`` lying inside ``area``."""
    return _intersection(box, area) / max(_area_of(box), 1e-9)


def assign(opening: Opening, areas: list[Area]) -> None:
    """Set the opening's area and inherited class, in place.

    Smallest containing area wins, matching detect._tag -- it matters where a
    type legend sits inside a schedule block.
    """
    best = None
    for area in areas:
        if area.type == "exclude":
            continue
        share = containment(opening.rect, area.rect)
        if share >= CONTAINMENT and (best is None or _area_of(area.rect) < _area_of(best[0].rect)):
            best = (area, share)

    if best is None:
        opening.area_id = ""
        opening.containment = 0.0
        if opening.cls_source != "human":
            opening.cls = ""
        return

    area, share = best
    opening.area_id = area.id
    opening.containment = round(share, 3)
    if opening.cls_source == "human":
        return
    if area.type in AREA_CLASS:
        opening.cls = AREA_CLASS[area.type]
        opening.cls_source = "inherited"
    else:
        # mixed / unknown areas cannot hand down a class; the human must say.
        opening.cls = ""
        opening.cls_source = "required"


# --- persistence ------------------------------------------------------------
#
# Annotations live at the repo root, NOT under out/. Everything in out/ is
# regenerated by the pipeline; a stray `rm -rf out` would take hand-made work
# with it. This is the only data here a human produced.

ANN_DIR = Path("annotations")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def path_for(sheet_key: str, ann_dir: Path = ANN_DIR) -> Path:
    return ann_dir / f"{sheet_key}.json"


def save(sheet: SheetAnnotation, ann_dir: Path = ANN_DIR) -> Path:
    """Write one sheet atomically, snapshotting the previous revision."""
    ann_dir.mkdir(parents=True, exist_ok=True)
    target = path_for(sheet.sheet_key, ann_dir)

    if target.exists():
        history = ann_dir / ".history" / sheet.sheet_key
        history.mkdir(parents=True, exist_ok=True)
        previous = json.loads(target.read_text())
        (history / f"{previous.get('rev', 0)}.json").write_text(
            json.dumps(previous, indent=1, sort_keys=True))

    sheet.updated_at = _now()
    # Same directory: os.replace is only atomic within one filesystem.
    tmp = target.with_suffix(f".{os.getpid()}.tmp")
    with tmp.open("w") as handle:
        # Stable key order so a git diff shows the box that changed, not a
        # reshuffle of the whole file.
        json.dump(sheet.to_dict(), handle, indent=1, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, target)

    with (ann_dir / "_journal.jsonl").open("a") as handle:
        handle.write(json.dumps({
            "ts": sheet.updated_at, "key": sheet.sheet_key, "rev": sheet.rev,
            "areas": len(sheet.areas), "openings": len(sheet.openings),
            "accepted": sum(1 for o in sheet.openings if o.state == "accepted"),
            "rejected": sum(1 for o in sheet.openings if o.state == "rejected"),
            "status": sheet.status, "dt_s": sheet.time_spent_s,
        }) + "\n")
    return target


def load(sheet_key: str, ann_dir: Path = ANN_DIR) -> SheetAnnotation | None:
    target = path_for(sheet_key, ann_dir)
    if not target.exists():
        return None
    return SheetAnnotation.from_dict(json.loads(target.read_text()))


def load_all(ann_dir: Path = ANN_DIR) -> dict[str, SheetAnnotation]:
    if not ann_dir.exists():
        return {}
    out = {}
    for path in sorted(ann_dir.glob("*.json")):
        try:
            sheet = SheetAnnotation.from_dict(json.loads(path.read_text()))
        except Exception:
            continue
        out[sheet.sheet_key] = sheet
    return out


# --- seeding ----------------------------------------------------------------

def _flags_for(rect: list[float], confidence: float) -> list[str]:
    """Reasons this box deserves a human's attention. Never a reason to drop it."""
    flags = []
    width, height = rect[2] - rect[0], rect[3] - rect[1]
    if min(width, height) < MIN_SIDE_PT:
        flags.append("tiny")
    if confidence and confidence < LOW_CONF:
        flags.append("low_conf")
    aspect = width / max(height, 1e-6)
    if aspect < EXTREME_ASPECT[0] or aspect > EXTREME_ASPECT[1]:
        flags.append("extreme_aspect")
    return flags


def seed_sheet(sheet_meta: dict, blocks: list[dict], detections: list[dict],
               zoom: float) -> SheetAnnotation:
    """Build a fresh annotation from pipeline output. Never overwrites."""
    sheet = SheetAnnotation(
        sheet_key=sheet_meta["sheet_key"],
        project_id=sheet_meta["project_id"],
        planset_key=sheet_meta["planset_key"],
        file=sheet_meta["file"],
        pdf_sha256=sheet_meta["pdf_sha256"],
        page=sheet_meta["page"],
        rotation=sheet_meta.get("rotation", 0),
        page_rect_pt=sheet_meta.get("page_rect_pt", []),
        render={"image": sheet_meta["image"], "zoom": zoom,
                "w": sheet_meta["width"], "h": sheet_meta["height"]},
    )

    for block in blocks:
        kind = block.get("kind", "")
        genre = block.get("genre", "")
        reason = ""
        if kind in {"notes", "hardware"}:
            seeded, reason = "exclude", f"{kind} block"
        elif kind == "schedule" and genre == "tabular":
            # A tabular schedule is a table of text -- it has no drawn openings
            # in it at all. Measured over the corpus: 1,254 detections land in
            # these blocks with median confidence 0.28 and a median side of
            # 10 px, and sampled crops are letter fragments and cells like
            # "NOMINAL COOLING (TONS)". The drawings live in the pictorial
            # blocks and the type legends. Excluding these removes a third of
            # all detections without losing a single real opening.
            seeded, reason = "exclude", "tabular schedule: text table, no drawn openings"
        else:
            seeded = SEED_AREA_TYPE.get(block.get("category", "unknown"), "unknown")
        sheet.areas.append(Area(
            id=f"a_{block['block_id']}",
            type=seeded,
            rect=[round(v / zoom, 2) for v in block["rect"]],
            seed={"type": seeded, "block_id": block["block_id"],
                  "title": block.get("title", ""),
                  "category": block.get("category", ""),
                  "kind": kind, "genre": genre,
                  "exclude_reason": reason},
        ))

    for index, det in enumerate(detections):
        rect = [round(det[k] / zoom, 2) for k in ("x0_px", "y0_px", "x1_px", "y1_px")] \
            if "x0_px" in det else [round(v / zoom, 2) for v in det["rect"]]
        confidence = float(det.get("confidence", 0.0))
        opening = Opening(
            id=f"o_{sheet.page}_{index:03d}",
            rect=rect,
            confidence=confidence,
            detector_source=det.get("source", ""),
            group_id=det.get("group_id", ""),
            flags=_flags_for(rect, confidence),
            seed={"rect": list(rect), "confidence": confidence},
        )
        assign(opening, sheet.areas)

        # assign() ignores excluded areas by design, so a box sitting inside one
        # comes back unassigned. Check separately -- these are the text-table
        # detections and they should start rejected, not merely unlabelled.
        in_excluded = any(
            a.type == "exclude" and containment(opening.rect, a.rect) >= CONTAINMENT
            for a in sheet.areas
        )
        if not opening.area_id and not in_excluded:
            opening.flags.append("outside_area")

        if in_excluded:
            opening.state = "rejected"
            opening.flags.append("in_excluded_area")
        elif (not opening.flags and opening.cls
                and confidence >= AUTO_ACCEPT_CONF):
            # Clean, confident, inside a real area: pre-accepted so the human
            # scans for misses instead of confirming the obvious.
            opening.state = "accepted"
        sheet.openings.append(opening)

    return sheet


# --- merging a re-run -------------------------------------------------------

MATCH_IOU = 0.5


def _iou(a, b) -> float:
    i = _intersection(a, b)
    union = _area_of(a) + _area_of(b) - i
    return i / union if union > 0 else 0.0


def merge(existing: SheetAnnotation, fresh: SheetAnnotation) -> tuple[SheetAnnotation, dict]:
    """Fold a fresh detector run into existing annotations.

    The invariant, and the whole point of this function:

        never remove an object with origin == "human" or edited == True,
        and never rewrite the rect of an object with edited == True.

    A human's hour is worth more than a model's second. Anything the detector
    newly proposes arrives as `pending` for review, never silently accepted.
    """
    delta = {"kept_human": 0, "refreshed": 0, "added_pending": 0,
             "unmatched_existing": 0, "downgraded": False}

    # --- areas: match on the block they were seeded from, then on geometry ---
    fresh_by_block = {a.seed.get("block_id"): a for a in fresh.areas if a.seed.get("block_id")}
    used: set[str] = set()
    for area in existing.areas:
        match = fresh_by_block.get(area.seed.get("block_id"))
        if match is None:
            candidates = [f for f in fresh.areas
                          if f.id not in used and _iou(area.rect, f.rect) >= MATCH_IOU]
            match = candidates[0] if candidates else None
        if match is None:
            delta["unmatched_existing"] += 1
            continue
        used.add(match.id)
        if not area.edited:
            area.rect = list(match.rect)
            delta["refreshed"] += 1
        else:
            delta["kept_human"] += 1
        area.seed = dict(match.seed)
        if area.type_source != "human":
            area.type = match.type

    for area in fresh.areas:
        if area.id not in used and area.seed.get("block_id") not in {
            a.seed.get("block_id") for a in existing.areas
        }:
            area.seed["new_since_seed"] = True
            existing.areas.append(area)

    # --- openings: greedy IoU, highest overlap first -------------------------
    pairs = sorted(
        ((_iou(o.rect, f.rect), oi, fi)
         for oi, o in enumerate(existing.openings)
         for fi, f in enumerate(fresh.openings)),
        reverse=True,
    )
    taken_existing: set[int] = set()
    taken_fresh: set[int] = set()
    for score, oi, fi in pairs:
        if score < MATCH_IOU or oi in taken_existing or fi in taken_fresh:
            continue
        taken_existing.add(oi)
        taken_fresh.add(fi)
        old, new = existing.openings[oi], fresh.openings[fi]
        old.confidence = new.confidence
        old.detector_source = new.detector_source
        old.seed = dict(new.seed)
        if old.edited or old.origin == "human":
            delta["kept_human"] += 1
        else:
            old.rect = list(new.rect)
            delta["refreshed"] += 1

    # "New since verify" only means something if the sheet was ever reviewed.
    # On a sheet nobody has looked at yet, forcing every box to pending throws
    # away the auto-accept triage and hands the human 3,400 boxes to confirm
    # instead of the few hundred that actually need a decision.
    reviewed = existing.status in {"verified", "needs_recheck", "in_progress"}
    for fi, new in enumerate(fresh.openings):
        if fi in taken_fresh:
            continue
        if reviewed:
            new.state = "pending"
            new.flags = list(new.flags) + ["new_since_verify"]
            delta["added_pending"] += 1
        else:
            delta["refreshed"] += 1
        existing.openings.append(new)

    for opening in existing.openings:
        assign(opening, existing.areas)

    if existing.status == "verified" and delta["added_pending"]:
        # Never silently still-verified once the model has proposed something new.
        existing.status = "needs_recheck"
        delta["downgraded"] = True

    existing.rev += 1
    return existing, delta


def progress(sheets: dict[str, SheetAnnotation]) -> dict:
    """Rollup for the sidebar and the work queue."""
    rows = []
    for key, sheet in sheets.items():
        flagged = sum(1 for o in sheet.openings
                      if o.state == "pending" or o.flags)
        unknown = sum(1 for a in sheet.areas if a.type in {"unknown", "mixed"})
        rows.append({
            "sheet_key": key,
            "project_id": sheet.project_id,
            "page": sheet.page,
            "status": sheet.status,
            "areas": len(sheet.areas),
            "openings": sum(1 for o in sheet.openings if o.state != "rejected"),
            "flagged": flagged,
            "needs_review_areas": unknown,
            "difficulty": round(
                flagged + 3 * unknown + 5 * (len(sheet.areas) == 0)
                + 0.2 * len(sheet.openings), 1),
            "time_spent_s": sheet.time_spent_s,
        })
    # Skipped sheets stay on disk with their work intact, but they leave the
    # queue and the export -- that is the point of skipping a duplicate.
    rows = [r for r in rows if r["status"] != "skipped"]
    rows.sort(key=lambda r: (r["project_id"], r["page"]))
    return {
        "sheets": rows,
        "totals": {
            "sheets": len(rows),
            "verified": sum(1 for r in rows if r["status"] == "verified"),
            "openings": sum(r["openings"] for r in rows),
            "flagged": sum(r["flagged"] for r in rows),
        },
    }


# --- driver -----------------------------------------------------------------

def is_deleted(key: str, ann_dir: Path = ANN_DIR) -> bool:
    """True when a human deleted this sheet through the annotator.

    Deletion writes ``.history/<key>/deleted.json`` as a tombstone before
    unlinking. Without checking it, seeding and merging silently resurrect every
    sheet the human pruned -- and since duplicates are pruned in bulk, the next
    merge hands back the whole pile. Measured on this corpus: 32 of 34 seedable
    sheets were resurrections.

    Undo is manual and deliberate: delete the tombstone.
    """
    return (Path(ann_dir) / ".history" / key / "deleted.json").exists()


def seed_corpus(viewer_index: Path, ann_dir: Path = ANN_DIR,
                force: bool = False) -> dict:
    """Seed annotations for every sheet in the viewer index.

    Seeding is never destructive: a sheet that already has an annotation file is
    skipped unless ``force`` is set. Fold a fresh detector run in with
    :func:`merge` instead.
    """
    index = json.loads(Path(viewer_index).read_text())
    zoom = index.get("zoom", 1.8)
    ann_dir.mkdir(parents=True, exist_ok=True)

    stats = {"seeded": 0, "skipped": 0, "areas": 0, "openings": 0,
             "accepted": 0, "pending": 0, "needs_review_areas": 0}

    for sheet in index["sheets"]:
        key = sheet.get("sheet_key")
        if not key:
            continue
        if path_for(key, ann_dir).exists() and not force:
            stats["skipped"] += 1
            continue
        if is_deleted(key, ann_dir) and not force:
            stats["skipped"] += 1
            stats["deleted"] = stats.get("deleted", 0) + 1
            continue

        meta = {
            "sheet_key": key,
            "project_id": sheet["project_id"],
            "planset_key": sheet.get("planset_key", sheet["project_id"]),
            "file": sheet["file"],
            "pdf_sha256": sheet.get("pdf_sha256", ""),
            "page": sheet["page"],
            "rotation": sheet.get("rotation", 0),
            "image": sheet["image"],
            "width": sheet["width"],
            "height": sheet["height"],
            "page_rect_pt": [0, 0, round(sheet["width"] / zoom, 1),
                             round(sheet["height"] / zoom, 1)],
        }
        annotation = seed_sheet(meta, sheet.get("blocks", []),
                                sheet.get("detections", []), zoom)
        save(annotation, ann_dir)

        stats["seeded"] += 1
        stats["areas"] += len(annotation.areas)
        stats["openings"] += len(annotation.openings)
        stats["accepted"] += sum(1 for o in annotation.openings if o.state == "accepted")
        stats["pending"] += sum(1 for o in annotation.openings if o.state == "pending")
        stats["needs_review_areas"] += sum(
            1 for a in annotation.areas if a.type in {"unknown", "mixed"})
    return stats


def merge_corpus(viewer_index: Path, ann_dir: Path = ANN_DIR,
                 apply: bool = False) -> dict:
    """Fold a fresh detector run into existing annotations.

    Dry run by default. Human work is never touched (see :func:`merge`), so the
    only thing at risk is model geometry the human has not yet looked at -- but
    printing the delta first is cheap and makes the merge auditable.
    """
    index = json.loads(Path(viewer_index).read_text())
    zoom = index.get("zoom", 1.8)

    totals = {"sheets": 0, "kept_human": 0, "refreshed": 0, "added_pending": 0,
              "downgraded": 0, "seeded_new": 0}
    rows = []

    for sheet in index["sheets"]:
        key = sheet.get("sheet_key")
        if not key:
            continue
        meta = {
            "sheet_key": key, "project_id": sheet["project_id"],
            "planset_key": sheet.get("planset_key", sheet["project_id"]),
            "file": sheet["file"], "pdf_sha256": sheet.get("pdf_sha256", ""),
            "page": sheet["page"], "rotation": sheet.get("rotation", 0),
            "image": sheet["image"], "width": sheet["width"], "height": sheet["height"],
            "page_rect_pt": [0, 0, round(sheet["width"] / zoom, 1),
                             round(sheet["height"] / zoom, 1)],
        }
        fresh = seed_sheet(meta, sheet.get("blocks", []),
                           sheet.get("detections", []), zoom)

        existing = load(key, ann_dir)
        if existing is None:
            if is_deleted(key, ann_dir):
                totals["skipped_deleted"] = totals.get("skipped_deleted", 0) + 1
                continue
            totals["seeded_new"] += 1
            if apply:
                save(fresh, ann_dir)
            continue

        merged, delta = merge(existing, fresh)
        totals["sheets"] += 1
        for field_name in ("kept_human", "refreshed", "added_pending"):
            totals[field_name] += delta[field_name]
        totals["downgraded"] += int(delta["downgraded"])
        if delta["added_pending"] or delta["kept_human"]:
            rows.append((key, delta))
        if apply:
            save(merged, ann_dir)

    totals["changed_sheets"] = rows[:20]
    return totals
