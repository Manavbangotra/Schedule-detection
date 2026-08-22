"""Render schedule sheets with the extracted boxes drawn on them (local viewer).

Rects stored by the pipeline are in *display* space -- already through the
rotation matrix -- and ``page.get_pixmap`` renders the displayed page, so image
pixels are simply display points times the zoom. That correspondence is what
lets the overlay sit exactly on the drawing with no per-page fudging.
"""

from __future__ import annotations

import json
from pathlib import Path

import pymupdf

from . import manifest as manifest_mod

# Big enough to read a schedule's small type, small enough to keep the whole
# corpus browsable from disk.
ZOOM = 1.8
JPEG_QUALITY = 78


def _scaled(rect, zoom: float) -> list[float]:
    return [round(v * zoom, 1) for v in rect]


def _load_detections(path: Path) -> dict[tuple[str, int], list[dict]]:
    """Openings keyed by (file, page), if a detection run has happened."""
    if not path.exists():
        return {}
    out: dict[tuple[str, int], list[dict]] = {}
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            out[(record["file"], record["page"])] = record["openings"]
    return out


def build(manifest_path: Path, results_dir: Path, out_dir: Path,
          zoom: float = ZOOM, detections_path: Path | None = None) -> dict:
    """Render every sheet that produced blocks, plus a JSON index of boxes."""
    # Key by filename, not project_id. Four projects (540, 5416, 5449, 723) were
    # uploaded twice as different files; keying by project_id silently kept the
    # last one, so project_540's boxes -- taken from the 15-page upload -- were
    # drawn over a render of the 26-page upload. Anyone annotating that sheet
    # would have been labelling the wrong page.
    records = {Path(r["path"]).name: r for r in manifest_mod.load(manifest_path)}
    images = out_dir / "pages"
    images.mkdir(parents=True, exist_ok=True)

    detections = _load_detections(
        detections_path or Path("out") / "openings.jsonl")

    sheets: list[dict] = []
    for path in sorted(results_dir.glob("*.json")):
        result = json.loads(path.read_text())
        source = result.get("source", {})
        project = source.get("project_id")
        record = records.get(source.get("file", ""))
        if not record or not result.get("sheets"):
            continue
        # Short hash disambiguates the duplicate uploads in every derived name.
        sha8 = record["sha256"][:8]

        items_by_page: dict[int, list[dict]] = {}
        for item in result.get("items", []):
            items_by_page.setdefault(item["page"], []).append(item)

        with pymupdf.open(record["path"]) as doc:
            for sheet in result["sheets"]:
                page_no = sheet["page"]
                if not sheet.get("blocks"):
                    continue
                page = doc[page_no - 1]
                name = f"{project}_{sha8}_p{page_no}.jpg"
                target = images / name
                # Rendering the corpus takes minutes; the boxes change far more
                # often than the pages do, so keep what is already on disk.
                if target.exists():
                    pixmap = pymupdf.Pixmap(target)
                else:
                    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
                    pixmap.save(target, jpg_quality=JPEG_QUALITY)

                sheets.append({
                    "project_id": project,
                    "sheet_key": f"{project}_{sha8}_p{page_no}",
                    "planset_key": f"{project}_{sha8}",
                    "pdf_sha256": record["sha256"],
                    "file": source.get("file", ""),
                    "page": page_no,
                    "sheet_name": sheet.get("sheet_name", ""),
                    "rotation": sheet.get("rotation", 0),
                    "image": f"pages/{name}",
                    "width": pixmap.width,
                    "height": pixmap.height,
                    "blocks": [
                        {
                            "block_id": b["block_id"],
                            "title": b["title"],
                            "category": b["category"],
                            "kind": b["kind"],
                            "genre": b["genre"],
                            "rect": _scaled(b["rect"], zoom),
                        }
                        for b in sheet["blocks"]
                    ],
                    "items": [
                        {
                            "mark": i["mark"],
                            "category": i["category"],
                            "block_id": i["block_id"],
                            "operation": (i.get("canonical") or {}).get("operation"),
                            "material": (i.get("canonical") or {}).get("material"),
                            "leaf_face": (i.get("canonical") or {}).get("leaf_face"),
                            "needs_review": bool((i.get("canonical") or {}).get("needs_review")),
                            "width_text": i.get("width_text", ""),
                            "height_text": i.get("height_text", ""),
                            "type_text": i.get("type_text", ""),
                            "extractor": i.get("extractor", ""),
                            "rect": _scaled(i["rect"], zoom) if i.get("rect") else None,
                        }
                        for i in items_by_page.get(page_no, [])
                    ],
                    # Detector output is already in pixels at detect.ZOOM; it
                    # matches this render because both use the same zoom.
                    "detections": [
                        {
                            "rect": [d["x0_px"], d["y0_px"], d["x1_px"], d["y1_px"]],
                            # Carried through so a later pass can attach the
                            # schedule row this opening belongs to. It is the
                            # only stable handle back to openings.csv; the rect
                            # here is in render pixels, not the points that CSV
                            # holds, so geometry cannot be used to re-pair them.
                            "opening_id": d.get("opening_id", ""),
                            "confidence": d["confidence"],
                            "source": d["source"],
                            "in_block": d["in_block"],
                            "block_title": d["block_title"],
                            "category": d["category"],
                        }
                        for d in detections.get((source.get("file", ""), page_no), [])
                    ],
                })

    index = {"zoom": zoom, "sheets": sheets}
    (out_dir / "index.json").write_text(json.dumps(index, separators=(",", ":")))
    return {
        "sheets": len(sheets),
        "blocks": sum(len(s["blocks"]) for s in sheets),
        "items": sum(len(s["items"]) for s in sheets),
        "boxed_items": sum(1 for s in sheets for i in s["items"] if i["rect"]),
        "detections": sum(len(s["detections"]) for s in sheets),
    }
