"""Run every stage over a planset and emit the result (stage S6)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from . import (genre, legend, locate, manifest as manifest_mod, normalize,
               pdfio, pictorial, segment, table)

# Column headers that carry each canonical field, matched case-insensitively
# against the (possibly group-qualified) column name.
COLUMN_ROLES = {
    "mark": r"\bMARK\b|\bMK\b|\bTAG\b|^NO\.?$|\bNUMBER\b|DOOR\s*NO",
    "width": r"\bWIDTH\b|^W\.?$",
    "height": r"\bHEIGHT\b|^H\.?$|^HT\.?$",
    "thickness": r"\bTHICK",
    "head_height": r"HEAD\s*(?:HEIGHT|HT)",
    "quantity": r"\bQTY\b|\bQUANTITY\b|LEAF\s*QTY",
    "type_code": r"\bTYPE\b",
    "material": r"\bMATERIAL\b|\bMAT\.?$",
    "frame": r"\bFRAME\b",
    "glazing": r"\bGLAZING\b|\bGLASS\b",
    "fire_rating": r"\bFIRE\b|\bRTG\b|\bRATING\b",
    "hardware": r"\bHARDWARE\b|\bHDW\b",
    "location": r"\bLOCATION\b|\bROOM\b",
    "remarks": r"\bREMARKS\b|\bCOMMENTS\b|\bNOTES\b",
}
_ROLES = {k: re.compile(v, re.I) for k, v in COLUMN_ROLES.items()}


@dataclass
class Item:
    item_id: str
    category: str
    mark: str
    page: int
    block_id: str
    raw: dict = field(default_factory=dict)
    verbatim: str = ""
    width_text: str = ""
    height_text: str = ""
    width_in: float | None = None
    height_in: float | None = None
    head_height_in: float | None = None
    quantity: str = ""
    type_code: str = ""
    type_text: str = ""
    canonical: dict = field(default_factory=dict)
    specs: list[str] = field(default_factory=list)
    rect: list[float] = field(default_factory=list)
    extractor: str = "text"

    def to_dict(self) -> dict:
        data = self.__dict__.copy()
        return data


def _roles_for(columns: list[str]) -> dict[str, str]:
    """Map canonical field -> column name, first match wins."""
    out: dict[str, str] = {}
    for role, pattern in _ROLES.items():
        for column in columns:
            if pattern.search(column) and role not in out:
                out[role] = column
    return out


# A mark shaped like A, B, 01, W3, 101A.
_LEADING_MARK = re.compile(r"^([A-Z]{0,2}\d{0,3}[A-Z]?)\s+(?=\S)")


def _split_leading_mark(columns: list[str], cells: dict[str, str]) -> tuple[str, dict[str, str]]:
    """Peel a mark off the first column when the sheet gave it no header.

    Common in unruled schedules: the mark column is drawn but never labelled,
    so it merges into the neighbour ("A SLIDER WINDOW"). Without this the row
    has no mark and gets dropped, losing the whole table.
    """
    for name in columns:
        value = cells.get(name, "").strip()
        if not value:
            continue
        match = _LEADING_MARK.match(value)
        if not match or not match.group(1):
            return "", cells
        remainder = value[match.end():].strip()
        updated = dict(cells)
        updated[name] = remainder
        return match.group(1), updated
    return "", cells


def _from_table(parsed: table.ParsedTable, block: segment.Block,
                page_no: int, project: str,
                legends: dict[str, dict[str, str]] | None = None) -> list[Item]:
    roles = _roles_for(parsed.columns)
    legends = legends or {}
    items: list[Item] = []
    for row in parsed.rows:
        cells = row.cells
        mark = cells.get(roles.get("mark", ""), "").strip()
        if not mark:
            mark, cells = _split_leading_mark(parsed.columns, cells)
        if not mark:
            continue

        # Everything that is not a size or an id contributes to the type text.
        structural = {roles.get(k, "") for k in
                      ("mark", "width", "height", "thickness", "head_height", "quantity")}
        type_text = " ".join(
            value for name, value in cells.items()
            if name not in structural and value
        ).strip()

        # A TYPE code is meaningless on its own; the sheet's own legend says
        # what F1 or A0 draws. Folding the caption in is what turns a bare
        # code into a leaf face.
        type_code = cells.get(roles.get("type_code", ""), "").strip()
        caption = legend.caption_for(legends, block.category, type_code)
        canonical = normalize.classify(
            " ".join([type_text, caption, block.title]), block.category, type_code
        )
        if caption:
            canonical.mapped_by = "legend"
        items.append(
            Item(
                item_id=f"{project}.{block.block_id}.{mark}",
                category=canonical.category or block.category,
                mark=mark,
                page=page_no,
                block_id=block.block_id,
                raw=cells,
                verbatim=row.verbatim,
                width_text=cells.get(roles.get("width", ""), ""),
                height_text=cells.get(roles.get("height", ""), ""),
                width_in=normalize.to_inches(cells.get(roles.get("width", ""), "")),
                height_in=normalize.to_inches(cells.get(roles.get("height", ""), "")),
                head_height_in=normalize.to_inches(
                    cells.get(roles.get("head_height", ""), "")
                ),
                quantity=cells.get(roles.get("quantity", ""), ""),
                type_code=type_code,
                type_text=type_text,
                canonical=canonical.to_dict(),
                rect=[round(v, 1) for v in row.rect],
                extractor="text.table",
            )
        )
    return items


def _from_pictorial(entries: list[pictorial.PictorialItem], block: segment.Block,
                    page_no: int, project: str) -> list[Item]:
    items: list[Item] = []
    for index, entry in enumerate(entries):
        mark = entry.mark or f"?{index + 1}"
        type_text = " ".join([entry.kind_text, *entry.specs]).strip()
        canonical = normalize.classify(
            f"{type_text} {block.title}", block.category
        )
        items.append(
            Item(
                item_id=f"{project}.{block.block_id}.{mark}",
                category=canonical.category or block.category,
                mark=mark,
                page=page_no,
                block_id=block.block_id,
                raw={"title": entry.title, "specs": entry.specs},
                verbatim=entry.verbatim,
                width_text=entry.width_text,
                height_text=entry.height_text,
                width_in=normalize.to_inches(entry.width_text),
                height_in=normalize.to_inches(entry.height_text),
                type_text=type_text,
                canonical=canonical.to_dict(),
                specs=entry.specs,
                rect=[round(v, 1) for v in entry.rect] if entry.rect else [],
                extractor="text.pictorial",
            )
        )
    return items


def _from_legend(page, block: segment.Block, page_no: int, project: str) -> list[Item]:
    """Emit a types legend as items, for sheets that carry no matching table."""
    items: list[Item] = []
    for entry in legend.parse(page, block):
        canonical = normalize.classify(
            f"{entry.caption} {block.title}", block.category, entry.code
        )
        canonical.mapped_by = "legend"
        items.append(
            Item(
                item_id=f"{project}.{block.block_id}.{entry.code}",
                category=canonical.category or block.category,
                mark=entry.code,
                page=page_no,
                block_id=block.block_id,
                raw={"code": entry.code, "caption": entry.caption},
                verbatim=f"TYPE {entry.code} {entry.caption}",
                type_code=entry.code,
                type_text=entry.caption,
                canonical=canonical.to_dict(),
                rect=[round(v, 1) for v in entry.rect],
                extractor="text.legend",
            )
        )
    return items


def run_one(pdf_path: Path, record: dict, candidates: list[dict]) -> dict:
    """Extract every schedule item from one planset."""
    project = record["project_id"]
    out: dict = {
        "schema_version": "1.0",
        "source": {
            "file": pdf_path.name,
            "project_id": project,
            "sha256": record["sha256"],
            "page_count": record["page_count"],
            "producer": record["producer"],
            "has_text_layer": record["has_text_layer"],
            "health": record["health"],
        },
        "sheets": [],
        "items": [],
        "warnings": [],
    }

    if record["health"] in {"truncated", "unopenable"}:
        out["warnings"].append(f"{record['health']}: {record.get('note', '')}")
        return out
    if not candidates:
        out["warnings"].append("no schedule sheet found")
        return out

    with pymupdf.open(pdf_path) as doc:
        for candidate in candidates:
            page_no = candidate["page_index"] + 1
            page = doc[candidate["page_index"]]
            try:
                blocks = genre.annotate(
                    page, segment.segment(page, locate.PageCandidate(
                        page_index=candidate["page_index"],
                        score=candidate["score"],
                        signals=candidate["signals"],
                        titles=[locate.TitleHit(**t) for t in candidate["titles"]],
                    ))
                )
            except Exception as exc:
                out["warnings"].append(f"p{page_no}: segmentation failed: {exc}")
                continue

            sheet = {
                "page": page_no,
                "sheet_name": candidate.get("sheet_name", ""),
                "rotation": candidate.get("rotation", 0),
                "score": candidate["score"],
                "signals": candidate["signals"],
                "blocks": [],
            }

            # Legends are project-local and sheet-local, so build the lookup
            # from this page before parsing any table on it.
            legends = legend.build_map(page, blocks)
            if legends:
                sheet["legends"] = legends

            # A types legend defines vocabulary; it is not a list of installed
            # openings. Its entries are already folded into `legends` above and
            # applied to the table rows, so emitting them as items too would
            # double-count. The exception is a sheet that shows a legend and no
            # matching schedule -- there the legend *is* the type list.
            scheduled = {
                b.category for b in blocks
                if b.kind == "schedule" and b.genre == "tabular"
            }

            for block in blocks:
                sheet["blocks"].append(block.to_dict())
                if not block.is_parseable:
                    continue
                if block.kind == "type_legend":
                    if block.category not in scheduled:
                        out["items"].extend(
                            _from_legend(page, block, page_no, project)
                        )
                    continue
                try:
                    if block.genre == "tabular":
                        parsed = table.parse(page, block)
                        if parsed is None:
                            out["warnings"].append(
                                f"p{page_no} {block.title!r}: tabular parse found nothing"
                            )
                            continue
                        out["items"].extend(
                            _from_table(parsed, block, page_no, project, legends)
                        )
                    else:
                        entries = pictorial.parse(page, block)
                        if not entries:
                            out["warnings"].append(
                                f"p{page_no} {block.title!r}: pictorial parse found nothing"
                            )
                            continue
                        out["items"].extend(
                            _from_pictorial(entries, block, page_no, project)
                        )
                except Exception as exc:
                    out["warnings"].append(
                        f"p{page_no} {block.title!r}: {type(exc).__name__}: {exc}"
                    )

            out["sheets"].append(sheet)

    out["items"] = [i.to_dict() for i in out["items"]]
    out["counts"] = {
        "schedule_sheets": len(out["sheets"]),
        "blocks": sum(len(s["blocks"]) for s in out["sheets"]),
        "items": len(out["items"]),
        "needs_review": sum(
            1 for i in out["items"] if i["canonical"].get("needs_review")
        ),
    }
    return out


def run(manifest_path: Path, candidates_path: Path, out_dir: Path) -> list[dict]:
    records = {Path(r["path"]).name: r for r in manifest_mod.load(manifest_path)}
    with candidates_path.open() as handle:
        found = {
            json.loads(line)["file"]: json.loads(line)["candidates"]
            for line in handle if line.strip()
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    for name, record in records.items():
        result = run_one(Path(record["path"]), record, found.get(name, []))
        (out_dir / f"project_{record['project_id']}_{Path(name).stem[-8:]}.json").write_text(
            json.dumps(result, indent=2)
        )
        summaries.append(
            {
                "file": name,
                "project_id": record["project_id"],
                "health": record["health"],
                **result.get("counts", {}),
                "warnings": len(result["warnings"]),
            }
        )
    return summaries
