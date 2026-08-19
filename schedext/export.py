"""Flatten per-planset JSON into one CSV for takeoff (stage S6b)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

COLUMNS = [
    "project_id", "file", "page", "sheet_name", "block_id", "block_title",
    "category", "mark", "type_code", "type_text",
    "operation", "leaf_face", "material", "glazing", "fire_rating_min", "flags",
    "width_text", "height_text", "width_in", "height_in", "head_height_in",
    "quantity", "mapped_by", "needs_review", "extractor", "verbatim",
]


def rows_for(result: dict):
    source = result.get("source", {})
    titles = {
        block["block_id"]: block["title"]
        for sheet in result.get("sheets", [])
        for block in sheet.get("blocks", [])
    }
    sheet_names = {
        sheet["page"]: sheet.get("sheet_name", "")
        for sheet in result.get("sheets", [])
    }
    for item in result.get("items", []):
        canonical = item.get("canonical", {})
        yield {
            "project_id": source.get("project_id", ""),
            "file": source.get("file", ""),
            "page": item.get("page", ""),
            "sheet_name": sheet_names.get(item.get("page"), ""),
            "block_id": item.get("block_id", ""),
            "block_title": titles.get(item.get("block_id", ""), ""),
            "category": item.get("category", ""),
            "mark": item.get("mark", ""),
            "type_code": item.get("type_code", ""),
            "type_text": item.get("type_text", ""),
            "operation": canonical.get("operation") or "",
            "leaf_face": canonical.get("leaf_face") or "",
            "material": canonical.get("material") or "",
            "glazing": "|".join(canonical.get("glazing") or []),
            "fire_rating_min": canonical.get("fire_rating_min") or "",
            "flags": "|".join(canonical.get("flags") or []),
            "width_text": item.get("width_text", ""),
            "height_text": item.get("height_text", ""),
            "width_in": item.get("width_in") if item.get("width_in") is not None else "",
            "height_in": item.get("height_in") if item.get("height_in") is not None else "",
            "head_height_in": item.get("head_height_in")
            if item.get("head_height_in") is not None else "",
            "quantity": item.get("quantity", ""),
            "mapped_by": canonical.get("mapped_by", ""),
            "needs_review": int(bool(canonical.get("needs_review"))),
            "extractor": item.get("extractor", ""),
            "verbatim": item.get("verbatim", ""),
        }


def write_csv(json_dir: Path, out_path: Path) -> int:
    count = 0
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for path in sorted(json_dir.glob("*.json")):
            result = json.loads(path.read_text())
            for row in rows_for(result):
                writer.writerow(row)
                count += 1
    return count


def type_summary(json_dir: Path) -> list[dict]:
    """Distinct (category, operation, leaf_face, material) with counts."""
    tally: dict[tuple, dict] = {}
    for path in sorted(json_dir.glob("*.json")):
        result = json.loads(path.read_text())
        project = result.get("source", {}).get("project_id", "")
        for item in result.get("items", []):
            canonical = item.get("canonical", {})
            key = (
                item.get("category", ""),
                canonical.get("operation") or "",
                canonical.get("leaf_face") or "",
                canonical.get("material") or "",
            )
            entry = tally.setdefault(
                key,
                {
                    "category": key[0], "operation": key[1],
                    "leaf_face": key[2], "material": key[3],
                    "count": 0, "projects": set(),
                },
            )
            entry["count"] += 1
            entry["projects"].add(project)
    out = []
    for entry in tally.values():
        entry["projects"] = len(entry["projects"])
        out.append(entry)
    return sorted(out, key=lambda e: -e["count"])
