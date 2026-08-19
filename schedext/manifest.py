"""Build the corpus manifest (stage S0)."""

from __future__ import annotations

import json
from pathlib import Path

from . import pdfio


def build(pdf_dir: Path, out_path: Path) -> list[pdfio.Manifest]:
    """Inspect every PDF in ``pdf_dir`` and write one JSON record per line."""
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    records: list[pdfio.Manifest] = []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as handle:
        for path in pdfs:
            record = pdfio.inspect(path)
            records.append(record)
            handle.write(json.dumps(record.to_dict()) + "\n")
            handle.flush()
    return records


def load(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarise(records: list[pdfio.Manifest]) -> str:
    total_pages = sum(r.page_count for r in records)
    by_health: dict[str, list[str]] = {}
    for record in records:
        by_health.setdefault(record.health, []).append(Path(record.path).name)

    rotated = sum(
        1 for r in records
        if any(k != 0 for k in r.rotations)
    )

    lines = [
        f"{len(records)} plansets, {total_pages} pages",
        f"{rotated} files contain at least one rotated page",
        "",
        "health:",
    ]
    for health, names in sorted(by_health.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"  {health:16s} {len(names):3d}")
        if health != "ok":
            for name in names:
                lines.append(f"                   - {name}")
    return "\n".join(lines)
