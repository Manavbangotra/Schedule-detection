"""Opening plansets safely, and describing what we got.

The corpus is not uniform: one file is a truncated S3 download, several have no
text layer at all, and one of those is a 36 DPI scan that nothing will recover.
Every consumer needs to know which case it is holding before it starts work, so
health triage happens once, here.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path

import pymupdf

# A page needs more than this many characters before we call it text-bearing.
# Title blocks and sheet numbers alone clear a handful of characters even on a
# pure scan.
TEXT_PAGE_MIN_CHARS = 200

# Below this share of text-bearing pages a file cannot be parsed from its text
# layer and has to go to the vision path.
TEXT_LAYER_MIN_RATIO = 0.10


@dataclass
class Manifest:
    """Everything known about a planset before any schedule work starts."""

    path: str
    project_id: str
    sha256: str
    size_bytes: int
    health: str  # ok | truncated | no_text_layer | unreadable_scan | unopenable
    page_count: int = 0
    producer: str = ""
    text_pages: int = 0
    rotations: dict[int, int] = field(default_factory=dict)
    page_sizes: dict[str, int] = field(default_factory=dict)
    toc: list[tuple[int, str]] = field(default_factory=list)
    mean_chars_per_page: float = 0.0
    note: str = ""

    @property
    def has_text_layer(self) -> bool:
        if not self.page_count:
            return False
        return self.text_pages / self.page_count >= TEXT_LAYER_MIN_RATIO

    def to_dict(self) -> dict:
        data = asdict(self)
        data["rotations"] = {str(k): v for k, v in self.rotations.items()}
        data["has_text_layer"] = self.has_text_layer
        return data


def project_id(path: Path) -> str:
    """`project_486_20260526_183817_0d86fa84.pdf` -> `486`.

    Several projects appear twice as revision pairs, so callers that build
    train/eval splits must group on this rather than on filename.
    """
    parts = path.stem.split("_")
    return parts[1] if len(parts) > 1 and parts[0] == "project" else path.stem


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _resolve_toc(doc: pymupdf.Document) -> list[tuple[int, str]]:
    """Bookmarks as ``(page_index, title)``.

    Uses PyMuPDF rather than ``qpdf --json``: these outlines use named
    destinations, which qpdf reports as a null page for every single entry.
    Entries that still fail to resolve keep their title -- the sheet name is
    useful even without a page.
    """
    out: list[tuple[int, str]] = []
    try:
        raw = doc.get_toc(simple=True)
    except Exception:
        return out
    seen: set[tuple[int, str]] = set()
    for entry in raw:
        if len(entry) < 3:
            continue
        title = str(entry[1]).strip()
        page = int(entry[2]) - 1  # get_toc is 1-based; -1 stays negative
        key = (page, title.upper())
        if not title or key in seen:
            continue
        seen.add(key)
        out.append((page, title))
    return out


def inspect(path: Path) -> Manifest:
    """Open a planset and triage it. Never raises."""
    stat = path.stat()
    base = Manifest(
        path=str(path),
        project_id=project_id(path),
        sha256=sha256(path),
        size_bytes=stat.st_size,
        health="ok",
    )

    try:
        doc = pymupdf.open(path)
    except Exception as exc:
        base.health = "unopenable"
        base.note = f"{type(exc).__name__}: {exc}"[:200]
        return base

    with doc:
        base.page_count = doc.page_count
        base.producer = (doc.metadata or {}).get("producer", "") or ""

        # A truncated download opens cleanly but reports zero pages. This is
        # how project_725 presents -- exactly 8 MiB, nothing readable.
        if base.page_count == 0:
            base.health = "truncated"
            base.note = "opens but reports 0 pages; re-download from S3"
            return base

        base.toc = _resolve_toc(doc)

        rotations: Counter[int] = Counter()
        sizes: Counter[str] = Counter()
        total_chars = 0
        image_pages = 0

        for page in doc:
            rotations[page.rotation] += 1
            rect = page.rect
            sizes[f"{round(rect.width)}x{round(rect.height)}"] += 1
            try:
                text = page.get_text("text")
            except Exception:
                text = ""
            total_chars += len(text)
            if len(text.strip()) >= TEXT_PAGE_MIN_CHARS:
                base.text_pages += 1
            try:
                if page.get_images():
                    image_pages += 1
            except Exception:
                pass

        base.rotations = dict(rotations)
        base.page_sizes = dict(sizes)
        base.mean_chars_per_page = round(total_chars / base.page_count, 1)

        if not base.has_text_layer:
            # Distinguish a real scan from CAD text converted to vector
            # outlines: the latter has no raster images and renders crisply at
            # any zoom, so a VLM reads it well. A genuine low-DPI scan does not.
            if image_pages > base.page_count / 2:
                base.health = "unreadable_scan"
                base.note = "raster scan with no text layer; check resolution"
            else:
                base.health = "no_text_layer"
                base.note = "vector-outlined text; renders sharply, use vision path"

    return base


def open_page(path: str | Path, page_index: int) -> tuple[pymupdf.Document, pymupdf.Page]:
    """Open one page. Caller closes the document."""
    doc = pymupdf.open(path)
    return doc, doc[page_index]
