"""Find the sheets that actually carry a door or window schedule (stage S1).

Three independent signals, fused. They are complementary rather than
redundant: measured across the corpus, some plansets are found only by their
bookmarks, many only by title typography, and a few only by spotting a header
row. Any one signal alone leaves plansets on the table.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import pymupdf

from . import geom, lexicon, pdfio

# A title must be at least this many times the page's body-text size. Real
# schedule titles run 19-28pt on most sheets but only ~12.5pt on project_682
# and project_564, so an absolute threshold misses them; scaling off the modal
# size catches both.
TITLE_SIZE_RATIO = 1.35
TITLE_SIZE_FLOOR = 11.0

# Two title lines join only if they are close in size and vertically adjacent.
TITLE_MERGE_SIZE_TOL = 0.10
TITLE_MERGE_GAP_FACTOR = 1.6

# A header row needs this many distinct header words, at least some of which
# are not shared with MEP schedules.
HEADER_MIN_HITS = 6
HEADER_MIN_UNAMBIGUOUS = 2
HEADER_BAND_HEIGHT = 5.0
# ...and a door/window word within this distance, or lighting schedules match.
HEADER_CONTEXT_RADIUS = 300.0

KEEP_SCORE = 0.6


@dataclass
class TitleHit:
    text: str
    rect: tuple[float, float, float, float]
    size: float
    category: str
    kind: str

    @property
    def as_rect(self) -> pymupdf.Rect:
        return pymupdf.Rect(self.rect)


@dataclass
class PageCandidate:
    page_index: int
    score: float
    signals: list[str] = field(default_factory=list)
    titles: list[TitleHit] = field(default_factory=list)
    sheet_name: str = ""
    rotation: int = 0
    page_size: tuple[float, float] = (0.0, 0.0)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["score"] = round(self.score, 3)
        return data


def _merged_title_lines(page_lines: list[geom.Line]) -> list[geom.Line]:
    """Join consecutive same-block lines of similar size into one candidate."""
    merged: list[geom.Line] = []
    index = 0
    while index < len(page_lines):
        current = page_lines[index]
        text, rect, size = current.text, pymupdf.Rect(current.rect), current.size
        lookahead = index + 1
        while lookahead < len(page_lines):
            nxt = page_lines[lookahead]
            if nxt.block != current.block:
                break
            if abs(nxt.size - size) > size * TITLE_MERGE_SIZE_TOL:
                break
            if nxt.rect.y0 - rect.y1 > size * TITLE_MERGE_GAP_FACTOR:
                break
            text = f"{text} {nxt.text}"
            rect |= nxt.rect
            lookahead += 1
        merged.append(geom.Line(rect, text, size, current.block))
        # Emit the merged form *and* keep scanning from the next line, so a
        # single-line title inside a multi-line block is not lost.
        index += 1
    return merged


def _body_size(page_lines: list[geom.Line]) -> float:
    """Modal line size, as a stand-in for body-text size.

    Derived from lines rather than spans so the page is only extracted once --
    across 4,030 pages the extra pass is not free.
    """
    counts: dict[float, int] = {}
    for line in page_lines:
        key = round(line.size, 1)
        counts[key] = counts.get(key, 0) + len(line.text)
    return max(counts.items(), key=lambda kv: kv[1])[0] if counts else 10.0


def _title_signal(page: pymupdf.Page) -> tuple[float, list[TitleHit]]:
    page_lines = geom.lines(page)
    if not page_lines:
        return 0.0, []
    threshold = max(TITLE_SIZE_FLOOR, _body_size(page_lines) * TITLE_SIZE_RATIO)

    hits: list[TitleHit] = []
    seen: set[str] = set()
    for line in _merged_title_lines(page_lines):
        if line.size < threshold:
            continue
        score = lexicon.title_score(line.text)
        if score <= 0:
            continue
        key = f"{round(line.rect.x0)}:{round(line.rect.y0)}:{line.text.upper()}"
        if key in seen:
            continue
        seen.add(key)
        hits.append(
            TitleHit(
                text=line.text,
                rect=tuple(round(v, 1) for v in line.rect),
                size=round(line.size, 1),
                category=lexicon.category_of(line.text),
                kind=lexicon.kind_of(line.text),
            )
        )

    if not hits:
        return 0.0, []

    # Deferred deliberately: titleblock_x reads page.get_drawings(), and these
    # CAD sheets carry tens of thousands of drawing ops (one page has 74,000).
    # Paying that on every one of 4,030 pages dominates the whole run, so it is
    # only worth it once a page has something to classify.
    titleblock_x = geom.titleblock_x(page)
    on_sheet = [h for h in hits if h.as_rect.x0 < titleblock_x]
    in_titleblock = [h for h in hits if h.as_rect.x0 >= titleblock_x]

    score = 0.0
    if on_sheet:
        strong = [h for h in on_sheet if lexicon.title_score(h.text) >= 1.0]
        score += 0.6 if strong else 0.3
    if in_titleblock:
        # The sheet being *named* a schedule is corroboration, not proof.
        score += 0.15
    return min(score, 0.75), on_sheet


def _header_signal(page: pymupdf.Page) -> float:
    """Look for a row that reads like a schedule header."""
    page_words = geom.words(page)
    if not page_words:
        return 0.0

    bands: dict[int, list[geom.Word]] = {}
    for word in page_words:
        bands.setdefault(int(word.y0 // HEADER_BAND_HEIGHT), []).append(word)

    for band in bands.values():
        total, unambiguous = lexicon.header_strength(w.text for w in band)
        if total < HEADER_MIN_HITS or unambiguous < HEADER_MIN_UNAMBIGUOUS:
            continue
        band_rect = geom.union([w.rect for w in band])
        near = pymupdf.Rect(
            band_rect.x0 - HEADER_CONTEXT_RADIUS,
            band_rect.y0 - HEADER_CONTEXT_RADIUS,
            band_rect.x1 + HEADER_CONTEXT_RADIUS,
            band_rect.y1 + HEADER_CONTEXT_RADIUS,
        )
        context = " ".join(
            w.text for w in page_words if pymupdf.Rect(w.rect).intersects(near)
        )
        if lexicon.CONTEXT_WORDS.search(context):
            return 0.5
    return 0.0


def _toc_pages(manifest: dict) -> dict[int, str]:
    """Bookmarked pages that name a schedule, mapped to their sheet name."""
    out: dict[int, str] = {}
    for page, title in manifest.get("toc", []):
        if page is None or page < 0:
            continue
        if lexicon.title_score(title) > 0:
            out[int(page)] = title
    return out


def locate(pdf_path: Path, manifest: dict) -> list[PageCandidate]:
    """Score every page of one planset; return the pages worth parsing."""
    if manifest.get("health") in {"truncated", "unopenable"}:
        return []

    toc_hits = _toc_pages(manifest)
    candidates: list[PageCandidate] = []

    with pymupdf.open(pdf_path) as doc:
        for index, page in enumerate(doc):
            score = 0.0
            signals: list[str] = []

            if index in toc_hits:
                score += 0.5
                signals.append("toc")

            try:
                title_score, titles = _title_signal(page)
            except Exception:
                title_score, titles = 0.0, []
            if title_score > 0:
                score += title_score
                signals.append("title_font")

            # The header sweep touches every word on the page, so only pay for
            # it when the cheaper signals left the page borderline.
            if 0 < score < KEEP_SCORE or (score == 0 and titles):
                try:
                    header = _header_signal(page)
                except Exception:
                    header = 0.0
                if header > 0:
                    score += header
                    signals.append("header_lexicon")

            if score < KEEP_SCORE:
                continue

            candidates.append(
                PageCandidate(
                    page_index=index,
                    score=score,
                    signals=signals,
                    titles=titles,
                    sheet_name=toc_hits.get(index, ""),
                    rotation=page.rotation,
                    page_size=(round(page.rect.width, 1), round(page.rect.height, 1)),
                )
            )
    return candidates


def run(manifest_path: Path, out_path: Path) -> dict[str, list[PageCandidate]]:
    """Locate schedules across the whole corpus."""
    from . import manifest as manifest_mod

    records = manifest_mod.load(manifest_path)
    results: dict[str, list[PageCandidate]] = {}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as handle:
        for record in records:
            path = Path(record["path"])
            try:
                found = locate(path, record)
            except Exception as exc:
                found = []
                record_note = f"{type(exc).__name__}: {exc}"[:200]
            else:
                record_note = ""
            results[path.name] = found
            handle.write(
                json.dumps(
                    {
                        "file": path.name,
                        "project_id": record["project_id"],
                        "health": record["health"],
                        "page_count": record["page_count"],
                        "error": record_note,
                        "candidates": [c.to_dict() for c in found],
                    }
                )
                + "\n"
            )
            handle.flush()
    return results
