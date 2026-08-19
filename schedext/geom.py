"""Rotation-safe geometry.

PyMuPDF reports ``page.rect`` in *display* space (what you see when the PDF is
opened) but returns text and drawing coordinates in *unrotated* mediabox space.
For a 270-degree page those two spaces have swapped axes, so a clip rectangle
built from what you see on screen selects the wrong region entirely -- and,
more subtly, the direction "below the title" stops being +y.

15 of the first 40 plansets in this corpus have a rotated first page, so every
coordinate in the pipeline goes through this module and nothing else touches
raw coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass

import pymupdf


@dataclass(frozen=True)
class Word:
    """A single extracted word in display space."""

    rect: pymupdf.Rect
    text: str
    block: int
    line: int
    word: int

    @property
    def x0(self) -> float:
        return self.rect.x0

    @property
    def y0(self) -> float:
        return self.rect.y0

    @property
    def x1(self) -> float:
        return self.rect.x1

    @property
    def y1(self) -> float:
        return self.rect.y1


@dataclass(frozen=True)
class Span:
    """A styled text span in display space."""

    rect: pymupdf.Rect
    text: str
    size: float
    font: str


def to_display(page: pymupdf.Page, rect) -> pymupdf.Rect:
    """Map an extraction-space rect into display space."""
    return pymupdf.Rect(rect) * page.rotation_matrix


def to_extraction(page: pymupdf.Page, rect) -> pymupdf.Rect:
    """Map a display-space rect back into extraction space.

    Use this for **text** APIs only. PyMuPDF is not consistent about clip
    coordinates:

        page.get_text(clip=...)    -> unrotated / extraction space  (use this)
        page.find_tables(clip=...) -> unrotated / extraction space  (use this)
        page.get_pixmap(clip=...)  -> DISPLAY space  (pass the rect directly)

    Verified on a 270-degree page: a display clip reproduces the corresponding
    slice of the full-page render exactly, while an extraction clip returns a
    transposed crop of the wrong region.
    """
    return pymupdf.Rect(rect) * ~page.rotation_matrix


def words(page: pymupdf.Page, clip=None) -> list[Word]:
    """All words on the page, in display space, reading order.

    ``clip`` is interpreted in *display* space and converted internally, so
    callers never have to think about rotation.
    """
    extraction_clip = to_extraction(page, clip) if clip is not None else None
    raw = page.get_text("words", clip=extraction_clip)
    out = [
        Word(to_display(page, w[:4]), w[4], w[5], w[6], w[7])
        for w in raw
    ]
    out.sort(key=lambda w: (round(w.y0, 1), w.x0))
    return out


def spans(page: pymupdf.Page, clip=None) -> list[Span]:
    """All styled spans on the page, in display space."""
    extraction_clip = to_extraction(page, clip) if clip is not None else None
    data = page.get_text("dict", clip=extraction_clip)
    out: list[Span] = []
    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span["text"].strip()
                if not text:
                    continue
                out.append(
                    Span(
                        rect=to_display(page, span["bbox"]),
                        text=text,
                        size=span["size"],
                        font=span.get("font", ""),
                    )
                )
    return out


@dataclass(frozen=True)
class Line:
    """A text line in display space, with its block id for grouping."""

    rect: pymupdf.Rect
    text: str
    size: float
    block: int


def lines(page: pymupdf.Page, clip=None) -> list[Line]:
    """Text lines in display space, ordered top-to-bottom then left-to-right.

    Titles are not always one line -- project_707's schedule is titled
    "Door & Opening" / "Schedule" across two -- so callers need line structure
    and a block id to know which lines may legitimately be joined.
    """
    extraction_clip = to_extraction(page, clip) if clip is not None else None
    data = page.get_text("dict", clip=extraction_clip)
    out: list[Line] = []
    for block_no, block in enumerate(data.get("blocks", [])):
        for line in block.get("lines", []):
            parts = [s["text"] for s in line.get("spans", []) if s["text"].strip()]
            if not parts:
                continue
            text = " ".join(" ".join(parts).split())
            size = max(s["size"] for s in line["spans"])
            out.append(Line(to_display(page, line["bbox"]), text, size, block_no))
    out.sort(key=lambda l: (round(l.rect.y0, 1), l.rect.x0))
    return out


def modal_font_size(page_spans: list[Span]) -> float:
    """The page's most common font size.

    Schedule titles are set relative to body text, not at an absolute size:
    they run 19-28pt on most sheets but only 12.5-12.7pt on project_682 and
    project_564. A fixed threshold misses those, so callers scale off this.
    """
    if not page_spans:
        return 10.0
    counts: dict[float, int] = {}
    for span in page_spans:
        key = round(span.size, 1)
        counts[key] = counts.get(key, 0) + len(span.text)
    return max(counts.items(), key=lambda kv: kv[1])[0]


def flow_is_vertical(page: pymupdf.Page) -> bool:
    """True when text runs top-to-bottom in *extraction* space.

    Only meaningful for callers that still work in extraction space; anything
    using :func:`words` or :func:`spans` is already in display space where flow
    is always left-to-right.
    """
    return page.rotation in (90, 270)


def band_below(rect: pymupdf.Rect, width: float, height: float,
               pad_left: float = 40.0, pad_top: float = 5.0) -> pymupdf.Rect:
    """A probe band starting at ``rect`` and extending down-and-right.

    Always correct in display space, which is the whole point of normalising
    first: on a rotated page "below the title" is not +y in extraction space.
    """
    return pymupdf.Rect(
        rect.x0 - pad_left,
        rect.y0 - pad_top,
        rect.x0 + width,
        rect.y0 + height,
    )


def titleblock_x(page: pymupdf.Page) -> float:
    """Left edge of the sheet's title-block strip, in display space.

    The strip repeats the sheet's own name -- "DOOR-WINDOW SCHEDULE" sits in
    project_486's title block -- so anything matched inside it describes the
    sheet rather than a schedule drawn on it. Detected from the longest
    full-height rule in the right-hand fifth of the sheet, with a proportional
    fallback when the frame is not drawn as long lines.
    """
    rect = page.rect
    fallback = rect.x0 + rect.width * 0.88
    threshold_x = rect.x0 + rect.width * 0.80
    best = fallback
    try:
        drawings = page.get_drawings()
    except Exception:
        return fallback

    matrix = page.rotation_matrix
    for path in drawings:
        for item in path.get("items", []):
            if item[0] != "l":
                continue
            start = pymupdf.Point(item[1]) * matrix
            end = pymupdf.Point(item[2]) * matrix
            if abs(start.x - end.x) > 1.0:
                continue
            if abs(start.y - end.y) < rect.height * 0.6:
                continue
            if start.x >= threshold_x:
                best = min(best, start.x)
    return best


def cluster_1d(values: list[float], gap: float) -> list[list[int]]:
    """Group indices whose sorted values are separated by less than ``gap``.

    Used for column anchors in pictorial schedules and for row banding.
    Returns groups of *original* indices.
    """
    if not values:
        return []
    order = sorted(range(len(values)), key=lambda i: values[i])
    groups = [[order[0]]]
    for idx in order[1:]:
        if values[idx] - values[groups[-1][-1]] > gap:
            groups.append([idx])
        else:
            groups[-1].append(idx)
    return groups


def union(rects: list[pymupdf.Rect]) -> pymupdf.Rect | None:
    """Bounding box of several rects."""
    if not rects:
        return None
    out = pymupdf.Rect(rects[0])
    for rect in rects[1:]:
        out |= rect
    return out


def merge_overlapping(boxes: list[tuple[float, float, float, float]],
                      expansion: float = 0.07) -> list[pymupdf.Rect]:
    """Transitively union boxes that overlap once expanded by ``expansion``.

    The detector emits the whole unit *and* each sash/lite/panel -- window C on
    project_559 produces five boxes for one window. This collapses them, and on
    every region tested it recovers the exact item count. Same expansion ratio
    the existing service uses.
    """
    rects = [pymupdf.Rect(b) for b in boxes]
    n = len(rects)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def grown(r: pymupdf.Rect) -> pymupdf.Rect:
        dx, dy = r.width * expansion, r.height * expansion
        return pymupdf.Rect(r.x0 - dx, r.y0 - dy, r.x1 + dx, r.y1 + dy)

    expanded = [grown(r) for r in rects]
    for i in range(n):
        for j in range(i + 1, n):
            if expanded[i].intersects(expanded[j]):
                parent[find(i)] = find(j)

    groups: dict[int, list[pymupdf.Rect]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(rects[i])
    return [union(g) for g in groups.values()]
