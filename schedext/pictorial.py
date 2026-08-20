"""Parse a pictorial schedule block (stage S3b).

These blocks have no table. Each schedule item is a drawn elevation with, below
it, a circled mark, a title line, and a bullet list of specs:

    (A)                          (B)
    4'-0" X 6'-0" WINDOW         3'-0" X 6'-0" WINDOW
    - FIXED                      - SINGLE HUNG
    - VINYL                      - VINYL
    - INSULATED GLASS            - INSULATED GLASS

Reading order from ``get_text`` is useless here -- the words interleave across
columns -- so items are recovered geometrically, by clustering the left edges
of bullet lines into columns.

This is the genre the detector got most wrong: it called marks B and C above
"fixed" and "casement" when the sheet plainly says SINGLE HUNG.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field

import pymupdf

from . import geom
from .segment import Block

BULLET_CHARS = "·•▪‣●■*"
BULLET_LINE = re.compile(rf"^[{re.escape(BULLET_CHARS)}]+\s*")
# The keyed-card variant: "TYPE: PELLA LIFESTYLE", "MATERIAL: ALUM. WOOD CLAD".
KEYED_LINE = re.compile(r"^[A-Z][A-Z .]{2,16}:\s*\S")

# Marks are short and set apart: A, B, C / 01, 02 / W1 / D-3.
MARK_LINE = re.compile(r"^[A-Z]{0,2}-?\d{0,3}[A-Z]?$", re.I)

# Item titles carry either an explicit size or a nominal call ("3068" = 3'-0" x
# 6'-8"), usually followed by what the item is.
# The inch mark is written inconsistently and is sometimes an apostrophe by
# mistake -- project_559 prints its 6'-0" window as `6'-0' X 6'-0"`. Accepting
# either mark keeps the match anchored on the first dimension; without it the
# regex slides forward and reads the width as `0'`.
_FEET_INCHES = r"\d+\s*'\s*-?\s*\d*\s*[\"']?"
SIZE_TITLE = re.compile(
    rf"(?P<w>{_FEET_INCHES})\s*[xX×]\s*(?P<h>{_FEET_INCHES})\s*(?P<kind>.*)$"
)
CALL_TITLE = re.compile(r"^(?:PAIR\s+)?(?P<call>\d{4})\s+(?P<kind>.+)$", re.I)

# Columns are split where the gap between bullet left-edges exceeds this
# fraction of the typical column pitch.
COLUMN_GAP_RATIO = 0.35
# An item ends at a vertical gap this many times the usual line spacing.
ITEM_BREAK_FACTOR = 2.5
MIN_ITEMS = 2
# Marks in one row share a baseline within a few points; 12pt is wide enough for
# mixed label sizes and far tighter than the gap to the spec lines below.
MARK_BAND_PT = 12.0


@dataclass
class PictorialItem:
    mark: str
    title: str
    specs: list[str] = field(default_factory=list)
    width_text: str = ""
    height_text: str = ""
    kind_text: str = ""
    rect: pymupdf.Rect | None = None

    @property
    def verbatim(self) -> str:
        return " ".join([self.title, *self.specs]).strip()

    def to_dict(self) -> dict:
        return {
            "mark": self.mark,
            "title": self.title,
            "specs": self.specs,
            "width_text": self.width_text,
            "height_text": self.height_text,
            "kind_text": self.kind_text,
            "verbatim": self.verbatim,
            "rect": [round(v, 1) for v in self.rect] if self.rect else None,
        }


def _is_bullet(text: str) -> bool:
    return bool(BULLET_LINE.match(text)) or bool(KEYED_LINE.match(text))


def _bullet_depth(text: str) -> int:
    """Nesting level of a bullet line; 0 when it is not a bullet.

    Sub-bullets are written with a doubled glyph ("`·· XXX OUTSIDE`") and sit
    indented, so treating them as column anchors invents a phantom column
    between every real pair -- project_559's four windows came out as eight.
    """
    match = BULLET_LINE.match(text)
    if match:
        return len(match.group(0).strip())
    return 1 if KEYED_LINE.match(text) else 0


def _strip_bullet(text: str) -> str:
    return BULLET_LINE.sub("", text).strip()


def _column_bands(lines: list[geom.Line], block: Block) -> list[tuple[float, float]]:
    """Left/right boundaries of each item column."""
    # Only top-level bullets define columns.
    anchors = sorted(
        line.rect.x0 for line in lines if _bullet_depth(line.text) == 1
    )
    if len(anchors) < 2:
        # No bullets to anchor on. Measured: 44 of the 52 pictorial/schedule
        # blocks that yield nothing at all fail exactly here, and between them
        # they hold 1,897 feet-inch tokens. The marks are a second anchor and a
        # better one -- they are what a reader uses -- so fall back to their
        # centres. Uses geom.cluster_1d, which was written for this ("column
        # anchors in pictorial schedules") and never called.
        anchors = _mark_anchors(lines)
        if len(anchors) < 2:
            return []

    # Collapse near-identical left edges (all bullets in a column share one).
    edges: list[float] = []
    for value in anchors:
        if not edges or value - edges[-1] > 4.0:
            edges.append(value)
    if len(edges) < MIN_ITEMS:
        return []

    gaps = [b - a for a, b in zip(edges, edges[1:])]
    if not gaps:
        return []
    pitch = statistics.median(gaps)
    if pitch <= 0:
        return []

    starts = [edges[0]]
    for previous, current in zip(edges, edges[1:]):
        if current - previous > pitch * COLUMN_GAP_RATIO:
            starts.append(current)
    if len(starts) < MIN_ITEMS:
        return []

    bands: list[tuple[float, float]] = []
    for index, start in enumerate(starts):
        left = start - pitch * 0.35 if index == 0 else (starts[index - 1] + start) / 2
        right = (start + starts[index + 1]) / 2 if index + 1 < len(starts) else block.rect.x1
        bands.append((max(left, block.rect.x0), min(right, block.rect.x1)))
    return bands


def _mark_anchors(lines: list[geom.Line]) -> list[float]:
    """Column anchors from the mark row, for blocks with no bullets.

    A pictorial schedule labels every item -- 01, 02, A, B -- and those labels
    share one horizontal band under the drawings. Clustering their *centres*
    rather than left edges is what matters: a mark is centred under its drawing
    the way a title is centred over its table, so left edges drift with the
    width of the label while centres do not.

    Returns left edges so the caller's pitch arithmetic is unchanged.
    """
    marks = [line for line in lines if MARK_LINE.match(line.text.strip())]
    if len(marks) < MIN_ITEMS:
        return []
    # The mark row is the band holding the most marks; anything else is a stray
    # code in a spec list.
    bands = geom.cluster_1d([m.rect.y0 for m in marks], MARK_BAND_PT)
    if not bands:
        return []
    row = max(bands, key=len)
    if len(row) < MIN_ITEMS:
        return []
    return sorted(marks[i].rect.x0 for i in row)


def _tidy_dimension(text: str) -> str:
    """`` 6' - 0' `` -> ``6'-0"``, fixing the stray apostrophe as it goes."""
    value = "".join(text.split())
    # Only the inches half of a feet-and-inches value may be corrected. A bare
    # "3'" is three feet and must stay that way.
    return re.sub(r"^(\d+'-\d+)'$", r'\1"', value)


def _parse_title(text: str) -> tuple[str, str, str]:
    """``(width_text, height_text, kind_text)`` from an item title."""
    match = SIZE_TITLE.search(text)
    if match:
        return (
            _tidy_dimension(match.group("w")),
            _tidy_dimension(match.group("h")),
            match.group("kind").strip(),
        )
    match = CALL_TITLE.match(text.strip())
    if match:
        # A nominal call packs feet and inches: 3068 -> 3'-0" x 6'-8".
        call = match.group("call")
        return (
            f"{call[0]}'-{call[1]}\"",
            f"{call[2]}'-{call[3]}\"",
            match.group("kind").strip(),
        )
    return "", "", text.strip()


def parse(page: pymupdf.Page, block: Block) -> list[PictorialItem]:
    """Recover one record per drawn schedule item."""
    lines = [l for l in geom.lines(page, clip=block.rect) if l.text.strip()]
    if len(lines) < MIN_ITEMS * 2:
        return []

    bands = _column_bands(lines, block)
    if not bands:
        return []

    heights = [l.rect.height for l in lines if l.rect.height > 0]
    spacing = statistics.median(heights) if heights else 10.0

    items: list[PictorialItem] = []
    for left, right in bands:
        column = sorted(
            (l for l in lines if left <= (l.rect.x0 + l.rect.x1) / 2 < right),
            key=lambda l: l.rect.y0,
        )
        if not column:
            continue

        first_bullet = next((i for i, l in enumerate(column) if _is_bullet(l.text)), None)
        if first_bullet is None:
            continue

        # Specs run from the first bullet until the block's content breaks.
        specs: list[str] = []
        last = column[first_bullet]
        used: list[geom.Line] = []
        for line in column[first_bullet:]:
            if line.rect.y0 - last.rect.y1 > spacing * ITEM_BREAK_FACTOR:
                break
            if _is_bullet(line.text):
                specs.append(_strip_bullet(line.text))
            elif specs:
                # Continuation of the previous bullet, wrapped onto a new line.
                specs[-1] = f"{specs[-1]} {line.text.strip()}"
            last = line
            used.append(line)

        above = column[:first_bullet]
        title_line = above[-1] if above else None
        title = title_line.text.strip() if title_line else ""

        mark = ""
        for line in reversed(above[:-1] if above else []):
            candidate = line.text.strip()
            if MARK_LINE.match(candidate) and 1 <= len(candidate) <= 4:
                mark = candidate
                break

        width, height, kind = _parse_title(title)
        rect = geom.union(
            [l.rect for l in used] + ([title_line.rect] if title_line else [])
        )
        items.append(
            PictorialItem(
                mark=mark,
                title=title,
                specs=specs,
                width_text=width,
                height_text=height,
                kind_text=kind,
                rect=rect,
            )
        )

    return [i for i in items if i.specs or i.title]
