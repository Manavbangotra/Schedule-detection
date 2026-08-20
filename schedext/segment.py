"""Cut a schedule sheet into blocks (stage S2).

A schedule sheet is never one table. project_486 p18 carries four schedules,
two pictogram legends and several note blocks; project_682 p26 carries seven
door tables. So the unit of work is the block, not the page.

Doing this before parsing is what makes the rest fast and accurate: clipping
``find_tables`` to a block runs ~15x quicker than a full-page call and stops
neighbouring details from contributing phantom rulings. It is also what keeps
the unit-entry elevation drawings on project_559 p1 -- the ones the detector
false-fired on -- out of the schedule.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
import pymupdf

from . import geom, lexicon
from .locate import PageCandidate, TitleHit

# Word-mask resolution. 2pt per pixel keeps a 3024x2160pt sheet at 1512x1080,
# which is plenty to separate blocks and cheap to dilate.
POINTS_PER_PIXEL = 2.0

# Dilation kernel, in points. Wide enough to bridge the gap between words and
# between adjacent table columns; narrower than the gutter between blocks.
DILATE_X_PT = 25.0
DILATE_Y_PT = 8.0

MIN_WORDS_PER_BLOCK = 15
# Blocks smaller than this in either direction are stray annotation.
MIN_BLOCK_PT = 40.0


@dataclass
class Block:
    """One titled region of a schedule sheet."""

    block_id: str
    title: str
    category: str
    kind: str
    rect: pymupdf.Rect
    title_rect: pymupdf.Rect
    word_count: int
    genre: str = "unknown"
    header_hits: list[str] = field(default_factory=list)

    @property
    def is_parseable(self) -> bool:
        """Notes and hardware keys are kept for their abbreviations, not rows."""
        return self.kind in {"schedule", "type_legend"}

    def to_dict(self) -> dict:
        return {
            "block_id": self.block_id,
            "title": self.title,
            "category": self.category,
            "kind": self.kind,
            "genre": self.genre,
            "rect": [round(v, 1) for v in self.rect],
            "title_rect": [round(v, 1) for v in self.title_rect],
            "word_count": self.word_count,
            "header_hits": sorted(self.header_hits),
        }


def _component_map(page: pymupdf.Page, words: list[geom.Word]):
    """Label connected components of dilated word boxes.

    Returns ``(labels, stats, origin, scale)`` where ``labels`` is the pixel
    label image and ``origin``/``scale`` convert between points and pixels.
    """
    rect = page.rect
    scale = 1.0 / POINTS_PER_PIXEL
    width = max(1, int(rect.width * scale) + 2)
    height = max(1, int(rect.height * scale) + 2)
    mask = np.zeros((height, width), dtype=np.uint8)

    for word in words:
        x0 = int((word.x0 - rect.x0) * scale)
        y0 = int((word.y0 - rect.y0) * scale)
        x1 = int((word.x1 - rect.x0) * scale) + 1
        y1 = int((word.y1 - rect.y0) * scale) + 1
        mask[max(0, y0):min(height, y1), max(0, x0):min(width, x1)] = 255

    kx = max(1, int(DILATE_X_PT * scale))
    ky = max(1, int(DILATE_Y_PT * scale))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx * 2 + 1, ky * 2 + 1))
    dilated = cv2.dilate(mask, kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    return count, labels, stats, (rect.x0, rect.y0), scale


def _label_at(labels, origin, scale, point: pymupdf.Point) -> int:
    col = int((point.x - origin[0]) * scale)
    row = int((point.y - origin[1]) * scale)
    if 0 <= row < labels.shape[0] and 0 <= col < labels.shape[1]:
        return int(labels[row, col])
    return 0


def _stats_rect(stats, label: int, origin, scale) -> pymupdf.Rect:
    x, y, w, h = stats[label][:4]
    return pymupdf.Rect(
        origin[0] + x / scale,
        origin[1] + y / scale,
        origin[0] + (x + w) / scale,
        origin[1] + (y + h) / scale,
    )


# A component is assigned to the title it most plausibly belongs to. Vertical
# gap counts once, horizontal misalignment counts more -- on a multi-column
# sheet a block's content starts at very nearly its title's left edge, so
# x-offset is the strongest cue for "this belongs to that other column".
X_MISALIGN_WEIGHT = 1.6
MAX_ASSIGN_COST_PT = 900.0
# Components larger than this share of the page are sheet borders or the
# drawing frame, not schedule content.
MAX_COMPONENT_AREA_RATIO = 0.55
# A title sitting this far down the page is a *caption* under its drawing, not
# a heading over it -- the architectural convention of naming a detail beneath
# it. Six located sheets segmented to zero blocks for this reason alone,
# including "WINDOW TYPES" (project_716 p14, title 95% down the page) and
# "WINDOW ELEVATIONS" (project_699 p85, 96% down), which are exactly the
# window legends this corpus is short of.
CAPTION_BAND = 0.80


def _assign_components(page: pymupdf.Page, titles: list[TitleHit],
                       count: int, stats, origin, scale,
                       titleblock_x: float) -> dict[int, list[pymupdf.Rect]]:
    """Map each title index to the component rects that belong under it.

    Titles are set apart from their content -- underlined, with clear space
    below -- so a title almost never shares a connected component with its own
    table. Growing each block outward from its title is what actually works.
    """
    page_area = page.rect.width * page.rect.height
    owned: dict[int, list[pymupdf.Rect]] = {}

    for label in range(1, count):
        rect = _stats_rect(stats, label, origin, scale)
        if rect.width * rect.height > page_area * MAX_COMPONENT_AREA_RATIO:
            continue
        if rect.x0 >= titleblock_x:
            continue

        caption_y = page.rect.y0 + page.rect.height * CAPTION_BAND
        best_index, best_cost = None, MAX_ASSIGN_COST_PT
        for index, title in enumerate(titles):
            anchor = title.as_rect
            if rect.y0 >= anchor.y1 - 2:
                # The normal case: content sits under its heading.
                gap = max(0.0, rect.y0 - anchor.y1)
            elif rect.y1 <= anchor.y0 + 2 and anchor.y0 > caption_y:
                # Caption: the title is near the foot of the sheet, so its
                # drawing is above it. Deliberately not allowed for titles
                # higher up -- there, content above a title belongs to whatever
                # heading precedes it, and claiming it would merge two blocks.
                gap = max(0.0, anchor.y0 - rect.y1)
            else:
                continue
            cost = gap + abs(rect.x0 - anchor.x0) * X_MISALIGN_WEIGHT
            if cost < best_cost:
                best_index, best_cost = index, cost

        if best_index is not None:
            owned.setdefault(best_index, []).append(rect)

    return owned


def _resolve_overlaps(blocks: list["Block"]) -> None:
    """Trim overlapping blocks apart, in place.

    Component assignment is greedy, so two side-by-side tables can each claim
    the strip between them. Splitting at the midpoint between the two titles'
    *centres* is what works here: schedule titles are centred over their table,
    not left-aligned, so comparing left edges puts the boundary in the wrong
    place (project_486 p18's DWELLING UNITS block ran 176pt into the COMMON
    AREA table beside it, and tail recovery then read the wrong table).
    """
    for i, first in enumerate(blocks):
        for second in blocks[i + 1:]:
            overlap = first.rect & second.rect
            if overlap.is_empty or overlap.width < 2 or overlap.height < 2:
                continue

            a_center = (first.title_rect.x0 + first.title_rect.x1) / 2, \
                       (first.title_rect.y0 + first.title_rect.y1) / 2
            b_center = (second.title_rect.x0 + second.title_rect.x1) / 2, \
                       (second.title_rect.y0 + second.title_rect.y1) / 2
            dx = abs(a_center[0] - b_center[0])
            dy = abs(a_center[1] - b_center[1])

            if dx >= dy:
                left, right = (first, second) if a_center[0] <= b_center[0] else (second, first)
                split = (a_center[0] + b_center[0]) / 2
                if left.rect.x0 < split < right.rect.x1:
                    left.rect.x1 = min(left.rect.x1, split)
                    right.rect.x0 = max(right.rect.x0, split)
            else:
                top, bottom = (first, second) if a_center[1] <= b_center[1] else (second, first)
                split = (a_center[1] + b_center[1]) / 2
                # Never cut a block above its own title.
                split = max(split, top.title_rect.y1 + 1)
                if top.rect.y0 < split < bottom.rect.y1:
                    top.rect.y1 = min(top.rect.y1, split)
                    bottom.rect.y0 = max(bottom.rect.y0, split)


def segment(page: pymupdf.Page, candidate: PageCandidate) -> list[Block]:
    """Turn the titles found on a page into bounded, typed blocks."""
    words = geom.words(page)
    if not words or not candidate.titles:
        return []

    count, labels, stats, origin, scale = _component_map(page, words)
    titleblock_x = geom.titleblock_x(page)

    titles = [t for t in candidate.titles if t.as_rect.x0 < titleblock_x]
    if not titles:
        return []

    owned = _assign_components(
        page, titles, count, stats, origin, scale, titleblock_x
    )

    blocks: list[Block] = []
    for index, title in enumerate(titles):
        parts = owned.get(index, [])
        rect = geom.union(parts + [title.as_rect])
        if rect is None:
            continue

        rect = rect & page.rect
        if rect.is_empty or rect.width < MIN_BLOCK_PT or rect.height < MIN_BLOCK_PT:
            continue

        blocks.append(
            Block(
                block_id="",
                title=title.text,
                category=title.category,
                kind=title.kind,
                rect=rect,
                title_rect=title.as_rect,
                word_count=0,
            )
        )

    _resolve_overlaps(blocks)

    # Word counts and header vocabulary are measured only after the boundaries
    # are final, so a block is never described by content it does not own.
    kept: list[Block] = []
    for block in blocks:
        if block.rect.width < MIN_BLOCK_PT or block.rect.height < MIN_BLOCK_PT:
            continue
        inside = [w for w in words if block.rect.contains(w.rect)]
        if len(inside) < MIN_WORDS_PER_BLOCK:
            continue
        block.word_count = len(inside)
        block.header_hits = sorted(lexicon.header_hits(w.text for w in inside))
        kept.append(block)

    kept.sort(key=lambda b: (round(b.rect.y0), b.rect.x0))
    for index, block in enumerate(kept):
        block.block_id = f"p{candidate.page_index + 1}_b{index}"
    return kept
