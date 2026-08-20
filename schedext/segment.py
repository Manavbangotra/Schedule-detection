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


# A component is assigned to the title it most plausibly belongs to. Vertical gap
# counts once, horizontal misalignment is weighted separately.
#
# These two numbers together imply a third that nobody wrote down: the furthest
# a component's left edge may sit from its title's left edge, even at zero
# vertical gap, is MAX_ASSIGN_COST_PT / X_MISALIGN_WEIGHT. At the old 900/1.6
# that was 562pt on sheets 2592-3024pt wide -- and window regions, which are wide
# short strips low on the sheet, need a median reach of 859pt.
#
# The consequence was measured, not guessed: of the word components whose centre
# falls inside a human-verified region, the cost cap discarded 50.7% of window
# and 31.7% of door components. The median cost of a component that genuinely
# belongs to a window region was 1166 -- the cap sat below the median of the
# distribution it was meant to accept.
#
# The human corrections say the same thing directionally: on edited areas the
# right edge moved out by a median of 228pt while the left edge moved 1.7pt. The
# region starts in the right place and stops too early, so the horizontal axis
# was the one being over-penalised.
#
# Swept against BOTH eval/coverage.py and eval/test_extract.py, because they
# disagree: a wider cap lifts coverage and simultaneously merges neighbouring
# blocks, which breaks table parsing on side-by-side schedules and column
# detection on pictorial ones. 3000 scores 68.8% coverage with 3 regressions;
# 1500 scores 66.2% with 1. Coverage is a corpus-wide average and the
# regression test is specific ground truth, so the specific evidence wins.
X_MISALIGN_WEIGHT = 0.8
MAX_ASSIGN_COST_PT = 1500.0
# Components larger than this share of the page are sheet borders or the
# drawing frame, not schedule content.
MAX_COMPONENT_AREA_RATIO = 0.55
# A title sitting this far down the page is a *caption* under its drawing, not
# a heading over it -- the architectural convention of naming a detail beneath
# it. Six located sheets segmented to zero blocks for this reason alone,
# including "WINDOW TYPES" (project_716 p14, title 95% down the page) and
# "WINDOW ELEVATIONS" (project_699 p85, 96% down), which are exactly the
# window legends this corpus is short of.
#
# Lowered from 0.80 after measurement: the rule also discards 12.9% of the window
# components and 14.6% of the door components that provably belong inside a
# human-verified region, because a title only has to be *below its own drawing*,
# not near the foot of the sheet. Window regions sit around 0.67 down the page
# and straddled the old threshold. 0.50 recovers them, but it also let
# project_559 p1's WINDOW SCHEDULE (title 53% down the page) claim the door
# schedule above it and lose marks A and B. 0.65 keeps the recovery and the
# ground truth: a title past two thirds of the sheet is a caption, one at half
# way is still a heading.
CAPTION_BAND = 0.65
# A block may be cut at an internal whitespace gutter only when that gutter is an
# outlier: at least GUTTER_MIN_PT wide AND GUTTER_RATIO times the next widest gap
# in the same block. Both conditions matter -- the median block already carries
# an 87pt gap between table columns, so width alone would shred real tables.
GUTTER_RATIO = 3.0
# Swept against coverage, wrong-class rate, region area and test_extract:
#   100pt  57.0% coverage,  77 wrong,  9.2% area, tests green
#   150pt  60.8% coverage,  92 wrong,  9.6% area, tests green   <- chosen
#   450pt  64.9% coverage, 119 wrong, 10.5% area, 1 failure
#   off    66.2% coverage, 121 wrong, 12.6% area, 1 failure
# Headline coverage is highest with the guard off, and that is the one measure
# it wins: turning it on drops 5.4 points of coverage but removes 29 wrong-class
# openings, tightens regions towards the human 7.2%, and turns the suite green.
# A wrong class is worse than a missing one -- it is inherited by every opening
# in the region -- and a smaller region also keeps openings larger in the
# stage-2 crop, where they are already only 23px at p10.
GUTTER_MIN_PT = 150.0


def _blocked_by_other_title(rect: pymupdf.Rect, anchor_title, titles, own: int) -> bool:
    """True when another title's centre sits between the component and its title.

    The cost function picks the *nearest* title, but nearest is not the same as
    unobstructed: with a generous cost cap a block will happily reach across a
    neighbouring schedule to claim content on its far side. Measured on
    project_486 p18, "DOOR SCHEDULE - COMMON AREA" grew from 697pt to 1897pt by
    reaching over "DOOR HARDWARE TYPES", and `table.parse` then found no
    coherent table at all.

    Titles are compared on their *centres*, for the reason `_resolve_overlaps`
    already documents: a schedule title is centred over its table, not
    left-aligned. Only titles sharing the component's vertical band count -- a
    title on another row of the sheet obstructs nothing.
    """
    anchor = anchor_title.as_rect
    own_x = (anchor.x0 + anchor.x1) / 2
    comp_x = (rect.x0 + rect.x1) / 2
    low, high = min(own_x, comp_x), max(own_x, comp_x)
    for index, other in enumerate(titles):
        if index == own:
            continue
        rival = other.as_rect
        # Same band: the rival's title must sit near the component vertically,
        # otherwise it belongs to a different row of the sheet entirely.
        if rival.y0 > rect.y1 or rival.y1 < anchor.y0 - 2:
            continue
        rival_x = (rival.x0 + rival.x1) / 2
        if low < rival_x < high:
            return True
    return False


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
            if rect.y1 >= anchor.y0 - 2:
                # The normal case: content sits under its heading. The test is
                # on the component's *bottom* edge, not its top -- a component
                # frequently overlaps its own title vertically, and requiring it
                # to start below the title drops ordinary schedules wholesale.
                gap = max(0.0, rect.y0 - anchor.y1)
            elif anchor.y0 > caption_y:
                # Caption: the title is near the foot of the sheet, so its
                # drawing is above it. Deliberately not allowed for titles
                # higher up -- there, content above a title belongs to whatever
                # heading precedes it, and claiming it would merge two blocks.
                gap = max(0.0, anchor.y0 - rect.y1)
            else:
                continue
            cost = gap + abs(rect.x0 - anchor.x0) * X_MISALIGN_WEIGHT
            if cost >= best_cost:
                continue
            if _blocked_by_other_title(rect, title, titles, index):
                continue
            best_index, best_cost = index, cost

        if best_index is not None:
            owned.setdefault(best_index, []).append(rect)

    return owned


def _trim_gutter(block: "Block", words: list[geom.Word],
                 ratio: float | None = None, floor: float | None = None) -> None:
    """Cut a block at an outlier whitespace gutter, in place.

    A block can reach past its own content into untitled material beside it,
    where `_blocked_by_other_title` has no rival title to fire on. project_486
    p18's "DOOR SCHEDULE - COMMON AREA" spans 877-2774pt that way and
    `table.parse` then finds no coherent table at all.

    A flat width threshold cannot do this: measured over 138 blocks the *median*
    block already has an 87pt internal gap, because schedule tables have wide
    column gutters, and cutting at 100pt would fire on 41% of them. What marks
    the real boundary is that the gap is an **outlier** -- on 486 p18 it is 197pt
    against a next-widest of 41pt.

    So cut only when the widest gap both exceeds ``floor`` and is ``ratio`` times
    the second widest, and always keep the side holding the title.
    """
    # Read the module constants at call time, not as default arguments: a
    # default binds once at import and a sweep that reassigns the constant then
    # measures the same run six times over. That mistake cost an hour here.
    ratio = GUTTER_RATIO if ratio is None else ratio
    floor = GUTTER_MIN_PT if floor is None else floor

    inside = [w for w in words if block.rect.contains(w.rect)]
    if len(inside) < 5:
        return
    # Both axes: a block over-reaches sideways into a neighbouring table and
    # downwards into the next one. On 486 p18 the horizontal cut alone left the
    # block 817pt tall against a true 208pt, and the extra rows merged the
    # header tiers just as badly as the extra columns did.
    _trim_axis(block, inside, ratio, floor, horizontal=True)
    _trim_axis(block, [w for w in inside if block.rect.contains(w.rect)],
               ratio, floor, horizontal=False)


def _trim_axis(block: "Block", inside: list[geom.Word], ratio: float,
               floor: float, horizontal: bool) -> None:
    """Keep only the band of content the block's own title sits in.

    Not "cut at the widest gap": a block that over-reaches usually spans several
    neighbours, so the separating gaps are all about the same size and no single
    one is an outlier. On 486 p18 the vertical gaps are 154pt and 152pt -- ratio
    1.0 -- and the real table is the band between them, which is exactly the one
    holding the title.

    Horizontally the floor has to be higher and the outlier test still applies,
    because a schedule table's own column gutters are wide: measured over 138
    blocks the median block already carries an 87pt horizontal gap.
    """
    lo = (lambda w: w.x0) if horizontal else (lambda w: w.y0)
    hi = (lambda w: w.x1) if horizontal else (lambda w: w.y1)
    inside = sorted(inside, key=lo)
    if len(inside) < 5:
        return

    gaps: list[tuple[float, float, float]] = []
    edge = hi(inside[0])
    for word in inside[1:]:
        if lo(word) > edge:
            gaps.append((lo(word) - edge, edge, lo(word)))
        edge = max(edge, hi(word))
    if not gaps:
        return

    wide = [g for g in gaps if g[0] >= floor]
    if not wide:
        return
    if horizontal:
        # Only an outlier gutter may split a table sideways.
        widest = max(gaps)
        others = [g[0] for g in gaps if g is not widest]
        if not others or widest[0] < max(others) * ratio:
            return
        wide = [widest]

    pad = DILATE_X_PT if horizontal else DILATE_Y_PT
    title = ((block.title_rect.x0 + block.title_rect.x1) / 2 if horizontal
             else block.title_rect.y1)
    low = max((g[2] for g in wide if g[2] <= title), default=None)
    high = min((g[1] for g in wide if g[1] >= title), default=None)

    if low is not None:
        low = max(low - pad, block.rect.y0 if not horizontal else block.rect.x0)
    if high is not None:
        high = min(high + pad, block.rect.y1 if not horizontal else block.rect.x1)

    if horizontal:
        if low is not None:
            block.rect.x0 = max(block.rect.x0, low)
        if high is not None:
            block.rect.x1 = min(block.rect.x1, high)
    else:
        if low is not None:
            block.rect.y0 = max(block.rect.y0, low)
        if high is not None:
            block.rect.y1 = min(block.rect.y1, high)


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
    for block in blocks:
        # After overlaps are settled, not before: trimming first would change
        # the rects _resolve_overlaps compares.
        _trim_gutter(block, words)

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
