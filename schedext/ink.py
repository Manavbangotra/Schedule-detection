"""Vector-stroke extents, for regions whose words do not cover their drawing.

`segment.py` builds a region from the extent of its *words*. On a schedule table
the words are the content, so that is right. On a pictorial legend -- a grid of
window elevations with a handful of dimension labels -- the drawing carries the
meaning and the words are a sparse annotation of it, so the region collapses onto
the labels and the openings fall outside it.

Measured with `eval/coverage.py` before this module existed: pictorial blocks
delivered 56.3% of their openings to stage 2, tabular 11.5%, overall 41.7%
against a human ceiling of 96.5%.

`page.get_drawings()` exposes every stroke, and it is cheap -- 0.0-0.2s on pages
carrying 1,200 to 15,000 paths -- so it is affordable on the ~150 located sheets.
It is NOT affordable on all 4,030 pages, which is why `geom.titleblock_x` already
defers it and why nothing here runs before a page is a candidate.
"""

from __future__ import annotations

import pymupdf

# A stroke this large is the sheet frame or a drawing border, not content. Same
# ratio segment.MAX_COMPONENT_AREA_RATIO uses on word components.
MAX_STROKE_SPAN = 0.55
# Strokes closer than this to a region are treated as part of it: dimension ticks
# and leader lines sit just outside the drawing they annotate.
INK_PAD_PT = 4.0
# A region may not balloon past this multiple of its word-derived area. Without
# it one leader line running to a detail bubble drags the region across the sheet.
MAX_GROWTH_RATIO = 3.5


def ink_boxes(page: pymupdf.Page, titleblock_x: float | None = None) -> list[pymupdf.Rect]:
    """Bounding box of every content stroke, in **display** space.

    Display space because that is what every rect in the pipeline uses and what
    ``get_pixmap(clip=)`` expects; ``get_drawings`` reports unrotated coordinates,
    so each rect goes through ``page.rotation_matrix``.
    """
    from . import geom

    if titleblock_x is None:
        titleblock_x = geom.titleblock_x(page)
    matrix = page.rotation_matrix
    max_w = page.rect.width * MAX_STROKE_SPAN
    max_h = page.rect.height * MAX_STROKE_SPAN

    boxes = []
    try:
        drawings = page.get_drawings()
    except Exception:
        return boxes

    for path in drawings:
        rect = pymupdf.Rect(path["rect"]) * matrix
        if rect.is_empty or rect.x0 >= titleblock_x:
            continue
        if rect.width > max_w or rect.height > max_h:
            continue
        boxes.append(rect)
    return boxes


def _area(rect) -> float:
    return max(0.0, rect[2] - rect[0]) * max(0.0, rect[3] - rect[1])


def grow_to_ink(block_rect: pymupdf.Rect, ink: list[pymupdf.Rect],
                bounds: pymupdf.Rect | None = None,
                max_growth: float = MAX_GROWTH_RATIO) -> tuple[pymupdf.Rect, float]:
    """Expand a region into the ink connected to it. Returns (rect, growth_ratio).

    Connected, not a bounding box over every stroke on the page: a region grows
    into strokes it touches, then into strokes those touch, and stops. That is
    what keeps a window legend from swallowing the floor plan beside it.

    ``bounds`` clips the result -- callers pass the gap to the next block's title
    so a region can never annex its neighbour's content.
    """
    if not ink:
        return pymupdf.Rect(block_rect), 1.0

    start_area = max(_area(block_rect), 1e-6)
    limit = start_area * max_growth
    current = pymupdf.Rect(block_rect)
    remaining = list(ink)

    while True:
        probe = pymupdf.Rect(current)
        probe.x0 -= INK_PAD_PT
        probe.y0 -= INK_PAD_PT
        probe.x1 += INK_PAD_PT
        probe.y1 += INK_PAD_PT

        touching, rest = [], []
        for rect in remaining:
            (touching if rect.intersects(probe) else rest).append(rect)
        if not touching:
            break

        grown = pymupdf.Rect(current)
        for rect in touching:
            grown |= rect
        if bounds is not None:
            grown &= bounds
        # Stop *before* crossing the cap rather than after: a single stroke that
        # would blow the budget is exactly the leader line this guards against.
        if _area(grown) > limit:
            break
        if grown == current:
            break
        current, remaining = grown, rest

    return current, _area(current) / start_area
