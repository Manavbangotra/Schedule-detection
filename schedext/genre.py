"""Tell a tabular schedule from a pictorial one (stage S2b).

Sheets mix the two freely -- project_486 p18 has three tables *and* a pictogram
legend -- so this is decided per block, never per page.

Vector rulings looked like the obvious signal and are not: counting
axis-aligned rules near each title got four of five blocks wrong. Table column
separators are drawn as many short segments, while pictogram line art (door
panels, mullions, garage-door slats) produces more long verticals than the
tables do. The header vocabulary is what actually separates them, and it was
correct on every block tested.
"""

from __future__ import annotations

import pymupdf

from . import geom, lexicon
from .segment import Block

# How far past the title to look for a header row.
PROBE_WIDTH_PT = 620.0
PROBE_HEIGHT_PT = 110.0

# Distinct header words needed to call a block tabular.
TABULAR_MIN_HITS = 3

# Words within this many points of each other vertically count as one row.
ROW_BAND_PT = 5.0

# Share of a row's words that must be header vocabulary.
HEADER_ROW_PURITY = 0.45

# Header words that are not also ordinary English or MEP column names.
MIN_UNAMBIGUOUS = 2


def _probe_band(block: Block, page: pymupdf.Page) -> pymupdf.Rect:
    band = geom.band_below(block.title_rect, PROBE_WIDTH_PT, PROBE_HEIGHT_PT)
    # Never look outside the block we were given, or a neighbouring table's
    # header leaks in and every block reads as tabular.
    return (band & block.rect) & page.rect


def _best_header_row(page: pymupdf.Page, rect: pymupdf.Rect) -> set[str]:
    """Most header-like single row of text inside ``rect``.

    Collinearity is the point. A table puts its header words side by side on
    one line; a pictorial block scatters the same vocabulary down its spec
    bullets -- project_559's door block contains "HARDWARE", "LOCATION" and
    "HEIGHT", but never together on a row. Counting words anywhere in the block
    calls every pictorial block a table.
    """
    if rect.is_empty:
        return set()

    bands: dict[int, list[str]] = {}
    for word in geom.words(page, clip=rect):
        bands.setdefault(int(word.y0 // ROW_BAND_PT), []).append(word.text)

    best: set[str] = set()
    for tokens in bands.values():
        meaningful = [t for t in tokens if any(ch.isalnum() for ch in t)]
        if not meaningful:
            continue
        hits = lexicon.header_hits(meaningful)
        if len(hits - lexicon.AMBIGUOUS_HEADERS) < MIN_UNAMBIGUOUS:
            continue
        # Purity separates a header row from a spec bullet that merely mentions
        # column-ish words. Measured against the raw word count, not the
        # distinct set: project_559 repeats "U-FACTOR & SHGC FACTOR TO" once per
        # column, so by distinct words that bullet looks pure, and by total
        # words it is three hits in twenty-four.
        if len(hits) / len(meaningful) < HEADER_ROW_PURITY:
            continue
        if len(hits) > len(best):
            best = hits
    return best


def classify(page: pymupdf.Page, block: Block) -> str:
    """``"tabular"`` or ``"pictorial"``."""
    # Prefer the band just under the title: it is where a header row lives, and
    # scoping tightly keeps a neighbouring table from voting.
    band = _probe_band(block, page)
    if len(_best_header_row(page, band)) >= TABULAR_MIN_HITS:
        return "tabular"

    # A tall table may carry its header further down than the probe reaches, so
    # fall back to the best row anywhere in the block -- still one row, not the
    # whole block.
    if len(_best_header_row(page, block.rect)) >= TABULAR_MIN_HITS:
        return "tabular"

    return "pictorial"


def annotate(page: pymupdf.Page, blocks: list[Block]) -> list[Block]:
    """Set ``genre`` on each block in place."""
    for block in blocks:
        try:
            block.genre = classify(page, block)
        except Exception:
            block.genre = "unknown"
    return blocks
