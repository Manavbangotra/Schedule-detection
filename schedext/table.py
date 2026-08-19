"""Parse a tabular schedule block (stage S3a).

Built on PyMuPDF's ``find_tables``, clipped to the block: on project_486 p18 a
clipped call takes ~1.1s and returns one correct table, where the full-page
call takes ~17s and returns seven, most of them junk.

The one thing ``find_tables`` gets badly wrong here is unruled columns, and on
these sheets that is the column that matters. On project_486's window schedule
``lines_strict`` returns a clean table and silently drops COMMENTS -- the
column holding VINYL SINGLE HUNG, DECORATIVE SHUTTER, LOUVER and GRILL, which
is the whole reason we are reading the sheet. ``strategy="text"`` sees that
column but shatters the dimensions ("3'", "-", "0\"" as three cells).

So: take the ruled skeleton for its rows, then recover whatever sits outside
the ruled grid, row band by row band.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pymupdf

from . import geom, lexicon
from .segment import Block

# A recovered tail column must be at least this wide to count as real content
# rather than a ruling artefact.
MIN_TAIL_WIDTH_PT = 12.0

# A candidate table must lie mostly inside the block it was found for.
MIN_INSIDE_RATIO = 0.70

MIN_ROWS = 2
MIN_COLS = 2

# How many leading rows may be searched for header tiers. Blocks often carry a
# blank row and their own title above the real header.
HEADER_SCAN_ROWS = 5

# Marks look like A, B, C / 01, 02 / W1, D-12 / 101A.
MARK_PATTERN = re.compile(r"^[A-Z]{0,3}[-.]?\d{0,4}[A-Z]?$", re.I)
DIMENSION_PATTERN = re.compile(
    r"\d+\s*'\s*-?\s*\d*\s*\"?|\d+\s*-\s*\d+|^\d{3,4}$|\bVARIES\b", re.I
)


@dataclass
class Row:
    """One parsed schedule row."""

    cells: dict[str, str]
    verbatim: str
    rect: pymupdf.Rect
    row_index: int

    def to_dict(self) -> dict:
        return {
            "cells": self.cells,
            "verbatim": self.verbatim,
            "rect": [round(v, 1) for v in self.rect],
            "row_index": self.row_index,
        }


@dataclass
class ParsedTable:
    columns: list[str]
    rows: list[Row]
    rect: pymupdf.Rect
    parser: str
    recovered_columns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "columns": self.columns,
            "recovered_columns": self.recovered_columns,
            "parser": self.parser,
            "rect": [round(v, 1) for v in self.rect],
            "rows": [r.to_dict() for r in self.rows],
        }


def _inside_ratio(table_rect: pymupdf.Rect, block_rect: pymupdf.Rect) -> float:
    overlap = pymupdf.Rect(table_rect) & block_rect
    area = table_rect.width * table_rect.height
    if area <= 0:
        return 0.0
    return (overlap.width * overlap.height) / area if not overlap.is_empty else 0.0


def _is_header_row(cells: list[str]) -> bool:
    values = [c for c in cells if c and c.strip()]
    if len(values) < 2:
        return False
    hits = lexicon.header_hits(
        token for value in values for token in re.split(r"[\s/]+", value)
    )
    return len(hits) >= max(2, len(values) * 0.4)


def _merge_header_rows(raw_rows: list[list[str | None]]) -> tuple[list[str], int]:
    """Build qualified column names and report how many rows were consumed.

    Schedules routinely use a two-tier header -- a DOOR SIZE group spanning
    WIDTH / HEIGHT / THICK. Forward-filling the group across its ``None`` span
    gives ``DOOR SIZE.WIDTH``, which matters because project_745 has TYPE,
    MATERIAL and FINISH twice on one table (door leaf and frame); unqualified
    names would silently overwrite each other.
    """
    header_rows: list[list[str | None]] = []
    consumed = 0
    # Scan far enough to clear the block's own title rows. project_486's door
    # tables put a blank row and a "DOOR SCHEDULE - DWELLING UNITS" row above
    # the header, so the second header tier sits at index 3.
    for index, row in enumerate(raw_rows[:HEADER_SCAN_ROWS]):
        cleaned = [(c or "").strip() or None for c in row]
        if _is_header_row([c or "" for c in cleaned]):
            header_rows.append(cleaned)
            consumed = index + 1
        elif header_rows:
            break
    if not header_rows:
        return [], 0

    width = max(len(r) for r in header_rows)
    filled: list[list[str]] = []
    for depth, row in enumerate(header_rows):
        row = list(row) + [None] * (width - len(row))
        # Only group tiers forward-fill: "DOOR SIZE" legitimately spans WIDTH,
        # HEIGHT and THICK. The final tier holds leaf names, and filling those
        # rightward invents columns -- FRAME.MATERIAL would bleed into
        # FIRE RTG..MATERIAL, HARDWARE SET.MATERIAL and COMMENTS.MATERIAL.
        if depth == len(header_rows) - 1:
            filled.append([cell or "" for cell in row])
            continue
        current = ""
        out: list[str] = []
        for cell in row:
            if cell:
                current = cell
            out.append(current)
        filled.append(out)

    names: list[str] = []
    for index in range(width):
        parts: list[str] = []
        for row in filled:
            value = row[index].replace("\n", " ").strip()
            if value and (not parts or parts[-1] != value):
                parts.append(value)
        names.append(".".join(parts) if parts else f"col{index}")

    return names, consumed


def _looks_like_data(cells: list[str]) -> bool:
    values = [c.strip() for c in cells if c and c.strip()]
    if not values:
        return False
    has_mark = any(MARK_PATTERN.match(v) and len(v) <= 5 for v in values[:2])
    has_dimension = any(DIMENSION_PATTERN.search(v) for v in values)
    return has_mark or has_dimension


def _recover_tail(page: pymupdf.Page, grid: pymupdf.Rect, block: Block,
                  row_rects: list[pymupdf.Rect]) -> list[str]:
    """Read whatever sits to the right of the ruled grid, one value per row.

    This is the column ``lines_strict`` drops. On project_486's window schedule
    it is COMMENTS, holding the operation type for every mark.
    """
    right_edge = min(block.rect.x1, page.rect.x1)
    if right_edge - grid.x1 < MIN_TAIL_WIDTH_PT:
        return []

    values: list[str] = []
    for rect in row_rects:
        tail = pymupdf.Rect(grid.x1 - 1, rect.y0, right_edge, rect.y1)
        words = geom.words(page, clip=tail)
        values.append(" ".join(w.text for w in sorted(words, key=lambda w: w.x0)).strip())
    return values


def parse(page: pymupdf.Page, block: Block) -> ParsedTable | None:
    """Parse one tabular block, or return ``None`` if nothing usable is found."""
    clip = geom.to_extraction(page, block.rect)
    try:
        finder = page.find_tables(clip=clip, strategy="lines_strict")
    except Exception:
        return None

    best = None
    best_score = 0.0
    for table in finder.tables:
        rect = pymupdf.Rect(table.bbox) * page.rotation_matrix
        if _inside_ratio(rect, block.rect) < MIN_INSIDE_RATIO:
            continue
        try:
            raw = table.extract()
        except Exception:
            continue
        if len(raw) < MIN_ROWS or max((len(r) for r in raw), default=0) < MIN_COLS:
            continue
        header_hits = len(
            lexicon.header_hits(
                token
                for row in raw[:HEADER_SCAN_ROWS]
                for cell in row
                if cell
                for token in re.split(r"[\s/]+", cell)
            )
        )
        if header_hits == 0:
            continue
        score = header_hits + len(raw) * 0.1
        if score > best_score:
            best, best_score = table, score

    if best is None:
        # No drawn grid. The table may still be there, set as aligned text.
        return parse_unruled(page, block)

    raw_rows = best.extract()
    columns, consumed = _merge_header_rows(raw_rows)
    if not columns:
        return parse_unruled(page, block)

    grid = pymupdf.Rect(best.bbox) * page.rotation_matrix
    row_rects = [pymupdf.Rect(r.bbox) * page.rotation_matrix for r in best.rows]
    tail_values = _recover_tail(page, grid, block, row_rects)

    # The tail's own name sits on the last header row, not on row 0 -- these
    # tables carry a blank row and their block title above the header.
    tail_header = ""
    if tail_values and 0 < consumed <= len(tail_values):
        tail_header = tail_values[consumed - 1]
    recovered = [tail_header] if tail_header else []

    rows: list[Row] = []
    for index in range(consumed, len(raw_rows)):
        cells = [(c or "").replace("\n", " ").strip() for c in raw_rows[index]]
        tail = tail_values[index] if index < len(tail_values) else ""
        if not _looks_like_data(cells + ([tail] if tail else [])):
            continue

        record: dict[str, str] = {}
        for position, name in enumerate(columns):
            value = cells[position] if position < len(cells) else ""
            if value:
                record[name] = value
        if tail and recovered:
            record[recovered[0]] = tail

        verbatim = " ".join(v for v in cells + [tail] if v)
        rect = row_rects[index] if index < len(row_rects) else block.rect
        rows.append(Row(cells=record, verbatim=verbatim, rect=rect, row_index=index))

    if not rows:
        return None

    return ParsedTable(
        columns=columns + recovered,
        rows=rows,
        rect=grid,
        parser="find_tables+tail_recovery" if recovered else "find_tables",
        recovered_columns=recovered,
    )


# --- fallback: unruled tables ----------------------------------------------
#
# find_tables(strategy="lines_strict") needs a drawn grid. Plenty of schedules
# in this corpus are set as aligned text with no rulings at all, or with
# rulings drawn as filled rects it does not treat as lines -- 83 blocks, some
# of them 900 words with 26 header hits, parsed to nothing. For those the
# structure still exists, just in the whitespace: rows are y-bands and columns
# are the x-gutters that stay empty across every row.

# Rows split where the vertical gap exceeds this share of the line height.
ROW_GAP_FACTOR = 0.6
# An x-range must be clear of words for this many points to count as a gutter.
MIN_GUTTER_PT = 4.0
MIN_FALLBACK_ROWS = 2


def _row_bands(words: list[geom.Word]) -> list[list[geom.Word]]:
    """Group words into rows by vertical position."""
    if not words:
        return []
    heights = sorted(w.rect.height for w in words if w.rect.height > 0)
    line_height = heights[len(heights) // 2] if heights else 10.0
    gap = max(1.5, line_height * ROW_GAP_FACTOR)

    rows: list[list[geom.Word]] = []
    for word in sorted(words, key=lambda w: (w.y0, w.x0)):
        if rows and word.y0 - max(w.y0 for w in rows[-1]) <= gap:
            rows[-1].append(word)
        else:
            rows.append([word])
    return [sorted(r, key=lambda w: w.x0) for r in rows]


def _gutters(rows: list[list[geom.Word]], left: float, right: float) -> list[float]:
    """Column boundaries, from x-ranges no row ever writes into."""
    width = int(max(1.0, right - left)) + 1
    covered = bytearray(width)
    for row in rows:
        for word in row:
            start = max(0, int(word.x0 - left))
            end = min(width, int(word.x1 - left) + 1)
            for i in range(start, end):
                covered[i] = 1

    edges: list[float] = []
    run_start = None
    for i in range(width):
        if not covered[i]:
            if run_start is None:
                run_start = i
        elif run_start is not None:
            if i - run_start >= MIN_GUTTER_PT:
                edges.append(left + (run_start + i) / 2)
            run_start = None
    return edges


# Words in a header row closer than this belong to the same column name
# ("HEAD HT", "FRAME MAT'L").
HEADER_WORD_GAP_PT = 7.0


def _header_bounds(header: list[geom.Word], block: Block) -> list[float]:
    """Column boundaries taken from the spacing of the header row itself."""
    if not header:
        return []
    cells: list[list[geom.Word]] = [[header[0]]]
    for word in header[1:]:
        if word.x0 - cells[-1][-1].x1 > HEADER_WORD_GAP_PT:
            cells.append([word])
        else:
            cells[-1].append(word)
    if len(cells) < 2:
        return []

    bounds = [min(block.rect.x0, cells[0][0].x0 - 2)]
    for left, right in zip(cells, cells[1:]):
        bounds.append((left[-1].x1 + right[0].x0) / 2)
    bounds.append(max(block.rect.x1, cells[-1][-1].x1 + 2))
    return bounds


def parse_unruled(page: pymupdf.Page, block: Block) -> ParsedTable | None:
    """Parse a table that has no usable rulings, using text geometry alone."""
    words = geom.words(page, clip=block.rect)
    rows = _row_bands(words)
    if len(rows) < MIN_FALLBACK_ROWS + 1:
        return None

    # The header is the highest row that reads like one.
    header_index = None
    for index, row in enumerate(rows):
        texts = [w.text for w in row]
        hits = lexicon.header_hits(texts)
        meaningful = [t for t in texts if any(c.isalnum() for c in t)]
        if not meaningful:
            continue
        if len(hits) >= 3 and len(hits) / len(meaningful) >= 0.4:
            header_index = index
            break
    if header_index is None:
        return None

    body = rows[header_index + 1:]
    if len(body) < MIN_FALLBACK_ROWS:
        return None

    # Prefer the header row's own layout for column boundaries. Gutters that
    # stay clear across *every* body row are rare once a NOTES column carries
    # long prose, and project_570's window schedule collapses to a single
    # column that way; its header, by contrast, is cleanly spaced.
    bounds = _header_bounds(rows[header_index], block)
    if len(bounds) < 3:
        edges = _gutters(body, block.rect.x0, block.rect.x1)
        bounds = [block.rect.x0] + edges + [block.rect.x1]
    if len(bounds) < 3:
        return None

    def bucket(word: geom.Word) -> int:
        centre = (word.x0 + word.x1) / 2
        for i in range(len(bounds) - 1):
            if bounds[i] <= centre < bounds[i + 1]:
                return i
        return len(bounds) - 2

    def cells_of(row: list[geom.Word]) -> list[str]:
        out = [""] * (len(bounds) - 1)
        for word in row:
            i = bucket(word)
            out[i] = f"{out[i]} {word.text}".strip()
        return out

    header_cells = cells_of(rows[header_index])
    columns = [c or f"col{i}" for i, c in enumerate(header_cells)]

    parsed_rows: list[Row] = []
    for offset, row in enumerate(body):
        cells = cells_of(row)
        if not _looks_like_data(cells):
            continue
        record = {columns[i]: v for i, v in enumerate(cells) if v}
        rect = geom.union([w.rect for w in row]) or block.rect
        parsed_rows.append(
            Row(
                cells=record,
                verbatim=" ".join(w.text for w in row),
                rect=rect,
                row_index=header_index + 1 + offset,
            )
        )

    if len(parsed_rows) < MIN_FALLBACK_ROWS:
        return None

    return ParsedTable(
        columns=columns,
        rows=parsed_rows,
        rect=geom.union([w.rect for w in words]) or block.rect,
        parser="text_bands",
    )
