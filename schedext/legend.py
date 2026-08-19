"""Read a pictogram type legend and use it to decode type codes (stage S5c).

A tabular schedule's TYPE column holds a code -- F1, A0, C2 -- that means
nothing on its own. The sheet explains it in a legend block beside the table:

    [drawing]      [drawing]      [drawing]
    TYPE A0        TYPE C1        TYPE F3
    1 PANEL        FULL GLASS     OVERHEAD PANEL

Without this, project_486's door rows resolve to a category and nothing else.
With it they gain a leaf face, which is the field a takeoff actually wants.

The codes are project-local, so the mapping is rebuilt per sheet and never
shared between plansets.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pymupdf

from . import geom
from .segment import Block

# "TYPE A0", "TYPE F3", and bare codes used as captions.
TYPE_LABEL = re.compile(r"^TYPE\s+(?P<code>[A-Z]{0,3}-?\d{0,3}[A-Z]?)\s*$", re.I)
BARE_CODE = re.compile(r"^(?P<code>[A-Z]{1,3}-?\d{1,3}[A-Z]?)\s*$")

# Captions sit directly under their code.
CAPTION_GAP_FACTOR = 2.2

# Boilerplate that is not a caption. Note this must not reject a leading digit
# outright -- "1 PANEL" and "2 PANEL" are exactly the captions we are after.
NOT_A_CAPTION = re.compile(
    r"^AS\s+SCHEDULED$|^N\.?T\.?S\.?$|SCALE|"
    r"^\d+\s*[\"']|^\d+\s*(?:TYP|MIN|MAX)\b|^\d+'\s*-",
    re.I,
)


@dataclass
class LegendEntry:
    code: str
    caption: str
    rect: pymupdf.Rect

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "caption": self.caption,
            "rect": [round(v, 1) for v in self.rect],
        }


def parse(page: pymupdf.Page, block: Block) -> list[LegendEntry]:
    """Extract ``code -> caption`` pairs from a type-legend block."""
    lines = [l for l in geom.lines(page, clip=block.rect) if l.text.strip()]
    if not lines:
        return []

    heights = [l.rect.height for l in lines if l.rect.height > 0]
    spacing = sorted(heights)[len(heights) // 2] if heights else 10.0

    entries: list[LegendEntry] = []
    for index, line in enumerate(lines):
        text = line.text.strip()
        match = TYPE_LABEL.match(text)
        if not match:
            continue
        code = match.group("code").upper()
        if not code:
            continue

        # The caption is the nearest line below, in the same column. Column
        # alignment has to lead: captions sit only a few points lower and
        # overlap the label's own box, so filtering on "starts below the
        # label's bottom edge" discards every one of them.
        caption = ""
        rect = pymupdf.Rect(line.rect)
        best: geom.Line | None = None
        for other in lines:
            if other is line:
                continue
            drop = other.rect.y0 - line.rect.y0
            if drop <= 1 or drop > spacing * CAPTION_GAP_FACTOR:
                continue
            centre = (other.rect.x0 + other.rect.x1) / 2
            if not (line.rect.x0 - spacing <= centre <= line.rect.x1 + spacing):
                continue
            if best is None or other.rect.y0 < best.rect.y0:
                best = other

        if best is not None and not NOT_A_CAPTION.match(best.text.strip()):
            caption = best.text.strip()
            rect |= best.rect

        entries.append(LegendEntry(code=code, caption=caption, rect=rect))

    return [e for e in entries if e.caption]


def build_map(page: pymupdf.Page, blocks: list[Block]) -> dict[str, dict[str, str]]:
    """Legend lookups for one sheet, keyed by category then code.

    Keeping the category separate matters: a window legend and a door legend on
    the same sheet can both define a code "A1".
    """
    out: dict[str, dict[str, str]] = {}
    for block in blocks:
        if block.kind != "type_legend":
            continue
        for entry in parse(page, block):
            out.setdefault(block.category, {})[entry.code] = entry.caption
    return out


def caption_for(legends: dict[str, dict[str, str]], category: str, code: str) -> str:
    """Look a code up, preferring the matching category then falling back."""
    if not code:
        return ""
    key = code.strip().upper()
    for name in (category, "both", "door", "window", "garage_door"):
        table = legends.get(name)
        if table and key in table:
            return table[key]
    return ""
