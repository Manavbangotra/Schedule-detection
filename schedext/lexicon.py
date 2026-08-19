"""Shared vocabulary for recognising schedules.

Kept in one module because the locator, the genre probe and the column mapper
all key off the same words, and they must not drift apart.
"""

from __future__ import annotations

import re

# --- Sheet / block titles ---------------------------------------------------

# Deliberately broader than "WINDOW SCHEDULE". A first pass that required
# SCHEDULE adjacent to WINDOW/DOOR found only 36 of 59 plansets; the misses were
# vocabulary variants, not absent schedules -- "WINDOWS CHART" (project_656),
# "WINDOW & DOOR LEGENDS & DETAILS" (project_744), a bare "SCHEDULES"
# (project_658).
# GLAZING stays; bare GLASS does not -- it matched "RIGID GLASS FIBER PIPE
# INSULATION SCHEDULE" and pulled a mechanical sheet into the corpus.
SUBJECT = r"(?:WINDOW|WINDOWS|DOOR|DOORS|GLAZING|STOREFRONT|OPENING|OPENINGS|SASH)"
# A tabular schedule is titled with one of these anywhere in the string.
KIND_TABLE = r"(?:SCHEDULE|SCHEDULES|CHART)"
# A pictogram legend is titled "DOOR TYPES" / "WINDOW ELEVATIONS" -- the kind
# word comes *last*. Anchoring it to the end is what keeps "TYPE A UNIT DOOR
# CLEARANCES" (project_486 p7) from reading as a door-type legend.
KIND_LEGEND = r"(?:TYPES|TYPE|ELEVATIONS|LEGENDS|LEGEND)"

TITLE_STRONG = re.compile(rf"\b{SUBJECT}\b.{{0,30}}?\b{KIND_TABLE}\b", re.I)
TITLE_REVERSE = re.compile(rf"\b{KIND_TABLE}\b.{{0,20}}?\b{SUBJECT}\b", re.I)
# Trailing "& DETAILS" is common on legend sheets ("WINDOW & DOOR LEGENDS &
# DETAILS", project_744 A-004) and does not change what the block is.
TITLE_LEGEND = re.compile(
    rf"\b{SUBJECT}\b.{{0,24}}?\b{KIND_LEGEND}\b(?:[\s,&]+(?:AND\s+)?DETAILS?)?[\s:.]*$",
    re.I,
)
# "SCHEDULES" on its own -- weak, needs corroboration from another signal.
TITLE_WEAK = re.compile(r"\b(?:SCHEDULES|SCHEDULE|CHART)\b", re.I)

# Sheets that talk about doors and windows without scheduling them. These are
# details, code-compliance diagrams and mounting-height studies; they carry no
# per-mark rows and would only add noise.
NOT_A_SCHEDULE = re.compile(
    r"CLEARANCE|MOUNTING|SIGNAGE|ACCESSIBILIT|MANEUVER|APPROACH|"
    r"\bSECTION\b|\bJAMB\b|\bHEAD\b|\bSILL\b|\bFLASHING\b|\bTHRESHOLD\b|"
    r"WALL\s+TYPE|\bPLAN\b|\bNOTES?\b|"
    # Schedules belonging to other disciplines. Only ever consulted when the
    # title has no door/window subject, so a door "PANEL" legend is unaffected.
    r"\bSHEAR\b|\bFOOTING\b|\bBEAM\b|\bCOLUMN\b|\bLINTEL\b|\bTRUSS\b|\bSLAB\b|"
    r"\bFINISH\b|\bROOM\b|\bLIGHTING\b|\bFIXTURE\b|\bEQUIPMENT\b|\bPLUMBING\b|"
    r"\bMATERIAL\b|\bINSULATION\b|\bPIPE\b|\bDUCT\b|\bPAINT\b|\bFLOOR(?:ING)?\b|"
    r"PANEL\s+SCHEDULE|\bMECHANICAL\b|\bELECTRICAL\b",
    re.I,
)

# Pages that merely *list* schedules rather than containing one.
INDEX_PAGE = re.compile(
    r"SHEET\s+INDEX|DRAWING\s+INDEX|SHEET\s+LIST|INDEX\s+OF\s+DRAWINGS|"
    r"TABLE\s+OF\s+CONTENTS|DRAWING\s+LIST",
    re.I,
)

# Prose references such as "(REF. TO DOOR SCHEDULE)" that appear in detail
# callouts all over a plan set.
PROSE_REFERENCE = re.compile(r"\b(?:REF|REFER|SEE|PER)\b\.?\s*(?:TO\s*)?$", re.I)


def title_score(text: str) -> float:
    """How strongly a title string suggests an actual schedule block."""
    clean = " ".join(text.split())
    if not clean or len(clean.split()) > 9:
        return 0.0
    if INDEX_PAGE.search(clean):
        return 0.0

    # An explicit "<subject> SCHEDULE" always counts, even when the title also
    # mentions notes or details -- project_745's sheet is genuinely titled
    # "DOOR SCHEDULE, NOTES, & DETAILS". Whether the block is worth parsing is
    # decided later by kind_of(), not here.
    if TITLE_STRONG.search(clean) or TITLE_REVERSE.search(clean):
        return 1.0

    # The weaker patterns match far more loosely, so they need the guard.
    if NOT_A_SCHEDULE.search(clean):
        return 0.0
    if TITLE_LEGEND.search(clean):
        return 1.0
    if TITLE_WEAK.search(clean):
        return 0.35
    return 0.0


# --- Column headers ---------------------------------------------------------

# Grounded in headers actually observed across the corpus. Used three ways:
# to score a candidate header row, to tell a table from a pictorial block, and
# as the match target for column mapping.
HEADER_TOKENS = {
    "MARK", "MK", "NO", "NO.", "NUMBER", "TAG", "ID", "SYMBOL",
    "TYPE", "SIZE", "WIDTH", "W", "W.", "HEIGHT", "H", "H.", "HT", "HT.",
    "THICK", "THICK.", "THICKNESS", "T.", "QTY", "QUANTITY", "COUNT",
    "LEAF", "LEAVES", "PAIR", "SWING", "HAND", "OPERATION", "OPER",
    "MATERIAL", "MAT", "MAT.", "MATL", "FRAME", "GLAZING", "GLASS", "GLZ",
    "LOUVER", "FINISH", "COLOR",
    "HEAD", "JAMB", "SILL", "DETAIL", "DETAILS", "THRESHOLD",
    "FIRE", "RATING", "RTG", "RTG.", "LABEL",
    "HDW", "HARDWARE", "SET", "LOCK", "CLOSER",
    "LOCATION", "ROOM", "FROM", "TO", "LEVEL", "FLOOR",
    "REMARKS", "COMMENTS", "NOTES", "NOTE",
    "MANUFACTURER", "MFR", "MODEL", "SERIES",
    "U-FACTOR", "SHGC", "EGRESS", "ELEVATION", "ROUGH", "OPENING", "R.O.",
}

# Header vocabulary that also appears in MEP schedules; seeing these alone must
# not be enough to call a page a door/window schedule. project_596 has lighting
# schedules headed TYPE / LUMENS / LAMP TYPE / WATTS / VOLTAGE / MOUNTING.
AMBIGUOUS_HEADERS = {"TYPE", "SIZE", "LOCATION", "REMARKS", "NOTES", "MODEL",
                     "MANUFACTURER", "MFR", "FINISH", "COLOR", "LEVEL", "NO", "NO.",
                     # Ordinary English that happens to name a column.
                     "TO", "FROM", "SET", "LABEL", "HEAD", "W", "H", "T."}

CONTEXT_WORDS = re.compile(r"\b(?:DOOR|DOORS|WINDOW|WINDOWS|GLAZING|STOREFRONT|SASH)\b", re.I)


def normalise_token(token: str) -> str:
    return token.upper().strip(" :.,()[]")


def header_hits(tokens) -> set[str]:
    """Distinct header-vocabulary tokens present in ``tokens``."""
    return {
        normalise_token(t) for t in tokens
        if normalise_token(t) in HEADER_TOKENS
    }


def header_strength(tokens) -> tuple[int, int]:
    """``(total_hits, unambiguous_hits)`` for a candidate header row."""
    hits = header_hits(tokens)
    return len(hits), len(hits - AMBIGUOUS_HEADERS)


# --- Category ---------------------------------------------------------------

def category_of(title: str) -> str:
    """Window vs door vs garage, taken from the block title.

    This is the most reliable category signal available and it is why the
    detector's worst failure -- labelling a block titled "DOOR TYPES" as
    windows -- is unreachable here.
    """
    upper = title.upper()
    has_window = bool(re.search(r"\bWINDOW|SASH|GLAZING\b", upper))
    has_door = bool(re.search(r"\bDOOR|OPENING|STOREFRONT\b", upper))
    if re.search(r"\bGARAGE|OVERHEAD|SECTIONAL|ROLL[- ]?UP\b", upper):
        return "garage_door"
    if has_window and has_door:
        return "both"
    if has_window:
        return "window"
    if has_door:
        return "door"
    return "unknown"


def kind_of(title: str) -> str:
    """What sort of block this is, from its title."""
    upper = title.upper()
    if re.search(r"\bHARDWARE\b", upper) and not re.search(r"\bSCHEDULE\b", upper):
        return "hardware"
    if re.search(r"\bNOTES?\b|ABBREVIATIONS?\b|\bKEY\b", upper):
        return "notes"
    if re.search(r"\bTYPES?\b|\bELEVATIONS?\b|\bLEGENDS?\b", upper):
        return "type_legend"
    if re.search(r"\bSCHEDULES?\b|\bCHART\b", upper):
        return "schedule"
    return "unknown"
