"""Turn schedule text into canonical, faceted types (stage S5).

The vocabulary on these sheets is faceted, not flat: "VINYL SINGLE HUNG" is a
material and an operation, and "RAISED MULTI-PANEL DOOR / HOLLOW CORE" is a
leaf face and a core. A single flat label cannot hold that, which is the second
reason -- independent of accuracy -- that the existing 17-class detector output
is the wrong shape for this job.

Nothing is ever guessed. Text that no rule matches is left unresolved and
flagged, so new vocabulary surfaces instead of being silently mislabelled.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict

# --- abbreviations ----------------------------------------------------------

# Expanded before matching. Scoped by where the text came from: a lone "F" means
# FLUSH in a type-code column and nothing at all in prose, so single letters are
# only expanded via CODE_ABBREVIATIONS.
ABBREVIATIONS = {
    "HM": "HOLLOW METAL", "SCWD": "SOLID CORE WOOD", "HCWD": "HOLLOW CORE WOOD",
    "ALWD": "ALUMINUM CLAD WOOD", "MTL": "METAL", "WD": "WOOD",
    "ALUM": "ALUMINUM", "MANUF": "MANUFACTURED", "GL": "GLASS",
    "TEMP": "TEMPERED", "INSUL": "INSULATED", "IG": "INSULATED GLASS",
    "SH": "SINGLE HUNG", "DH": "DOUBLE HUNG", "FG": "FULL GLASS",
    "OH": "OVERHEAD", "PR": "PAIR", "SGD": "SLIDING GLASS DOOR",
    "NR": "NON RATED", "PT": "PAINTED", "STL": "STEEL", "ALM": "ALUMINUM",
    "FRP": "FIBERGLASS", "CSMT": "CASEMENT", "FX": "FIXED", "SL": "SLIDING",
    "SGL": "SINGLE", "DBL": "DOUBLE", "UL": "UNDERWRITERS LABORATORIES",
    "ALUM.": "ALUMINUM", "INSUL.": "INSULATED", "TEMP.": "TEMPERED",
}

CODE_ABBREVIATIONS = {"F": "FLUSH", "P": "PANEL", "G": "GLASS", "L": "LOUVER"}

# --- canonical facets -------------------------------------------------------

# (pattern, value). First match wins, so order longest/most specific first.
CATEGORY_RULES = [
    (r"\bGARAGE\b|\bOVERHEAD\b|\bSECTIONAL\b|\bROLL[- ]?UP\b|\bCOILING\b", "garage_door"),
    (r"\bSTOREFRONT\b|\bCURTAIN\s*WALL\b", "storefront"),
    (r"\bDECORATIVE\s+SHUTTER\b|\bSHUTTER\b", "shutter"),
    (r"\bLOUVER\b|\bLOUVRE\b", "louver"),
    (r"\bGRILL\b|\bGRILLE\b", "grille"),
    (r"\bSKYLIGHT\b", "skylight"),
    (r"\bGATE\b", "gate"),
    (r"\bDOOR\b", "door"),
    (r"\bWINDOW\b|\bSASH\b", "window"),
]

OPERATION_RULES = [
    (r"\bSINGLE\s*HUNG\b", "single_hung"),
    (r"\bDOUBLE\s*HUNG\b", "double_hung"),
    (r"\bCASEMENT\b", "casement"),
    (r"\bAWNING\b", "awning"),
    (r"\bHOPPER\b", "hopper"),
    (r"\bJALOUSIE\b", "jalousie"),
    (r"\bTILT[- ]?TURN\b", "tilt_turn"),
    (r"\bHORIZONTAL\s*SLID|SLIDING\s*GLASS\s*DOOR|\bSLIDER\b", "sliding"),
    (r"\bOVERHEAD\b|\bSECTIONAL\b", "overhead_sectional"),
    (r"\bROLL[- ]?UP\b|\bCOILING\b", "roll_up"),
    (r"\bBI[- ]?FOLD\b|\bBIFOLD\b", "bifold"),
    (r"\bBY[- ]?PASS\b|\bBYPASS\b", "bypass"),
    (r"\bPOCKET\b", "pocket"),
    (r"\bBARN\b|\bFARM\b", "barn"),
    (r"\bDUTCH\b", "dutch"),
    (r"\bFRENCH\b", "french_pair"),
    (r"\bREVOLVING\b", "revolving"),
    (r"\bSLIDING\b", "sliding"),
    (r"\bFIXED\b|\bPICTURE\b", "fixed"),
    (r"\bTRANSOM\b", "transom"),
    (r"\bPAIR\b|\bDOUBLE\s+DOOR\b", "swing_pair"),
    # A leaf count is the only operation some door schedules state.
    (r"\bSINGLE\b", "swing_single"),
]

LEAF_FACE_RULES = [
    (r"\bRAISED\s+MULTI[- ]?PANEL\b", "raised_multi_panel"),
    (r"\bMULTI[- ]?PANEL\b", "raised_multi_panel"),
    (r"\bFULL\s*GLASS\b|\bALL\s*GLASS\b|\bGLASS\s+DOOR\b", "full_glass"),
    (r"\bHALF\s*(?:GLASS|LITE|LIGHT)\b", "half_lite"),
    (r"\b3/4\s*(?:GLASS|LITE)\b", "three_quarter_lite"),
    (r"\bSIDE\s*LITES?\b", "sidelite"),
    (r"\bLOUVERED\b", "louvered"),
    (r"\bSHAKER\b", "shaker"),
    (r"\bSTOREFRONT\b", "glazed_storefront"),
    (r"\b1\s*-?\s*LITE\b|\bSINGLE\s+LITE\b", "half_lite"),
    (r"\b(?:\d+)\s*-?\s*LITE\b", "multi_lite"),
    # "FLUSH" only names a door face on its own. "INSTALL FLUSH WITH
    # INTERIOR" and "FLUSH TO/AGAINST ..." describe how a unit is set.
    (r"\bFLUSH\b(?!\s+(?:WITH|TO|AGAINST|MOUNT))", "flush"),
    (r"\b(\d)\s*[- ]?PANEL\b", "panel"),
    (r"\bPANEL\b", "panel"),
]

MATERIAL_RULES = [
    (r"\bALUMINUM\s+CLAD\s+WOOD\b|\bALUM\.?\s*WOOD\s*CLAD\b", "aluminum_clad_wood"),
    (r"\bSOLID\s+CORE\s+WOOD\b", "solid_core_wood"),
    (r"\bHOLLOW\s+CORE\b", "hollow_core_wood"),
    (r"\bHOLLOW\s+METAL\b", "hollow_metal"),
    (r"\bFIBERGLASS\b", "fiberglass"),
    (r"\bALUMINUM\b", "aluminum"),
    (r"\bVINYL\b", "vinyl"),
    (r"\bSTEEL\b", "steel"),
    (r"\bMANUFACTURED\b", "manufactured"),
    (r"\bMETAL\b", "metal"),
    (r"\bWOOD\b", "wood"),
]

GLAZING_RULES = [
    (r"\bTEMPERED\b", "tempered"),
    (r"\bINSULATED\b", "insulated"),
    (r"\bLOW[- ]?E\b", "low_e"),
    (r"\bLAMINATED\b", "laminated"),
    (r"\bOBSCURE\b|\bFROSTED\b", "obscure"),
    (r"\bCLEAR\b", "clear"),
]

FLAG_RULES = [
    (r"\bEXTERIOR\b", "exterior"),
    (r"\bINTERIOR\b", "interior"),
    (r"\bEGRESS\b", "egress"),
    (r"\bIMPACT\b", "impact_rated"),
    (r"\bSELF[- ]?CLOSING\b", "self_closing"),
    (r"\bADA\b|\bACCESSIBLE\b", "ada"),
    (r"\bPAIR\b", "pair"),
]

FIRE_RATING = re.compile(r"\b(20|45|60|90|180)\s*(?:MIN|MINUTE)", re.I)

# These have no operation and no leaf face; a missing one is not a gap.
NON_OPERABLE = {"louver", "shutter", "grille", "skylight"}

_COMPILED = {
    "category": [(re.compile(p, re.I), v) for p, v in CATEGORY_RULES],
    "operation": [(re.compile(p, re.I), v) for p, v in OPERATION_RULES],
    "leaf_face": [(re.compile(p, re.I), v) for p, v in LEAF_FACE_RULES],
    "material": [(re.compile(p, re.I), v) for p, v in MATERIAL_RULES],
}
_GLAZING = [(re.compile(p, re.I), v) for p, v in GLAZING_RULES]
_FLAGS = [(re.compile(p, re.I), v) for p, v in FLAG_RULES]


@dataclass
class Canonical:
    category: str | None = None
    operation: str | None = None
    leaf_face: str | None = None
    material: str | None = None
    glazing: list[str] = field(default_factory=list)
    fire_rating_min: int | None = None
    flags: list[str] = field(default_factory=list)
    mapped_by: str = "lexicon"
    needs_review: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


_DOTTED = re.compile(r"\b([A-Z])\.([A-Z])\.(?:([A-Z])\.)?")


def expand(text: str, from_code: bool = False) -> str:
    """Expand abbreviations so the rules only need to know full words."""
    upper = text.upper()
    # Collapse dotted acronyms first: schedules write hollow metal as "H.M."
    # about as often as "HM", and tokenising on non-word characters would
    # otherwise split it into three meaningless pieces.
    upper = _DOTTED.sub(lambda m: "".join(g for g in m.groups() if g), upper)

    table = dict(ABBREVIATIONS)
    if from_code:
        table.update(CODE_ABBREVIATIONS)
    out = []
    for token in re.split(r"(\W+)", upper):
        out.append(table.get(token.strip("."), token))
    return "".join(out)


def _first(kind: str, text: str) -> str | None:
    for pattern, value in _COMPILED[kind]:
        if pattern.search(text):
            return value
    return None


def classify(text: str, block_category: str = "unknown",
             type_code: str = "") -> Canonical:
    """Map free schedule text onto the canonical facets.

    ``block_category`` comes from the block title and is authoritative when the
    row text itself is silent -- a row inside a block titled "DOOR TYPES" is a
    door even if its own words never say so. That single rule is what makes the
    detector's worst failure mode unreachable here.
    """
    body = expand(text)
    if type_code:
        body = f"{body} {expand(type_code, from_code=True)}"

    result = Canonical()
    result.category = _first("category", body)
    if result.category is None and block_category in {
        "window", "door", "garage_door"
    }:
        result.category = block_category
        result.mapped_by = "block_title"

    result.operation = _first("operation", body)
    result.leaf_face = _first("leaf_face", body)
    result.material = _first("material", body)
    result.glazing = sorted({v for pattern, v in _GLAZING if pattern.search(body)})
    result.flags = sorted({v for pattern, v in _FLAGS if pattern.search(body)})

    rating = FIRE_RATING.search(body)
    if rating:
        result.fire_rating_min = int(rating.group(1))

    # Say so rather than guessing: an item with no category, or an operable
    # item whose operation and face are both unknown, is vocabulary we have not
    # seen. Louvers, shutters and grilles have neither by nature, so demanding
    # one of them would flag every correctly-read item of those kinds.
    if result.category is None:
        result.needs_review = True
    elif result.category not in NON_OPERABLE and (
        result.operation is None and result.leaf_face is None
    ):
        result.needs_review = True
    return result


# --- dimensions -------------------------------------------------------------

_FEET_INCHES = re.compile(r"(?P<ft>\d+)\s*'\s*-?\s*(?P<in>\d+)?\s*[\"']?")


def to_inches(text: str) -> float | None:
    """``3' - 6"`` -> 42.0. Returns ``None`` when nothing parses."""
    if not text:
        return None
    if re.search(r"VARIES", text, re.I):
        return None
    match = _FEET_INCHES.search(text)
    if match:
        feet = int(match.group("ft"))
        inches = int(match.group("in") or 0)
        return float(feet * 12 + inches)
    bare = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*\"?\s*", text)
    return float(bare.group(1)) if bare else None
