"""Title-scoring cases, all taken from titles that actually appear in the corpus.

Run: .venv/bin/python -m schedext.eval.test_lexicon
"""

from __future__ import annotations

import sys

from schedext.lexicon import category_of, kind_of, title_score

# (title, expected score, expected category, expected kind)
CASES = [
    # --- real schedule blocks -------------------------------------------
    ("DOOR SCHEDULE - DWELLING UNITS", 1.0, "door", "schedule"),
    ("DOOR SCHEDULE - COMMON AREA", 1.0, "door", "schedule"),
    ("WINDOW SCHEDULE", 1.0, "window", "schedule"),
    ("WINDOW SCHEDULE:", 1.0, "window", "schedule"),
    ("DOOR-WINDOW SCHEDULE", 1.0, "both", "schedule"),
    ("CLUBHOUSE STOREFRONT SCHEDULE", 1.0, "door", "schedule"),
    ("BUILDING DOOR SCHEDULE", 1.0, "door", "schedule"),
    ("WINDOWS CHART", 1.0, "window", "schedule"),
    # Two-line title, joined before matching (project_707).
    ("Door & Opening Schedule", 1.0, "door", "schedule"),
    # Title also names notes/details but is still a schedule (project_745).
    ("DOOR SCHEDULE, NOTES, & DETAILS", 1.0, "door", "notes"),

    # --- pictogram legends ----------------------------------------------
    ("DOOR TYPES", 1.0, "door", "type_legend"),
    ("WINDOW TYPES", 1.0, "window", "type_legend"),
    ("STOREFRONT DOOR/ WINDOW TYPES", 1.0, "both", "type_legend"),
    ("WINDOW & DOOR LEGENDS & DETAILS", 1.0, "both", "type_legend"),

    # --- weak, needs a second signal ------------------------------------
    ("SCHEDULES", 0.35, "unknown", "schedule"),

    # --- must NOT match --------------------------------------------------
    # "TYPE" mid-title is a unit type, not a door-type legend (project_486 p7).
    ("TYPE A UNIT DOOR CLEARANCES", 0.0, "door", "type_legend"),
    ("DOOR CLEARANCES FOR ACCESSIBLE & TYPE A", 0.0, "door", "type_legend"),
    ("TYP. MOUNTING HTS. @ DOORS", 0.0, "door", "unknown"),
    ("TYP. WINDOW HEAD DETAIL AT STAIRS", 0.0, "window", "unknown"),
    ("AIR SEALING - WINDOW JAMB DETAIL", 0.0, "window", "unknown"),
    ("SHEET INDEX", 0.0, "unknown", "unknown"),
    ("FLOOR PLAN LEGEND", 0.0, "unknown", "type_legend"),
    ("LIFE SAFETY LEGEND", 0.0, "unknown", "type_legend"),
    # A schedule, but not of doors or windows.
    ("WALL FOOTING SCHEDULE", 0.0, "unknown", "schedule"),
    ("SHEAR WALL SCHEDULE", 0.0, "unknown", "schedule"),
    # Interior/MEP schedules that reached the corpus as category "unknown"
    # before bare GLASS was dropped from the subject vocabulary.
    ("MATERIAL SCHEDULE", 0.0, "unknown", "schedule"),
    ("UNIT MATERIAL SCHEDULE - FINISHES", 0.0, "unknown", "schedule"),
    ("RIGID GLASS FIBER PIPE INSULATION SCHEDULE", 0.0, "unknown", "schedule"),
    # ...while a genuinely door-scoped material schedule still counts.
    ("DOOR MATERIAL SCHEDULE", 1.0, "door", "schedule"),
    ("GLAZING SCHEDULE", 1.0, "window", "schedule"),

    # --- from the corpus-wide title scan (3,746 pages, 44 plansets) ---------
    # Every case below was found by scanning every large-text line in the
    # corpus and listing what the vocabulary rejected. Counts are hits on
    # pages that already hold a located schedule sheet.

    # Dropped over punctuation. x1, and not a judgement anyone meant to make.
    ("DOOR TYPES,", 1.0, "door", "type_legend"),
    # ELEVATION singular was missing from KIND_LEGEND; only the plural was
    # listed. x14 across the SF-/ISF- storefront series on located sheets.
    ("SF-1 - STOREFRONT ELEVATION", 1.0, "door", "type_legend"),
    ("SF-4 - STOREFRONT ELEVATION", 1.0, "door", "type_legend"),
    # ...while a subject word is still required, so the 158 KITCHEN and 50
    # BATHROOM elevations stay out.
    ("BATHROOM ELEVATION", 0.0, "unknown", "type_legend"),
    ("KITCHEN ELEVATION", 0.0, "unknown", "type_legend"),
    ("EXTERIOR ELEVATIONS", 0.0, "unknown", "type_legend"),
    ("COURTYARD ELEVATION", 0.0, "unknown", "type_legend"),

    # Implied subject: weak tier, promoted by locate only when the sheet also
    # carries a strong door/window title. x6 in the corpus.
    ("FRAME TYPES", 0.35, "unknown", "type_legend"),
    ("FRAME TYPE", 0.35, "unknown", "type_legend"),
    ("PANEL TYPES", 0.35, "unknown", "type_legend"),
    ("LOUVER TYPES", 0.35, "unknown", "type_legend"),
    ("MULLION TYPES", 0.35, "unknown", "type_legend"),
    # ...but only bare. A stated non-opening subject still wins.
    ("WALL PANEL TYPES", 0.0, "unknown", "type_legend"),
    ("WALL TYPES", 0.0, "unknown", "type_legend"),
    # ...and the electrical homonym must stay dead.
    ("PANEL SCHEDULE", 0.0, "unknown", "schedule"),

    # Structural/MEP schedules that were leaking through at the weak 0.35 tier.
    ("WOOD JOIST SCHEDULE", 0.0, "unknown", "schedule"),
    ("WOOD HEADER SCHEDULE", 0.0, "unknown", "schedule"),
    ("WALL HEADER SCHEDULE", 0.0, "unknown", "schedule"),
    ("CONCRETE PIER SCHEDULE", 0.0, "unknown", "schedule"),
    ("FAN SCHEDULE", 0.0, "unknown", "schedule"),
    ("FIRE EXTINGUISHER SCHEDULE", 0.0, "unknown", "schedule"),
    # HEADER is scoped to "HEADER SCHEDULE", so a door-frame legend survives.
    ("DOOR HEADER TYPES", 1.0, "door", "type_legend"),
    ("WINDOW HEADER SCHEDULE", 1.0, "window", "schedule"),

    # Code-compliance elevations. Adding singular ELEVATION to KIND_LEGEND made
    # these read as perfect matches -- OPENING is a subject word -- and pulled
    # 10 false sheets out of projects 5416 and 5418. They are IBC 705.8
    # exterior-wall studies counting glazed area against a property line.
    ("UNPROTECTED OPENINGS - COURTYARD WEST ELEVATION", 0.0, "door", "type_legend"),
    ("UNPROTECTED OPENINGS - SOUTH ELEVATION", 0.0, "door", "type_legend"),
    ("ALLOWABLE AREA OF OPENINGS", 0.0, "door", "unknown"),
    # ...while the genuine finds from the same change must survive.
    ("OPENING ELEVATION", 1.0, "door", "type_legend"),
    ("STOREFRONT ELEVATION", 1.0, "door", "type_legend"),
]


def main() -> int:
    failures = 0
    for title, want_score, want_category, want_kind in CASES:
        got_score = title_score(title)
        got_category = category_of(title)
        got_kind = kind_of(title)
        problems = []
        if abs(got_score - want_score) > 1e-6:
            problems.append(f"score {got_score} != {want_score}")
        if got_category != want_category:
            problems.append(f"category {got_category!r} != {want_category!r}")
        if got_kind != want_kind:
            problems.append(f"kind {got_kind!r} != {want_kind!r}")
        if problems:
            failures += 1
            print(f"FAIL {title!r}: {'; '.join(problems)}")

    print(f"{len(CASES) - failures}/{len(CASES)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
