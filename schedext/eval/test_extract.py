"""Row-level ground truth, read by eye off the rendered sheets.

These are the two cases the detector got wrong, so they are the ones worth
pinning: project_559's windows B and C are printed SINGLE HUNG (the model said
"fixed" and "casement"), and project_486's window schedule carries LOUVER,
DECORATIVE SHUTTER and GRILL in an unruled COMMENTS column that
``find_tables`` drops on its own.

Run: .venv/bin/python -m schedext.eval.test_extract
"""

from __future__ import annotations

import sys
from pathlib import Path

import pymupdf

from schedext import genre, locate, pdfio, pictorial, segment, table

PDF_DIR = Path("all_plansets")

# --- tabular ---------------------------------------------------------------

TABULAR_CASES = [
    {
        "pdf": "project_486_20260526_183817_0d86fa84.pdf",
        "page": 18,
        "block": "WINDOW SCHEDULE",
        "mark_column": "MARK",
        "rows": {
            "A": {"WIDTH": "3' - 0\"", "HEIGHT": "6' - 0\"", "COMMENTS": "VINYL SINGLE HUNG"},
            "B": {"WIDTH": "6' - 0\"", "HEIGHT": "6' - 0\"", "COMMENTS": "VINYL SINGLE HUNG"},
            "C": {"WIDTH": "9' - 0\"", "HEIGHT": "6' - 0\"", "COMMENTS": "VINYL SINGLE HUNG"},
            "D": {"WIDTH": "2' - 0\"", "HEIGHT": "4' - 6\"", "COMMENTS": "VINYL SINGLE HUNG"},
            "E": {"WIDTH": "3' - 0\"", "HEIGHT": "4' - 6\"", "COMMENTS": "VINYL SINGLE HUNG"},
            "F": {"WIDTH": "2' - 0\"", "HEIGHT": "4' - 6\"", "COMMENTS": "DECORATIVE SHUTTER"},
            "L1": {"WIDTH": "1' - 6\"", "HEIGHT": "3' - 0\"", "COMMENTS": "LOUVER"},
            "L2": {"WIDTH": "1' - 4\"", "HEIGHT": "0' - 8\""},
        },
    },
    {
        "pdf": "project_486_20260526_183817_0d86fa84.pdf",
        "page": 18,
        "block": "DOOR SCHEDULE - COMMON AREA",
        "mark_column": "MARK",
        "rows": {
            # Two-tier header: DOOR SIZE spans WIDTH / HEIGHT / THICK.
            "101": {"DOOR SIZE.WIDTH": "3' - 0\"", "DOOR SIZE.HEIGHT": "8' - 0\"",
                    "DOOR DETAILS.TYPE": "F1", "FRAME.MATERIAL": "MTL"},
            "104": {"DOOR SIZE.WIDTH": "16' - 0\"", "DOOR DETAILS.TYPE": "F3",
                    "FRAME.MATERIAL": "MANUF"},
        },
    },
]

# --- pictorial -------------------------------------------------------------

PICTORIAL_CASES = [
    {
        "pdf": "project_559_20260708_104642_88bfdb0a.pdf",
        "page": 1,
        "block": "WINDOW SCHEDULE:",
        "items": {
            "A": {"width_text": "4'-0\"", "height_text": "6'-0\"", "spec": "FIXED"},
            "B": {"width_text": "3'-0\"", "height_text": "6'-0\"", "spec": "SINGLE HUNG"},
            # The sheet mistypes this width as `6'-0'`; it is six feet.
            "C": {"width_text": "6'-0\"", "height_text": "6'-0\"", "spec": "SINGLE HUNG"},
            "D": {"width_text": "3'-0\"", "height_text": "6'-0\"", "spec": "FIXED"},
        },
    },
    {
        "pdf": "project_559_20260708_104642_88bfdb0a.pdf",
        "page": 1,
        "block": "DOOR SCHEDULE:",
        "items": {
            # "3068" is a nominal call: 3'-0" x 6'-8".
            "01": {"width_text": "3'-0\"", "height_text": "6'-8\"", "spec": "RAISED MULTI-PANEL"},
            "02": {"width_text": "3'-0\"", "height_text": "8'-0\"", "spec": "ALUMINUM STOREFRONT"},
            "06": {"width_text": "1'-6\"", "height_text": "6'-8\"", "spec": "RAISED MULTI-PANEL"},
        },
    },
]


def _blocks_for(path: Path, page_no: int):
    manifest = pdfio.inspect(path).to_dict()
    candidates = [c for c in locate.locate(path, manifest) if c.page_index == page_no - 1]
    if not candidates:
        return None, None, None
    doc = pymupdf.open(path)
    page = doc[page_no - 1]
    return doc, page, genre.annotate(page, segment.segment(page, candidates[0]))


def check_tabular(case: dict) -> list[str]:
    problems: list[str] = []
    doc, page, blocks = _blocks_for(PDF_DIR / case["pdf"], case["page"])
    if blocks is None:
        return [f"{case['pdf']} p{case['page']}: page not located"]
    with doc:
        match = next((b for b in blocks if b.title == case["block"]), None)
        if match is None:
            return [f"{case['pdf']}: block {case['block']!r} not found"]
        parsed = table.parse(page, match)
        if parsed is None:
            return [f"{case['block']!r}: no table parsed"]

        by_mark = {
            r.cells.get(case["mark_column"], ""): r.cells for r in parsed.rows
        }
        for mark, wanted in case["rows"].items():
            row = by_mark.get(mark)
            if row is None:
                problems.append(f"{case['block']!r}: mark {mark!r} missing "
                                f"(got {sorted(k for k in by_mark if k)})")
                continue
            for column, value in wanted.items():
                got = row.get(column, "")
                if got != value:
                    problems.append(
                        f"{case['block']!r} {mark}.{column}: {got!r} != {value!r}"
                    )
    return problems


def check_pictorial(case: dict) -> list[str]:
    problems: list[str] = []
    doc, page, blocks = _blocks_for(PDF_DIR / case["pdf"], case["page"])
    if blocks is None:
        return [f"{case['pdf']} p{case['page']}: page not located"]
    with doc:
        match = next((b for b in blocks if b.title == case["block"]), None)
        if match is None:
            return [f"{case['pdf']}: block {case['block']!r} not found"]
        items = {i.mark: i for i in pictorial.parse(page, match)}
        for mark, wanted in case["items"].items():
            item = items.get(mark)
            if item is None:
                problems.append(f"{case['block']!r}: mark {mark!r} missing "
                                f"(got {sorted(items)})")
                continue
            if item.width_text != wanted["width_text"]:
                problems.append(f"{case['block']!r} {mark}.width: "
                                f"{item.width_text!r} != {wanted['width_text']!r}")
            if item.height_text != wanted["height_text"]:
                problems.append(f"{case['block']!r} {mark}.height: "
                                f"{item.height_text!r} != {wanted['height_text']!r}")
            blob = " ".join(item.specs).upper()
            if wanted["spec"] not in blob:
                problems.append(f"{case['block']!r} {mark}: spec "
                                f"{wanted['spec']!r} not in {item.specs}")
    return problems


def main() -> int:
    problems: list[str] = []
    total = 0
    for case in TABULAR_CASES:
        total += len(case["rows"])
        problems.extend(check_tabular(case))
    for case in PICTORIAL_CASES:
        total += len(case["items"])
        problems.extend(check_pictorial(case))

    for problem in problems:
        print(f"FAIL {problem}")
    print(f"{total} ground-truth items checked, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
