"""End-to-end checks for locate -> segment -> genre on hand-verified sheets.

Every expectation here was confirmed by eye against the rendered sheet. The
rotated cases are the important ones: project_559 is a 270-degree page, and an
earlier version of the pipeline silently returned zero blocks for it.

Run: .venv/bin/python -m schedext.eval.test_blocks
"""

from __future__ import annotations

import sys
from pathlib import Path

import pymupdf

from schedext import genre, locate, pdfio, segment

PDF_DIR = Path("all_plansets")

# file -> page (1-based) -> {block title: expected genre}
EXPECTED: dict[str, dict[int, dict[str, str]]] = {
    "project_486_20260526_183817_0d86fa84.pdf": {
        18: {
            "WINDOW SCHEDULE": "tabular",
            "DOOR SCHEDULE - DWELLING UNITS": "tabular",
            "DOOR SCHEDULE - COMMON AREA": "tabular",
            "DOOR TYPES": "pictorial",
        },
    },
    # Rotated 270. Pictorial: drawn elevations with bullet spec lists, no table.
    "project_559_20260708_104642_88bfdb0a.pdf": {
        1: {
            "DOOR SCHEDULE:": "pictorial",
            "WINDOW SCHEDULE:": "pictorial",
        },
    },
    "project_558_20260708_103302_34eaf218.pdf": {
        1: {
            "DOOR SCHEDULE:": "pictorial",
            "WINDOW SCHEDULE:": "pictorial",
        },
    },
    "project_524_20260521_124928_ffc4b080.pdf": {
        28: {
            "RESIDENT ROOM DOOR SCHEDULE": "tabular",
            "BUILDING DOOR SCHEDULE": "tabular",
        },
    },
}


def check(pdf_name: str, expectations: dict[int, dict[str, str]]) -> list[str]:
    path = PDF_DIR / pdf_name
    manifest = pdfio.inspect(path).to_dict()
    candidates = {c.page_index + 1: c for c in locate.locate(path, manifest)}
    problems: list[str] = []

    with pymupdf.open(path) as doc:
        for page_no, wanted in expectations.items():
            candidate = candidates.get(page_no)
            if candidate is None:
                problems.append(f"{pdf_name} p{page_no}: not located at all")
                continue
            page = doc[page_no - 1]
            blocks = genre.annotate(page, segment.segment(page, candidate))
            found = {b.title: b for b in blocks}
            for title, want_genre in wanted.items():
                block = found.get(title)
                if block is None:
                    problems.append(
                        f"{pdf_name} p{page_no}: missing block {title!r} "
                        f"(found {sorted(found)})"
                    )
                    continue
                if block.genre != want_genre:
                    problems.append(
                        f"{pdf_name} p{page_no} {title!r}: genre "
                        f"{block.genre!r} != {want_genre!r}"
                    )
    return problems


def main() -> int:
    problems: list[str] = []
    checked = 0
    for pdf_name, expectations in EXPECTED.items():
        checked += sum(len(v) for v in expectations.values())
        problems.extend(check(pdf_name, expectations))

    for problem in problems:
        print(f"FAIL {problem}")
    print(f"{checked - len(problems)}/{checked} block expectations passed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
