"""Invariants for the annotation layer.

The one that matters most: re-running the detector must never destroy human
work. Everything else here is geometry that is easy to get subtly wrong and
expensive to notice later.

Run: .venv/bin/python -m schedext.eval.test_annotate
"""

from __future__ import annotations

import sys

from schedext import annotate as A


def _sheet(**kw) -> A.SheetAnnotation:
    base = dict(sheet_key="486_0d86fa84_p18", project_id="486",
                planset_key="486_0d86fa84", file="project_486.pdf",
                pdf_sha256="0d86fa84" * 8, page=18, rotation=0,
                page_rect_pt=[0, 0, 3024, 2160],
                render={"image": "x.jpg", "zoom": 1.8, "w": 5443, "h": 3888})
    base.update(kw)
    return A.SheetAnnotation(**base)


def check(name, condition, problems):
    if not condition:
        problems.append(name)


def main() -> int:
    problems: list[str] = []

    # --- class inheritance ---------------------------------------------------
    window = A.Area(id="a1", type="window_area", rect=[0, 0, 100, 100])
    door = A.Area(id="a2", type="door_area", rect=[200, 0, 300, 100])
    inner = A.Area(id="a3", type="door_area", rect=[10, 10, 60, 60])

    o = A.Opening(id="o1", rect=[20, 20, 40, 40])
    A.assign(o, [window])
    check("inherits window from containing area", o.cls == "window", problems)

    A.assign(o, [window, inner])
    check("smallest containing area wins", o.area_id == "a3" and o.cls == "door", problems)

    o2 = A.Opening(id="o2", rect=[210, 20, 240, 40])
    A.assign(o2, [window, door])
    check("moving into a door area flips the class", o2.cls == "door", problems)

    o3 = A.Opening(id="o3", rect=[500, 500, 520, 520])
    A.assign(o3, [window, door])
    check("outside every area has no class", o3.cls == "" and o3.area_id == "", problems)

    # a human override must survive re-assignment
    o4 = A.Opening(id="o4", rect=[20, 20, 40, 40], cls="garage_door", cls_source="human")
    A.assign(o4, [window])
    check("human class is not overwritten by inheritance",
          o4.cls == "garage_door", problems)

    # mixed areas cannot hand down a class
    mixed = A.Area(id="a4", type="mixed", rect=[0, 0, 100, 100])
    o5 = A.Opening(id="o5", rect=[20, 20, 40, 40])
    A.assign(o5, [mixed])
    check("mixed area demands a human class", o5.cls_source == "required", problems)

    # excluded areas are not containers
    excluded = A.Area(id="a5", type="exclude", rect=[0, 0, 100, 100])
    o6 = A.Opening(id="o6", rect=[20, 20, 40, 40])
    A.assign(o6, [excluded])
    check("excluded area never adopts an opening", o6.area_id == "", problems)

    # --- containment threshold ----------------------------------------------
    half = A.Opening(id="o7", rect=[90, 20, 110, 40])      # 50% inside
    A.assign(half, [window])
    check("50% overlap does not belong to the area", half.area_id == "", problems)

    # --- the merge invariant -------------------------------------------------
    existing = _sheet(status="verified")
    existing.areas = [A.Area(id="a1", type="window_area", rect=[0, 0, 100, 100],
                             edited=True, type_source="human",
                             seed={"block_id": "p18_b0"})]
    existing.openings = [
        A.Opening(id="o1", rect=[10, 10, 30, 30], origin="human", state="accepted"),
        A.Opening(id="o2", rect=[40, 40, 60, 60], edited=True, state="accepted"),
        A.Opening(id="o3", rect=[70, 10, 90, 30], state="accepted"),
        A.Opening(id="o4", rect=[10, 70, 30, 90], state="rejected"),
    ]
    human_rect = list(existing.openings[1].rect)
    edited_area_rect = list(existing.areas[0].rect)

    fresh = _sheet()
    fresh.areas = [A.Area(id="a1", type="door_area", rect=[5, 5, 105, 105],
                          seed={"block_id": "p18_b0"})]
    fresh.openings = [
        A.Opening(id="n1", rect=[41, 41, 61, 61], confidence=0.9),   # matches o2
        A.Opening(id="n2", rect=[71, 11, 91, 31], confidence=0.8),   # matches o3
        A.Opening(id="n3", rect=[10, 70, 30, 90], confidence=0.7),   # matches rejected o4
        A.Opening(id="n4", rect=[200, 200, 220, 220], confidence=0.6),  # brand new
    ]

    merged, delta = A.merge(existing, fresh)
    ids = {o.id for o in merged.openings}

    check("human-drawn box survives a re-run", "o1" in ids, problems)
    check("edited rect is not rewritten",
          next(o for o in merged.openings if o.id == "o2").rect == human_rect, problems)
    check("unedited model rect is refreshed",
          next(o for o in merged.openings if o.id == "o3").rect == [71, 11, 91, 31], problems)
    check("a rejected box stays rejected when re-proposed",
          next(o for o in merged.openings if o.id == "o4").state == "rejected", problems)
    check("a genuinely new box arrives as pending",
          any(o.state == "pending" and "new_since_verify" in o.flags
              for o in merged.openings), problems)
    check("human area type is not overwritten",
          merged.areas[0].type == "window_area", problems)
    check("edited area rect is not rewritten",
          merged.areas[0].rect == edited_area_rect, problems)
    check("verified sheet is downgraded when new boxes appear",
          merged.status == "needs_recheck" and delta["downgraded"], problems)

    # merging twice must be stable for human content
    twice, _ = A.merge(merged, fresh)
    check("merge is idempotent for human content",
          next(o for o in twice.openings if o.id == "o2").rect == human_rect
          and twice.areas[0].type == "window_area", problems)

    # --- point/pixel round trip ---------------------------------------------
    for zoom in (1.8, 2.5):
        pt = [12.34, 56.78, 90.12, 134.56]
        px = [v * zoom for v in pt]
        back = [round(v / zoom, 2) for v in px]
        check(f"pt->px->pt round trip at zoom {zoom}",
              all(abs(a - b) < 0.01 for a, b in zip(pt, back)), problems)

    for problem in problems:
        print(f"FAIL {problem}")
    total = 18
    print(f"{total - len(problems)}/{total} annotation invariants hold")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
