"""Command line entry point.

    .venv/bin/python -m schedext.cli all          # manifest -> locate -> extract
    .venv/bin/python -m schedext.cli manifest
    .venv/bin/python -m schedext.cli locate
    .venv/bin/python -m schedext.cli extract
    .venv/bin/python -m schedext.cli one <pdf> [page]
    .venv/bin/python -m schedext.cli viewer        # render boxes, serve on :8000
    .venv/bin/python -m schedext.cli detect --limit 10   # find openings with best.pt
    .venv/bin/python -m schedext.cli annotate            # seed + serve the editor
    .venv/bin/python -m schedext.cli export-dataset --tag v1
    .venv/bin/python -m schedext.cli train --arms all    # stage-2 detector
    .venv/bin/python -m schedext.cli evaluate <weights.pt>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import (annotate, annserver, dataset, detect, evaluate, export, locate,
               manifest as manifest_mod, pdfio, pipeline, viewer)

PDF_DIR = Path("all_plansets")
OUT_DIR = Path("out")
MANIFEST = OUT_DIR / "manifest.jsonl"
CANDIDATES = OUT_DIR / "candidates.jsonl"
RESULTS = OUT_DIR / "plansets"


def cmd_manifest() -> None:
    records = manifest_mod.build(PDF_DIR, MANIFEST)
    print(manifest_mod.summarise(records))


def cmd_locate() -> None:
    found = locate.run(MANIFEST, CANDIDATES)
    pages = sum(len(v) for v in found.values())
    with_any = sum(1 for v in found.values() if v)
    print(f"{with_any}/{len(found)} plansets carry a schedule sheet; {pages} sheets")


def cmd_extract() -> None:
    summaries = pipeline.run(MANIFEST, CANDIDATES, RESULTS)
    total = sum(s.get("items", 0) or 0 for s in summaries)
    review = sum(s.get("needs_review", 0) or 0 for s in summaries)
    with_items = sum(1 for s in summaries if s.get("items"))

    rows = export.write_csv(RESULTS, OUT_DIR / "schedule_items.csv")
    (OUT_DIR / "type_summary.json").write_text(
        json.dumps(export.type_summary(RESULTS), indent=2)
    )

    print(f"{with_items}/{len(summaries)} plansets yielded items")
    print(f"{total} schedule items, {review} flagged for review "
          f"({100 * review / total:.1f}%)" if total else "no items")
    print(f"wrote {rows} rows to {OUT_DIR / 'schedule_items.csv'}")
    print(f"per-planset JSON in {RESULTS}/")

    worst = sorted(summaries, key=lambda s: -(s.get("warnings") or 0))[:8]
    print("\nmost warnings:")
    for entry in worst:
        print(f"  {entry['file'][:44]:46s} items={entry.get('items', 0):4} "
              f"warnings={entry.get('warnings', 0):3} health={entry['health']}")


WEIGHTS = "/var/www/html/WindowsDoorsClassification/best.pt"


def cmd_detect(limit, conf, no_crops, group, weights, workers) -> None:
    """Run best.pt over the located schedule sheets and write coordinates."""
    if workers > 1:
        stats = detect.run_parallel(MANIFEST, CANDIDATES, OUT_DIR, weights,
                                    workers=workers, limit=limit, conf=conf,
                                    crops=not no_crops)
    else:
        stats = detect.run(MANIFEST, CANDIDATES, OUT_DIR, weights,
                           limit=limit, conf=conf, crops=not no_crops,
                           do_group=group)
    if stats.get("errors"):
        print(f"  {stats['errors']} sheet(s) failed — see out/detect_summary.json")
    print(f"{stats['sheets']} sheets · {stats['openings']} openings "
          f"({stats['in_block']} inside a schedule block, {stats['outside']} outside)")
    print(f"  found by: whole-sheet pass {stats['by_source']['sheet']}, "
          f"per-block pass {stats['by_source']['block']}")
    print(f"  wrote {OUT_DIR / 'openings.csv'} and {OUT_DIR / 'openings.jsonl'}")
    if not no_crops:
        print(f"  crops in {OUT_DIR / 'crops'}/")


ANN_DIR = Path("annotations")
VIEWER_DIR = Path("viewer")


def cmd_annotate(port: int, seed_only: bool, force: bool) -> None:
    """Seed annotations from pipeline output, then serve the editor."""
    index = VIEWER_DIR / "index.json"
    if not index.exists():
        print("run `cli viewer --no-serve` first to render sheets")
        return
    stats = annotate.seed_corpus(index, ANN_DIR, force=force)
    print(f"seeded {stats['seeded']} sheets ({stats['skipped']} already annotated)")
    print(f"  {stats['areas']} areas · {stats['needs_review_areas']} need a type")
    print(f"  {stats['openings']} openings · {stats['accepted']} pre-accepted · "
          f"{stats['pending']} to review")
    if seed_only:
        return
    annserver.serve(VIEWER_DIR, ANN_DIR, port)


def cmd_annotate_merge(apply: bool) -> None:
    """Fold a fresh detector run into existing annotations."""
    totals = annotate.merge_corpus(VIEWER_DIR / "index.json", ANN_DIR, apply=apply)
    mode = "APPLIED" if apply else "DRY RUN (pass --apply to write)"
    print(f"{mode}\n")
    print(f"  {totals['sheets']} sheets merged, {totals['seeded_new']} newly seeded")
    print(f"  {totals['kept_human']} human/edited objects kept untouched")
    print(f"  {totals['refreshed']} model objects refreshed")
    print(f"  {totals['added_pending']} new boxes added as pending")
    print(f"  {totals['downgraded']} verified sheets downgraded to needs_recheck")


def cmd_export_dataset(tag: str, no_negatives: bool) -> None:
    """Turn verified annotations into a YOLO dataset."""
    report = dataset.build(ANN_DIR, MANIFEST, OUT_DIR / "dataset",
                           tag=tag, negatives=not no_negatives)
    print(f"exported {report['images']} images "
          f"({report['positives']} with boxes, {report['negatives']} background)")
    print(f"  {report['boxes']} boxes: {report['classes']}")
    print(f"  train {report['train']['images']} imgs / {report['train']['boxes']} boxes "
          f"({report['plansets']['train']} plansets)  {report['train']['by_class']}")
    print(f"  val   {report['val']['images']} imgs / {report['val']['boxes']} boxes "
          f"({report['plansets']['val']} plansets)  {report['val']['by_class']}")
    if report["duplicates_dropped"]:
        print(f"  dropped {len(report['duplicates_dropped'])} duplicate sheet(s)")
    for w in report["warnings"]:
        print(f"  WARNING: {w}")
    print(f"  -> {OUT_DIR / 'dataset' / tag}")


def cmd_train(tag: str, arms: str, skip_gate: bool, imgsz, epochs,
              patience, batch, workers, cache) -> None:
    """Train the stage-2 opening detector."""
    from . import train as train_mod

    argv = ["--tag", tag, "--arms", arms]
    if skip_gate:
        argv.append("--skip-gate")
    if cache:
        argv.append("--cache")
    for flag, value in (("--imgsz", imgsz), ("--epochs", epochs),
                        ("--patience", patience), ("--batch", batch),
                        ("--workers", workers)):
        if value is not None:
            argv += [flag, str(value)]
    train_mod.main(argv)


def cmd_evaluate(weights: str, tag: str, conf: float) -> None:
    """Score a trained detector on the held-out plansets."""
    evaluate.main([weights, "--tag", tag, "--conf", str(conf)])


def cmd_viewer(port: int, no_serve: bool) -> None:
    """Render the located sheets with their boxes, then serve them locally."""
    out = Path("viewer")
    stats = viewer.build(MANIFEST, RESULTS, out)
    print(f"rendered {stats['sheets']} sheets · {stats['blocks']} blocks · "
          f"{stats['boxed_items']}/{stats['items']} items have boxes")
    if no_serve:
        print(f"open {out / 'index.html'}")
        return

    import functools
    import http.server
    import socketserver

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(out))
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", port), handler) as httpd:
        print(f"\n  http://localhost:{port}\n\nCtrl-C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")


def cmd_one(pdf: str, page: int | None) -> None:
    path = Path(pdf)
    record = pdfio.inspect(path).to_dict()
    candidates = [c.to_dict() for c in locate.locate(path, record)]
    if page is not None:
        candidates = [c for c in candidates if c["page_index"] == page - 1]
    result = pipeline.run_one(path, record, candidates)
    print(json.dumps(result, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="schedext")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("all")
    sub.add_parser("manifest")
    sub.add_parser("locate")
    sub.add_parser("extract")
    one = sub.add_parser("one")
    one.add_argument("pdf")
    one.add_argument("page", nargs="?", type=int)
    det = sub.add_parser("detect")
    det.add_argument("--limit", type=int, default=None,
                     help="only process the first N sheets (smoke run)")
    det.add_argument("--conf", type=float, default=detect.CONF)
    det.add_argument("--no-crops", action="store_true")
    det.add_argument("--group-mullions", action="store_true",
                     help="merge mulled leaves into one box per assembly")
    det.add_argument("--weights", default=WEIGHTS)
    det.add_argument("--workers", type=int, default=3,
                     help="worker processes; 1 thread each (threading does not "
                          "scale on this model). 1 = single process.")
    ann = sub.add_parser("annotate")
    ann.add_argument("--port", type=int, default=8000)
    ann.add_argument("--seed-only", action="store_true")
    ann.add_argument("--force", action="store_true",
                     help="re-seed sheets that already have annotations (destructive)")
    mrg = sub.add_parser("annotate-merge")
    mrg.add_argument("--apply", action="store_true")
    exp = sub.add_parser("export-dataset")
    exp.add_argument("--tag", default="v1")
    exp.add_argument("--no-negatives", action="store_true")
    trn = sub.add_parser("train")
    trn.add_argument("--tag", default="v1")
    trn.add_argument("--arms", default="coco-s",
                     help="comma-separated: best,coco-l,coco-s  (or 'all')")
    trn.add_argument("--skip-gate", action="store_true")
    trn.add_argument("--imgsz", type=int, default=None)
    trn.add_argument("--epochs", type=int, default=None)
    trn.add_argument("--patience", type=int, default=None)
    trn.add_argument("--batch", type=int, default=None)
    trn.add_argument("--workers", type=int, default=None)
    trn.add_argument("--cache", action="store_true")
    ev = sub.add_parser("evaluate")
    ev.add_argument("weights")
    ev.add_argument("--tag", default="v1")
    ev.add_argument("--conf", type=float, default=evaluate.CONF)
    view = sub.add_parser("viewer")
    view.add_argument("--port", type=int, default=8000)
    view.add_argument("--no-serve", action="store_true")

    args = parser.parse_args(argv)
    OUT_DIR.mkdir(exist_ok=True)

    if args.command == "all":
        cmd_manifest()
        cmd_locate()
        cmd_extract()
    elif args.command == "manifest":
        cmd_manifest()
    elif args.command == "locate":
        cmd_locate()
    elif args.command == "extract":
        cmd_extract()
    elif args.command == "detect":
        cmd_detect(args.limit, args.conf, args.no_crops,
                   args.group_mullions, args.weights, args.workers)
    elif args.command == "annotate":
        cmd_annotate(args.port, args.seed_only, args.force)
    elif args.command == "annotate-merge":
        cmd_annotate_merge(args.apply)
    elif args.command == "export-dataset":
        cmd_export_dataset(args.tag, args.no_negatives)
    elif args.command == "train":
        cmd_train(args.tag, args.arms, args.skip_gate, args.imgsz,
                  args.epochs, args.patience, args.batch, args.workers,
                  args.cache)
    elif args.command == "evaluate":
        cmd_evaluate(args.weights, args.tag, args.conf)
    elif args.command == "viewer":
        cmd_viewer(args.port, args.no_serve)
    elif args.command == "one":
        cmd_one(args.pdf, args.page)
    return 0


if __name__ == "__main__":
    sys.exit(main())
