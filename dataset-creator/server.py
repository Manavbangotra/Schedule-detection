"""dataset-creator — drop in a PDF, get an annotated opening dataset.

Run:  ./dataset-creator/run.sh        (or python dataset-creator/server.py)

Stdlib HTTP only. Serves the static UI, accepts PDF uploads, runs the schedext
pipeline in the background, and reads/writes the *same* annotation store the
schedext annotator uses -- one store, two front-ends, so nothing diverges and
there is never a question about which copy the export reads.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import jobs as jobs_mod                       # noqa: E402
from schedext import annotate                 # noqa: E402

DATA = HERE / "data"
STATIC = HERE / "static"
WEIGHTS = "/var/www/html/WindowsDoorsClassification/best.pt"

# The shared store. Pointing at the repo-level annotations/ rather than a copy
# under data/ is deliberate -- see the module docstring.
ANN_DIR = ROOT / "annotations"

KEY_PATTERN = re.compile(r"^[A-Za-z0-9]{1,16}_[0-9a-f]{8}_p\d{1,4}$")
PDF_ID_PATTERN = re.compile(r"^[0-9a-f]{8}$")
MAX_UPLOAD = 500 * 1024 * 1024
MAX_BODY = 8 * 1024 * 1024

runner: jobs_mod.Runner | None = None


def _parse_multipart(body: bytes, content_type: str) -> tuple[str, bytes | None]:
    """Pull the first file part out of a multipart/form-data body.

    Hand-rolled because `cgi` is removed in Python 3.13 and this needs exactly
    one thing: the bytes of one uploaded file. The filename is returned only for
    display -- the stored name comes from the content hash.
    """
    match = re.search(r"boundary=([^;]+)", content_type)
    if not match:
        return "upload.pdf", None
    boundary = b"--" + match.group(1).strip('"').encode()

    filename = "upload.pdf"
    for part in body.split(boundary):
        head, sep, payload = part.partition(b"\r\n\r\n")
        if not sep or b"filename=" not in head:
            continue
        name_match = re.search(rb'filename="([^"]*)"', head)
        if name_match and name_match.group(1):
            filename = name_match.group(1).decode("utf-8", "replace")
        # Exactly one trailing CRLF belongs to the framing, not the file.
        # rstrip() with a character set is wrong here -- it ate the newline a
        # PDF legitimately ends with after %%EOF, changing the content hash and
        # defeating the dedup that hash is for.
        if payload.endswith(b"\r\n"):
            payload = payload[:-2]
        return filename, payload
    return filename, None



class Handler(SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        if args and "/api/" in str(args[0]):
            return
        super().log_message(fmt, *args)

    # --- helpers ---------------------------------------------------------
    def _json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict | None:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY:
            return None
        try:
            return json.loads(self.rfile.read(length))
        except Exception:
            return None

    # --- library ---------------------------------------------------------
    def _library(self) -> dict:
        sheets = annotate.load_all(ANN_DIR)
        by_planset: dict[str, dict] = {}
        totals = {"sheets": 0, "verified": 0, "boxes": 0,
                  "window": 0, "door": 0, "garage_door": 0}

        for sheet in sheets.values():
            entry = by_planset.setdefault(sheet.planset_key, {
                "planset_key": sheet.planset_key,
                "project_id": sheet.project_id,
                "file": sheet.file, "sheets": 0, "verified": 0, "boxes": 0,
            })
            entry["sheets"] += 1
            totals["sheets"] += 1
            if sheet.status == "verified":
                entry["verified"] += 1
                totals["verified"] += 1
            live = [o for o in sheet.openings if o.state != "rejected"]
            entry["boxes"] += len(live)
            totals["boxes"] += len(live)
            for opening in live:
                if opening.cls in totals:
                    totals[opening.cls] += 1

        totals["plansets"] = len(by_planset)
        door, window = totals["door"], totals["window"]
        totals["door_window_ratio"] = round(door / window, 2) if window else None
        return {
            "plansets": sorted(by_planset.values(),
                               key=lambda e: (e["project_id"], e["planset_key"])),
            "totals": totals,
            "jobs": [j.to_dict() for j in
                     sorted(runner.jobs.values(), key=lambda j: -j.started)[:20]],
        }

    # --- routes ----------------------------------------------------------
    def do_GET(self):
        path = self.path.split("?", 1)[0]

        if path == "/" or path == "":
            self.path = "/index.html"
            return super().do_GET()

        if path == "/api/library":
            return self._json(self._library())

        if path == "/api/duplicates":
            return self._json({"groups": _duplicate_groups(annotate.load_all(ANN_DIR))})

        if path.startswith("/api/jobs/"):
            job = runner.jobs.get(path.rsplit("/", 1)[-1])
            return self._json(job.to_dict() if job else {"error": "unknown job"},
                              200 if job else 404)

        if path == "/api/progress":
            return self._json(annotate.progress(annotate.load_all(ANN_DIR)))

        if path.startswith("/api/ann/"):
            key = path.rsplit("/", 1)[-1]
            if not KEY_PATTERN.match(key):
                return self._json({"error": "bad key"}, 400)
            sheet = annotate.load(key, ANN_DIR)
            if sheet is None:
                return self._json({"error": "not seeded"}, 404)
            return self._json(sheet.to_dict())

        # Sheet images live under data/sheets or the repo's viewer/pages, so the
        # tool can reuse renders that already exist instead of redoing them.
        if path.startswith("/sheets/") or path.startswith("/pages/"):
            for base in (DATA, ROOT / "viewer"):
                candidate = base / path.lstrip("/")
                if candidate.is_file():
                    return self._send_file(candidate)
            return self._json({"error": "not found"}, 404)

        return super().do_GET()

    def _send_file(self, path: Path) -> None:
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        path = self.path.split("?", 1)[0]

        if path == "/api/upload":
            return self._upload()

        if path == "/api/import-existing":
            return self._import_existing()

        if path.startswith("/api/ann/"):
            return self.do_PUT()

        return self._json({"error": "not found"}, 404)

    def _upload(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_UPLOAD:
            return self._json({"error": "empty or too large (500MB cap)"}, 400)

        filename, raw = _parse_multipart(
            self.rfile.read(length), self.headers.get("Content-Type", ""))
        if raw is None:
            return self._json({"error": "could not parse the upload"}, 400)

        # Trust the bytes, not the name. The stored filename is derived from the
        # content hash, so a hostile upload name never reaches the filesystem.
        if not raw.startswith(b"%PDF"):
            return self._json({"error": "not a PDF (magic bytes)"}, 400)

        staging = DATA / "pdfs" / f".upload-{os.getpid()}-{time.time_ns()}.tmp"
        staging.parent.mkdir(parents=True, exist_ok=True)
        staging.write_bytes(raw)
        try:
            pdf_id, _ = runner.register(staging, filename)
        finally:
            staging.unlink(missing_ok=True)

        job = runner.submit(pdf_id, filename)
        return self._json({"pdf_id": pdf_id, "job_id": job.job_id})

    def _import_existing(self):
        """Register PDFs already on disk (the 59-planset corpus) without copying."""
        payload = self._body() or {}
        folder = Path(payload.get("dir") or (ROOT / "all_plansets"))
        if not folder.is_dir():
            return self._json({"error": f"no such directory: {folder}"}, 400)
        added = []
        for pdf in sorted(folder.glob("*.pdf")):
            try:
                pdf_id, _ = runner.register(pdf, pdf.name, copy=False)
                added.append(pdf_id)
            except Exception:
                continue
        return self._json({"registered": len(added)})

    def do_DELETE(self):
        path = self.path.split("?", 1)[0]
        if not path.startswith("/api/ann/"):
            return self._json({"error": "not found"}, 404)
        key = path.rsplit("/", 1)[-1]
        if not KEY_PATTERN.match(key):
            return self._json({"error": "bad key"}, 400)
        target = annotate.path_for(key, ANN_DIR)
        if not target.exists():
            return self._json({"error": "not found"}, 404)
        # Keep a copy under .history -- deleting a sheet should be undoable,
        # and the annotations directory is hand-made data.
        graveyard = ANN_DIR / ".history" / key
        graveyard.mkdir(parents=True, exist_ok=True)
        (graveyard / "deleted.json").write_text(target.read_text())
        target.unlink()
        return self._json({"deleted": key})

    def do_PUT(self):
        path = self.path.split("?", 1)[0]
        if not path.startswith("/api/ann/"):
            return self._json({"error": "not found"}, 404)
        key = path.rsplit("/", 1)[-1]
        if not KEY_PATTERN.match(key):
            return self._json({"error": "bad key"}, 400)
        payload = self._body()
        if payload is None:
            return self._json({"error": "bad body"}, 400)

        current = annotate.load(key, ANN_DIR)
        base_rev = payload.get("base_rev")
        if current is not None and base_rev is not None and base_rev < current.rev:
            return self._json({"error": "conflict", "server": current.to_dict()}, 409)

        sheet = annotate.SheetAnnotation.from_dict(payload["sheet"])
        sheet.rev = (current.rev + 1) if current else 1
        for opening in sheet.openings:
            annotate.assign(opening, sheet.areas)
        annotate.save(sheet, ANN_DIR)
        return self._json({"rev": sheet.rev, "saved": sheet.updated_at})



def _page_words(sheet) -> frozenset:
    """The set of words printed on a sheet, used as a content fingerprint."""
    import pymupdf

    record = _manifest_by_sha().get(sheet.pdf_sha256[:8])
    if not record:
        return frozenset()
    try:
        with pymupdf.open(record["path"]) as doc:
            page = doc[sheet.page - 1]
            return frozenset(w[4].upper() for w in page.get_text("words") if len(w[4]) > 2)
    except Exception:
        return frozenset()


_MANIFEST_CACHE: dict | None = None


def _manifest_by_sha() -> dict:
    global _MANIFEST_CACHE
    if _MANIFEST_CACHE is None:
        _MANIFEST_CACHE = {}
        manifest = ROOT / "out" / "manifest.jsonl"
        if manifest.exists():
            for line in manifest.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    _MANIFEST_CACHE[rec["sha256"][:8]] = rec
    return _MANIFEST_CACHE


DUPLICATE_JACCARD = 0.95

_FINGERPRINTS: dict[str, frozenset] = {}


def _duplicate_groups(sheets: dict) -> list[dict]:
    """Sheets that are the same drawing, compared by content not by name.

    Two things make this necessary. Big plansets repeat a schedule page once per
    building type. And -- less obvious, more damaging -- the same drawing turns
    up under different project ids: 543, 558 and 559 carry byte-identical door
    and window schedules. Comparing within a project misses those entirely, and
    they are the worst case, because splitting by planset then puts one copy in
    train and its twin in val.

    Fingerprint is the set of words on the page; Jaccard >= 0.95 is the same
    drawing. Below that (0.83-0.88 is common here) means the same architect's
    template with different openings, which is genuinely new data and is kept.
    """
    live = [s for s in sheets.values() if s.status != "skipped"]
    for sheet in live:
        if sheet.sheet_key not in _FINGERPRINTS:
            _FINGERPRINTS[sheet.sheet_key] = _page_words(sheet)

    used: set[str] = set()
    out = []
    for i, sheet in enumerate(live):
        if sheet.sheet_key in used:
            continue
        words = _FINGERPRINTS[sheet.sheet_key]
        if not words:
            continue
        members = [sheet]
        for other in live[i + 1:]:
            if other.sheet_key in used:
                continue
            other_words = _FINGERPRINTS[other.sheet_key]
            if not other_words:
                continue
            union = len(words | other_words)
            if union and len(words & other_words) / union >= DUPLICATE_JACCARD:
                members.append(other)
        if len(members) < 2:
            continue
        for m in members:
            used.add(m.sheet_key)
        projects = sorted({m.project_id for m in members})
        out.append({
            "project_id": "+".join(projects),
            "cross_project": len(projects) > 1,
            "titles": sorted({a.seed.get("title", "") for a in sheet.areas if a.seed.get("title")})[:3],
            "keys": [m.sheet_key for m in members],
            "verified": [m.sheet_key for m in members if m.status == "verified"],
            "count": len(members),
        })
    return sorted(out, key=lambda g: -g["count"])


def main() -> int:
    global runner
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8010
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "sheets").mkdir(exist_ok=True)
    runner = jobs_mod.Runner(DATA, WEIGHTS, ann_dir=ANN_DIR)

    handler = partial(Handler, directory=str(STATIC))
    ThreadingHTTPServer.allow_reuse_address = True
    with ThreadingHTTPServer(("127.0.0.1", port), handler) as server:
        print(f"\n  dataset-creator   http://localhost:{port}")
        print(f"  annotations       {ANN_DIR}  (shared with schedext)")
        print(f"  uploads           {DATA / 'pdfs'}\n\nCtrl-C to stop.")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
