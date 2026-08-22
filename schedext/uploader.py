"""Drop in a PDF, get the Schedule Extraction Viewer for it.

The viewer under ``viewer/`` is the good UI -- pan/zoom, per-sheet cards, and
independent Blocks / Items / Detections / Labels layers. It is normally driven by
``cli viewer`` over the whole corpus, which needs ``extract`` to have run first.
This serves the same UI for a single uploaded PDF instead: upload, wait, look.

    python -m schedext.uploader [port]        # default 8000

Runs the real pipeline end to end -- locate, extract, detect, render -- so what
you see is what the pipeline produces. Nothing is written to the annotation
store; each upload lives in its own temp directory. For verifying and correcting
proposals into the store, use dataset-creator on :8010 instead.

Weights resolve to the newest trained stage-2 model, overridable with
SCHEDEXT_WEIGHTS.
"""

from __future__ import annotations

import html
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VIEWER = ROOT / "viewer"
MAX_UPLOAD = 500 * 1024 * 1024

JOBS: dict[str, dict] = {}
CURRENT: dict[str, str | None] = {"job": None}
LOCK = threading.Lock()


def _weights() -> str:
    env = os.environ.get("SCHEDEXT_WEIGHTS")
    if env and Path(env).exists():
        return env
    for tag in sorted((ROOT / "out" / "dataset").glob("v*"), reverse=True):
        for rel in ("cv/full/runs/coco-s/weights/best.pt",
                    "runs/coco-s/weights/best.pt"):
            if (tag / rel).exists():
                return str(tag / rel)
    from .train import BEST_PT
    return BEST_PT


def _run_job(job_id: str, pdf: Path, work: Path) -> None:
    from . import detect, link as link_mod, locate, pdfio, pipeline, viewer

    def step(msg: str) -> None:
        with LOCK:
            JOBS[job_id]["status"] = msg

    t0 = time.perf_counter()
    timing: dict[str, float] = {}

    def mark(name: str, since: float) -> float:
        """Record a stage's wall time and return a fresh clock reading."""
        now = time.perf_counter()
        timing[name] = round(now - since, 1)
        with LOCK:
            JOBS[job_id]["timing"] = dict(timing)
        return now

    try:
        step("reading the PDF")
        record = pdfio.inspect(pdf).to_dict()
        pages = record.get("page_count", "?")
        t = mark("read", t0)

        step(f"stage 1 — scanning {pages} pages for schedule sheets")
        found = [c.to_dict() for c in locate.locate(pdf, record)]
        t = mark("locate", t)
        if not found:
            with LOCK:
                JOBS[job_id].update(done=True, sheets=0, status=(
                    f"No schedule sheets found in {pages} pages. Stage 1 reads "
                    f"sheet titles from the text layer, so a scanned PDF with no "
                    f"text will always come back empty."))
            return

        # detect.run and pipeline.run both read a manifest and a candidates file.
        # Give them ones holding only this upload, so nothing touches out/.
        manifest = work / "manifest.jsonl"
        candidates = work / "candidates.jsonl"
        manifest.write_text(json.dumps(record) + "\n")
        candidates.write_text(json.dumps(
            {"file": pdf.name,
             "project_id": record.get("project_id") or "upload",
             "health": record.get("health", "ok"),
             "page_count": record.get("page_count", 0), "error": "",
             "candidates": found}) + "\n")

        step(f"stage 1 — {len(found)} sheet(s); reading schedule rows and sizes")
        pipeline.run(manifest, candidates, work / "plansets")
        t = mark("extract", t)

        step("stage 2 — detecting openings")
        stats = detect.run(manifest, candidates, work, _weights(), crops=True)
        t = mark("detect", t)

        step("rendering sheets")
        built = viewer.build(manifest, work / "plansets", work / "viewer",
                             detections_path=work / "openings.jsonl")
        t = mark("render", t)

        step("linking openings to their schedule rows")
        linked = _link_sizes(link_mod, work)
        mark("link", t)
        timing["total"] = round(time.perf_counter() - t0, 1)

        with LOCK:
            JOBS[job_id].update(
                done=True, stats=stats, built=built, timing=dict(timing),
                linked=linked,
                status=(f"{built.get('sheets', 0)} sheets · "
                        f"{built.get('items', 0)} text items · "
                        f"{stats['openings']} detections · "
                        f"{linked} sized · "
                        f"{timing['detect']}s detect / {timing['total']}s total"))
            CURRENT["job"] = job_id
    except Exception as exc:               # surfaced in the page, never swallowed
        with LOCK:
            JOBS[job_id].update(done=True, error=f"{type(exc).__name__}: {exc}")


def _link_sizes(link_mod, work: Path) -> int:
    """Fold each opening's schedule row into the rendered index.

    viewer.build has already written index.json with a detections list; this
    stamps mark/size onto those entries so the viewer can label a box with what
    it actually is rather than a confidence score. Done after the render rather
    than inside it so viewer.build stays the same function `cli viewer` calls.
    """
    index = work / "viewer" / "index.json"
    openings_csv = work / "openings.csv"
    if not index.exists() or not openings_csv.exists():
        return 0

    import csv as _csv
    with openings_csv.open() as handle:
        openings = list(_csv.DictReader(handle))

    items = []
    for result in (work / "plansets").glob("*.json"):
        items += json.loads(result.read_text()).get("items", [])
    if not items:
        return 0

    linked = link_mod.link(openings, items)
    data = json.loads(index.read_text())
    stamped = 0
    for sheet in data.get("sheets", []):
        for det in sheet.get("detections", []):
            hit = linked.get(det.get("opening_id"))
            if not hit:
                continue
            det["mark"] = hit["mark"]
            det["width_in"] = hit["width_in"]
            det["height_in"] = hit["height_in"]
            det["shape_agrees"] = hit["agrees"]
            if hit.get("width_text") and hit.get("height_text"):
                det["size_text"] = f'{hit["width_text"]} x {hit["height_text"]}'
            stamped += 1
    index.write_text(json.dumps(data, separators=(",", ":")))
    return stamped


# Injected INTO the viewer's sidebar, immediately above the filters.
#
# It must not become a child of <body>: that is a two-column CSS grid
# (grid-template-columns:320px 1fr; overflow:hidden), so an extra top-level
# element becomes a third grid item and breaks the layout. Inside <aside> it is
# just another sidebar control, and it reuses the viewer's own CSS variables so
# it looks native rather than bolted on.
BAR = """<div class="filters" style="gap:6px">
  <label for=upfile style="font-size:11px;letter-spacing:.08em;
         text-transform:uppercase;color:var(--faint);font-family:var(--mono)">
    Upload a planset</label>
  <input type=file id=upfile accept=application/pdf
         style="font:12px var(--body);color:var(--ink)">
  <div id=upstatus style="font-size:11px;color:var(--muted);
       font-family:var(--mono);min-height:1.2em"></div>
  <div style="font-size:10px;color:var(--faint);font-family:var(--mono)">__MODEL__</div>
</div>
<script>
(function(){
 const f=document.getElementById('upfile'),s=document.getElementById('upstatus');
 // Survive the reload that swaps in the new upload, so the breakdown for the
 // run you just did is still on screen when the viewer comes back.
 try{
   const last=sessionStorage.getItem('lastTiming');
   if(last){const t=JSON.parse(last);
     s.textContent=Object.entries(t).map(([k,v])=>k+' '+v+'s').join(' · ');}
 }catch(e){}
 f.onchange=async()=>{
   if(!f.files[0])return;
   f.disabled=true;
   s.textContent='uploading '+f.files[0].name+' …';
   try{
     const r=await fetch('/upload',{method:'POST',body:f.files[0]});
     const {job}=await r.json();
     (async function poll(){
       const d=await (await fetch('/status/'+job)).json();
       s.textContent=d.error?('error: '+d.error):d.status;
       if(!d.done){setTimeout(poll,900);return}
       f.disabled=false;
       if(d.timing){
         sessionStorage.setItem('lastTiming',JSON.stringify(d.timing));
       }
       if(!d.error&&d.sheets!==0){s.textContent=d.status+' — loading…';
                                  setTimeout(()=>location.reload(),500)}
     })();
   }catch(e){s.textContent='upload failed: '+e;f.disabled=false}
 };
})();
// Delete a sheet from the current upload. The viewer re-renders #list on every
// filter keystroke and on select(), so rather than patch renderList -- whose
// reference is already captured by the search box's oninput -- watch the list
// and re-add the buttons whenever it changes.
// Deferred: this markup is injected into the sidebar ABOVE #list, so at parse
// time #list does not exist yet and a direct lookup returns null.
document.addEventListener('DOMContentLoaded',function(){
 const list=document.getElementById('list');
 if(!list)return;
 function decorate(){
   list.querySelectorAll('.row:not([data-del])').forEach(row=>{
     row.dataset.del='1';
     row.style.position='relative';
     const b=document.createElement('button');
     b.textContent='×';
     b.title='Remove this sheet from the view';
     b.style.cssText='position:absolute;top:6px;right:6px;border:0;background:none;'
       +'color:var(--faint);font-size:16px;line-height:1;cursor:pointer;padding:2px 5px';
     b.onmouseenter=()=>b.style.color='var(--box-block)';
     b.onmouseleave=()=>b.style.color='var(--faint)';
     b.onclick=async e=>{
       e.stopPropagation();                 // do not also select the sheet
       b.disabled=true; b.textContent='…';
       try{
         // Resolve the key from the server rather than the page. The viewer
         // declares its state as `let DATA` in a classic script, which is a
         // script-scoped binding and NOT a window property -- window.DATA is
         // undefined. data-i is the index into the unfiltered sheet list, which
         // is exactly the order index.json holds.
         const idx=await (await fetch('/index.json')).json();
         const sheet=idx.sheets[+row.dataset.i];
         if(!sheet)throw new Error('sheet not found');
         const r=await fetch('/sheet/'+encodeURIComponent(sheet.sheet_key),
                             {method:'DELETE'});
         if(!r.ok)throw new Error('HTTP '+r.status);
         location.reload();
       }catch(err){
         b.disabled=false; b.textContent='×'; b.title='delete failed: '+err.message;
       }
     };
     row.appendChild(b);
   });
 }
 new MutationObserver(decorate).observe(list,{childList:true});
 decorate();
})();
</script>
"""


EMPTY = {"zoom": 1.8, "sheets": []}


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, body: bytes, ctype: str, code: int = 200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(json.dumps(obj).encode(), "application/json", code)

    def _job_dir(self) -> Path | None:
        with LOCK:
            job = JOBS.get(CURRENT["job"] or "")
        return Path(job["work"]) / "viewer" if job else None

    def do_GET(self):
        path = self.path.split("?", 1)[0]

        if path in ("/", "/index.html"):
            page = (VIEWER / "index.html").read_text(encoding="utf-8")
            model = Path(_weights())
            bar = BAR.replace("__MODEL__", html.escape(
                f"model: {model.parent.parent.name}/{model.name}"))
            # Before the sidebar's own filters block -- inside <aside>, so the
            # body grid keeps exactly two children. Falls back to after <body>
            # only if the viewer markup ever changes shape.
            marker = '<div class="filters">'
            i = page.find(marker)
            if i >= 0:
                page = page[:i] + bar + page[i:]
            else:
                j = page.lower().find("<body")
                j = page.find(">", j) + 1 if j >= 0 else 0
                page = page[:j] + bar + page[j:]
            return self._send(page.encode("utf-8"), "text/html; charset=utf-8")

        if path == "/index.json":
            d = self._job_dir()
            src = d / "index.json" if d else None
            if src and src.exists():
                return self._send(src.read_bytes(), "application/json")
            return self._json(EMPTY)

        if path.startswith("/pages/"):
            name = path[len("/pages/"):]
            d = self._job_dir()
            # Only by base name, out of this job's own pages dir, so a crafted
            # path cannot walk out of it.
            if not d or "/" in name or "\\" in name or ".." in name:
                return self._json({"error": "not found"}, 404)
            img = d / "pages" / name
            if not img.exists():
                return self._json({"error": "not found"}, 404)
            ctype = "image/jpeg" if img.suffix.lower() in (".jpg", ".jpeg") else "image/png"
            return self._send(img.read_bytes(), ctype)

        if path.startswith("/status/"):
            with LOCK:
                job = JOBS.get(path.rsplit("/", 1)[-1])
            if not job:
                return self._json({"error": "unknown job"}, 404)
            return self._json({k: v for k, v in job.items() if k != "work"})

        self._json({"error": "not found"}, 404)

    def do_DELETE(self):
        """Drop one sheet from the current upload's index.

        View-level only: uploads live in a temp directory and never enter the
        annotation store, so there is nothing durable to protect and re-uploading
        restores it. Deleting from the store is dataset-creator's job, where it
        is tombstoned and undoable.
        """
        path = self.path.split("?", 1)[0]
        if not path.startswith("/sheet/"):
            return self._json({"error": "not found"}, 404)
        from urllib.parse import unquote
        key = unquote(path[len("/sheet/"):])

        d = self._job_dir()
        index = d / "index.json" if d else None
        if not index or not index.exists():
            return self._json({"error": "no upload loaded"}, 404)

        with LOCK:
            data = json.loads(index.read_text())
            before = len(data.get("sheets", []))
            data["sheets"] = [x for x in data.get("sheets", [])
                              if x.get("sheet_key") != key]
            if len(data["sheets"]) == before:
                return self._json({"error": "unknown sheet"}, 404)
            index.write_text(json.dumps(data, separators=(",", ":")))
        return self._json({"removed": key, "sheets": len(data["sheets"])})

    def do_POST(self):
        if not self.path.startswith("/upload"):
            return self._json({"error": "not found"}, 404)
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_UPLOAD:
            return self._json({"error": "bad upload size"}, 400)

        job_id = uuid.uuid4().hex[:12]
        work = Path(tempfile.mkdtemp(prefix="schedext-upload-"))
        pdf = work / "upload.pdf"
        pdf.write_bytes(self.rfile.read(length))

        with LOCK:
            JOBS[job_id] = {"done": False, "status": "queued", "work": str(work)}
        threading.Thread(target=_run_job, args=(job_id, pdf, work),
                         daemon=True).start()
        self._json({"job": job_id})


def main(argv: list[str]) -> int:
    port = int(argv[0]) if argv else 8000
    if not (VIEWER / "index.html").exists():
        print(f"missing {VIEWER / 'index.html'}")
        return 1
    print(f"  model     {_weights()}")
    print(f"  uploader  http://localhost:{port}")
    print("\nCtrl-C to stop.")
    try:
        ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for job in JOBS.values():
            shutil.rmtree(job.get("work", ""), ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
