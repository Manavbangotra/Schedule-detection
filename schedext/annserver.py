"""Local HTTP server for the annotation editor.

The read-only viewer is static files under `python -m http.server`, which
cannot save. This is the smallest thing that can: the same static directory,
plus a handful of JSON endpoints that write one file per sheet.

Stdlib only -- no Flask. It binds to 127.0.0.1 and validates every sheet key
against a strict pattern before it touches a path, because this process serves
a directory and accepts writes.
"""

from __future__ import annotations

import json
import re
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import annotate

# project_sha8_pN. Anything else never reaches the filesystem.
KEY_PATTERN = re.compile(r"^[A-Za-z0-9]{1,16}_[0-9a-f]{8}_p\d{1,4}$")

MAX_BODY = 4 * 1024 * 1024


class Handler(SimpleHTTPRequestHandler):
    """Static viewer plus /api/. Anything not under /api/ falls through."""

    ann_dir = annotate.ANN_DIR

    def log_message(self, fmt, *args):        # quieter than the default
        if "api" in (args[0] if args else ""):
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

    def _key(self) -> str | None:
        key = self.path.rsplit("/", 1)[-1]
        return key if KEY_PATTERN.match(key) else None

    def _body(self) -> dict | None:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY:
            return None
        try:
            return json.loads(self.rfile.read(length))
        except Exception:
            return None

    # --- routes ----------------------------------------------------------
    def do_GET(self):
        if self.path.startswith("/api/progress"):
            return self._json(annotate.progress(annotate.load_all(self.ann_dir)))
        if self.path.startswith("/api/ann/"):
            key = self._key()
            if not key:
                return self._json({"error": "bad key"}, 400)
            sheet = annotate.load(key, self.ann_dir)
            if sheet is None:
                return self._json({"error": "not seeded"}, 404)
            return self._json(sheet.to_dict())
        return super().do_GET()

    def do_PUT(self):
        if not self.path.startswith("/api/ann/"):
            return self._json({"error": "not found"}, 404)
        key = self._key()
        if not key:
            return self._json({"error": "bad key"}, 400)
        payload = self._body()
        if payload is None:
            return self._json({"error": "bad body"}, 400)

        current = annotate.load(key, self.ann_dir)
        base_rev = payload.get("base_rev")
        if current is not None and base_rev is not None and base_rev < current.rev:
            # Another tab saved first. Never clobber; hand back the server copy.
            return self._json({"error": "conflict", "server": current.to_dict()}, 409)

        sheet = annotate.SheetAnnotation.from_dict(payload["sheet"])
        sheet.rev = (current.rev + 1) if current else 1
        for opening in sheet.openings:
            annotate.assign(opening, sheet.areas)
        annotate.save(sheet, self.ann_dir)
        return self._json({"rev": sheet.rev, "saved": sheet.updated_at})

    def do_POST(self):
        # sendBeacon cannot issue PUT, so accept the same payload on POST.
        return self.do_PUT()


def serve(directory: Path, ann_dir: Path, port: int = 8000) -> None:
    Handler.ann_dir = ann_dir
    handler = partial(Handler, directory=str(directory))
    ThreadingHTTPServer.allow_reuse_address = True
    with ThreadingHTTPServer(("127.0.0.1", port), handler) as server:
        print(f"\n  annotation editor:  http://localhost:{port}/annotate.html")
        print(f"  read-only viewer:   http://localhost:{port}/index.html")
        print(f"  saving to:          {ann_dir}/\n\nCtrl-C to stop.")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")
