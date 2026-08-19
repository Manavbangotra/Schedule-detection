"""Background pipeline runner for dataset-creator.

Uploading a PDF kicks off a job: inspect -> locate schedule sheets -> per sheet
render, segment areas, detect openings, seed an annotation. Detection dominates
the cost (~23s per sheet plus ~15s per area on CPU), so a typical planset takes
two to five minutes and the UI must not block on it.

One worker thread, one job at a time, and the YOLO model is loaded **once** and
kept -- loading best.pt costs about four seconds and paying that per sheet would
double the wall clock on small plansets.

Annotations are written to the same store the schedext annotator uses. Not a
copy: one store, one truth, so work done in either front-end is visible in the
other and there is never a question about which one the export reads.
"""

from __future__ import annotations

import hashlib
import queue
import shutil
import threading
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path

import pymupdf

from schedext import annotate, detect, genre, locate, pdfio, segment

RENDER_ZOOM = detect.ZOOM          # 1.8 -- must match, boxes are stored in points
THUMB_ZOOM = 0.25
JPEG_QUALITY = 78


@dataclass
class Job:
    job_id: str
    pdf_id: str
    filename: str
    stage: str = "queued"
    message: str = ""
    sheet_done: int = 0
    sheet_total: int = 0
    sheet_keys: list[str] = field(default_factory=list)
    health: str = ""
    page_count: int = 0
    needs_page_pick: bool = False
    error: str = ""
    started: float = field(default_factory=time.time)
    finished: float = 0.0

    def to_dict(self) -> dict:
        data = asdict(self)
        data["elapsed_s"] = round((self.finished or time.time()) - self.started, 1)
        return data


class Runner:
    """Serial job queue with a warm model."""

    def __init__(self, data_dir: Path, weights: str):
        self.data = data_dir
        self.weights = weights
        self.pdfs = data_dir / "pdfs"
        self.sheets = data_dir / "sheets"
        self.ann_dir = data_dir / "annotations"
        for d in (self.pdfs, self.sheets, self.ann_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.jobs: dict[str, Job] = {}
        self._queue: queue.Queue[str] = queue.Queue()
        self._model = None
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._work, daemon=True)
        self._thread.start()

    # --- model ----------------------------------------------------------
    def _get_model(self):
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self.weights)
        return self._model

    # --- registration ---------------------------------------------------
    def register(self, source: Path, original_name: str, copy: bool = True) -> tuple[str, Path]:
        """Content-address a PDF. Re-uploading a known file is a no-op.

        The stored name is derived from the file's own hash, never from the
        client's filename -- that is both the dedup mechanism and the reason a
        malicious upload name cannot reach the filesystem.
        """
        digest = pdfio.sha256(source)
        pdf_id = digest[:8]
        target = self.pdfs / f"{pdf_id}.pdf"
        if not target.exists():
            if copy:
                shutil.copy2(source, target)
            else:
                target.symlink_to(source.resolve())
        meta = self.pdfs / f"{pdf_id}.json"
        if not meta.exists():
            import json

            meta.write_text(json.dumps({
                "pdf_id": pdf_id, "sha256": digest,
                "original_name": original_name,
                "source": str(source.resolve()),
                "added": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }, indent=1))
        return pdf_id, target

    def submit(self, pdf_id: str, filename: str) -> Job:
        job_id = f"j{int(time.time() * 1000):x}"
        job = Job(job_id=job_id, pdf_id=pdf_id, filename=filename)
        self.jobs[job_id] = job
        self._queue.put(job_id)
        return job

    # --- worker ---------------------------------------------------------
    def _work(self) -> None:
        while True:
            job_id = self._queue.get()
            job = self.jobs[job_id]
            try:
                self._run(job)
            except Exception as exc:
                job.stage = "error"
                job.error = f"{type(exc).__name__}: {exc}"[:300]
                job.message = traceback.format_exc(limit=2)[-300:]
            finally:
                job.finished = time.time()
                self._queue.task_done()


    def _project_id(self, pdf_id: str, fallback: str) -> str:
        """Project id from the name the PDF was uploaded under."""
        import json as _json

        meta = self.pdfs / f"{pdf_id}.json"
        if meta.exists():
            try:
                original = _json.loads(meta.read_text()).get("original_name", "")
                if original:
                    derived = pdfio.project_id(Path(original))
                    if derived:
                        return derived
            except Exception:
                pass
        return fallback

    def _run(self, job: Job) -> None:
        path = self.pdfs / f"{job.pdf_id}.pdf"

        job.stage = "inspect"
        job.message = "reading the PDF"
        record = pdfio.inspect(path).to_dict()
        # Files are stored under their content hash, so the stored name carries
        # no project id. Recover it from the name the file arrived with,
        # otherwise every re-upload mints a fresh planset key and the store
        # gains duplicates instead of merging.
        record["project_id"] = self._project_id(job.pdf_id, record["project_id"])
        job.health = record["health"]
        job.page_count = record["page_count"]

        if record["health"] in {"truncated", "unopenable"}:
            job.stage = "error"
            job.error = f"{record['health']}: {record.get('note', '')}"
            return

        job.stage = "locate"
        job.message = "looking for schedule sheets"
        candidates = locate.locate(path, record) if record["has_text_layer"] else []

        if not candidates:
            # No text layer, or none of its titles look like a schedule. Either
            # way the human has to say which pages matter.
            job.stage = "needs_pages"
            job.needs_page_pick = True
            job.message = ("no text layer — pick the schedule pages by hand"
                           if not record["has_text_layer"]
                           else "no schedule titles found — pick the pages by hand")
            return

        job.sheet_total = len(candidates)
        job.stage = "detect"
        self._process(job, path, record, candidates)
        job.stage = "done"
        job.message = f"{len(job.sheet_keys)} sheet(s) ready"

    def _process(self, job: Job, path: Path, record: dict,
                 candidates: list) -> None:
        model = self._get_model()
        sha8 = record["sha256"][:8]
        project = record["project_id"]

        with pymupdf.open(path) as doc:
            for candidate in candidates:
                page_no = candidate.page_index + 1
                job.message = f"sheet {job.sheet_done + 1} of {job.sheet_total} (page {page_no})"
                page = doc[candidate.page_index]

                key = f"{project}_{sha8}_p{page_no}"
                image_rel = f"sheets/{key}.jpg"
                image_abs = self.data / image_rel
                if not image_abs.exists():
                    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(RENDER_ZOOM, RENDER_ZOOM))
                    pixmap.save(image_abs, jpg_quality=JPEG_QUALITY)
                    width, height = pixmap.width, pixmap.height
                else:
                    probe = pymupdf.Pixmap(image_abs)
                    width, height = probe.width, probe.height

                blocks = genre.annotate(page, segment.segment(page, candidate))
                boxes = detect.detect_sheet(model, page, blocks, RENDER_ZOOM, detect.CONF)

                meta = {
                    "sheet_key": key, "project_id": project,
                    "planset_key": f"{project}_{sha8}",
                    "file": path.name, "pdf_sha256": record["sha256"],
                    "page": page_no, "rotation": page.rotation,
                    "image": image_rel, "width": width, "height": height,
                    "page_rect_pt": [0, 0, round(page.rect.width, 1),
                                     round(page.rect.height, 1)],
                }
                fresh = annotate.seed_sheet(
                    meta,
                    [b.to_dict() for b in blocks],
                    [{"rect": [round(v, 1) for v in box], "confidence": round(conf, 3),
                      "source": src} for box, conf, src in boxes],
                    RENDER_ZOOM,
                )

                # Never clobber human work: if this sheet has been annotated
                # before, fold the new detections in rather than overwrite.
                existing = annotate.load(key, self.ann_dir)
                if existing is None:
                    annotate.save(fresh, self.ann_dir)
                else:
                    merged, _ = annotate.merge(existing, fresh)
                    annotate.save(merged, self.ann_dir)

                job.sheet_keys.append(key)
                job.sheet_done += 1
