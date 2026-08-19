/* dataset-creator shell: library, upload, job progress, and the annotator.
 *
 * The annotator itself is imported unchanged from annotate.js -- the same
 * editor that is already tested against the corpus. This file only adds the
 * views around it.
 */
import { boot, open as openSheet } from './annotate.js';

const $ = s => document.querySelector(s);
const esc = s => String(s ?? '').replace(/[&<>"]/g,
  c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

let view = 'library', booted = false, watching = new Set();

function show(next) {
  view = next;
  $('#libview').hidden = next !== 'library';
  $('#annview').hidden = next !== 'annotate';
  document.querySelectorAll('.tab').forEach(t =>
    t.classList.toggle('on', t.dataset.v === next));
  if (next === 'annotate' && !booted) { booted = true; boot(); }
}
document.querySelectorAll('.tab').forEach(t => t.onclick = () => show(t.dataset.v));

/* ---------- library ---------- */
async function refresh() {
  const lib = await (await fetch('/api/library')).json();
  const t = lib.totals;

  $('#stats').innerHTML = [
    ['Plansets', t.plansets], ['Sheets', t.sheets], ['Verified', t.verified],
    ['Boxes', t.boxes], ['Windows', t.window], ['Doors', t.door],
  ].map(([k, v]) => `<div class="stat"><b>${v}</b><span>${k}</span></div>`).join('');

  $('#totals').textContent =
    `${t.verified}/${t.sheets} verified · ${t.boxes} boxes`
    + (t.door_window_ratio ? ` · door:window ${t.door_window_ratio}:1` : '');

  $('#libbody').innerHTML = lib.plansets.map(p => `
    <tr data-p="${esc(p.planset_key)}">
      <td><code>${esc(p.planset_key)}</code></td>
      <td>${esc(p.file).slice(0, 46)}</td>
      <td>${p.sheets}</td>
      <td>${p.verified ? p.verified : '<span style="color:var(--muted)">0</span>'}</td>
      <td>${p.boxes}</td>
    </tr>`).join('') || '<tr><td colspan="5" style="color:var(--muted)">nothing yet — drop a PDF above</td></tr>';
  $('#libbody').querySelectorAll('tr[data-p]').forEach(r =>
    r.onclick = () => show('annotate'));

  const active = lib.jobs.filter(j => !['done', 'error'].includes(j.stage));
  $('#jobs').innerHTML = lib.jobs.slice(0, 6).map(j => {
    const pct = j.sheet_total ? Math.round(100 * j.sheet_done / j.sheet_total)
                              : (j.stage === 'done' ? 100 : 8);
    const colour = j.stage === 'error' ? 'var(--bad)'
                 : j.stage === 'done' ? 'var(--ok)' : 'var(--accent)';
    return `<div class="jobrow">
      ${esc(j.filename).slice(0, 40)} — <b>${esc(j.stage)}</b>
      ${esc(j.message || j.error).slice(0, 90)} · ${j.elapsed_s}s
      <div class="jbar"><i style="width:${pct}%;background:${colour}"></i></div>
    </div>`;
  }).join('');

  // Poll only while something is running; idle pages should be silent.
  if (active.length) setTimeout(refresh, 1500);
  else renderDupes();
}


/* ---------- near-duplicate sheets ----------
 * Large plansets repeat the same schedule page once per building type. Keeping
 * all of them costs annotation time and, worse, splits near-identical sheets
 * across train and val -- which quietly inflates the score. Keep one, skip the
 * rest; skipping is reversible and preserves any work already done.
 */
async function renderDupes() {
  const { groups } = await (await fetch('/api/duplicates')).json();
  const el = $('#dupes');
  if (!groups.length) { el.innerHTML = ''; return; }
  const total = groups.reduce((n, g) => n + g.count - 1, 0);
  el.innerHTML = `<h3 style="font-size:13px;margin:0 0 8px">Near-duplicate sheets — ${total} redundant</h3>`
    + groups.map((g, i) => `
      <div class="dupe">
        <b>project ${esc(g.project_id)}</b> · ${g.count} copies ·
        <code>${g.titles.map(esc).join(' / ') || 'no titles'}</code>
        <div class="keys">${g.keys.map((k, j) =>
          `<span class="${j === 0 ? 'keep' : ''}">${esc(k)}${j === 0 ? ' (keep)' : ''}</span>`).join('')}</div>
        <button data-g="${i}" style="margin-top:7px">Skip the other ${g.count - 1}</button>
      </div>`).join('');
  el.querySelectorAll('button[data-g]').forEach(b =>
    b.onclick = () => skipGroup(groups[+b.dataset.g], b));
}

async function skipGroup(group, button) {
  button.disabled = true;
  button.textContent = 'skipping…';
  for (const key of group.keys.slice(1)) {
    const res = await fetch('/api/ann/' + key);
    if (!res.ok) continue;
    const sheet = await res.json();
    sheet.status = 'skipped';
    await fetch('/api/ann/' + key, {
      method: 'PUT', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ base_rev: sheet.rev, sheet }),
    });
  }
  msg(`skipped ${group.keys.length - 1} duplicate sheet(s) from project ${group.project_id}`);
  await renderDupes();
  refresh();
}

/* ---------- upload ---------- */
const drop = $('#drop'), file = $('#file');
drop.onclick = () => file.click();
drop.ondragover = e => { e.preventDefault(); drop.classList.add('hot'); };
drop.ondragleave = () => drop.classList.remove('hot');
drop.ondrop = e => {
  e.preventDefault(); drop.classList.remove('hot');
  if (e.dataTransfer.files.length) upload(e.dataTransfer.files[0]);
};
file.onchange = () => { if (file.files.length) upload(file.files[0]); };

async function upload(f) {
  if (!/\.pdf$/i.test(f.name)) { msg('Not a PDF.', true); return; }
  msg(`uploading ${f.name} (${(f.size / 1e6).toFixed(1)} MB)…`);
  const body = new FormData();
  body.append('file', f, f.name);
  const res = await fetch('/api/upload', { method: 'POST', body });
  const out = await res.json();
  if (!res.ok) { msg(out.error || 'upload failed', true); return; }
  msg(`processing — detection takes a few minutes; sheets appear as they finish`);
  watching.add(out.job_id);
  refresh();
  poll(out.job_id);
}

async function poll(jobId) {
  const j = await (await fetch('/api/jobs/' + jobId)).json();
  if (j.stage === 'error') { msg('failed: ' + j.error, true); return; }
  if (j.stage === 'needs_pages') { msg(j.message, true); return; }
  if (j.stage === 'done') {
    msg(`done — ${j.sheet_done} sheet(s) ready. Open the Annotate tab.`);
    refresh();
    return;
  }
  setTimeout(() => poll(jobId), 1200);
}

const msg = (text, bad) => {
  const el = $('#msg');
  el.textContent = text;
  el.style.color = bad ? 'var(--bad)' : 'var(--muted)';
};

/* ---------- register the PDFs already on disk ---------- */
async function importExisting() {
  const res = await fetch('/api/import-existing', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({}),
  });
  const out = await res.json();
  if (out.registered) msg(`registered ${out.registered} PDF(s) already on disk`);
  refresh();
}

await importExisting();
show('library');
