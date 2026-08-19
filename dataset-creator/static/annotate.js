/* Opening annotator.
 *
 * Rects live in DISPLAY POINTS everywhere in the model, and are converted to
 * pixels only for drawing. Pixels depend on the render zoom; points do not, so
 * re-rendering the corpus at a different zoom cannot shift a single annotation.
 */
const $ = s => document.querySelector(s);
const AREA_TYPES = ['window_area', 'door_area', 'garage_area', 'mixed', 'unknown', 'exclude'];
const AREA_CLASS = { window_area: 'window', door_area: 'door', garage_area: 'garage_door' };
const CONTAINMENT = 0.70;

let progress = null, sheet = null, zoom = 1, mode = 'select';
let selection = new Set(), showRejected = false;
let undoStack = [], redoStack = [], saveTimer = null, dirty = false;
let enteredAt = Date.now();

/* ---------- geometry (points) ---------- */
const area = r => Math.max(0, r[2] - r[0]) * Math.max(0, r[3] - r[1]);
const inter = (a, b) => Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]))
                      * Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
const contain = (box, ar) => inter(box, ar) / Math.max(area(box), 1e-9);

/* Mirror of annotate.assign() in Python: smallest containing area wins, and a
 * human-set class is never overwritten. Kept in sync deliberately — the server
 * re-runs the same rule on save, so a divergence would show up immediately. */
function assign(op) {
  let best = null;
  for (const a of sheet.areas) {
    if (a.type === 'exclude') continue;
    const share = contain(op.rect, a.rect);
    if (share >= CONTAINMENT && (!best || area(a.rect) < area(best.a.rect))) best = { a, share };
  }
  if (!best) { op.area_id = ''; op.containment = 0; if (op.cls_source !== 'human') op.cls = ''; return; }
  op.area_id = best.a.id; op.containment = +best.share.toFixed(3);
  if (op.cls_source === 'human') return;
  if (AREA_CLASS[best.a.type]) { op.cls = AREA_CLASS[best.a.type]; op.cls_source = 'inherited'; }
  else { op.cls = ''; op.cls_source = 'required'; }
}
const reassign = () => sheet.openings.forEach(assign);

/* ---------- undo ---------- */
function snapshot() {
  if (!sheet) return;
  undoStack.push(JSON.stringify({ areas: sheet.areas, openings: sheet.openings }));
  if (undoStack.length > 100) undoStack.shift();
  redoStack.length = 0;
}
function restore(stack, other) {
  if (!stack.length) return;
  other.push(JSON.stringify({ areas: sheet.areas, openings: sheet.openings }));
  const s = JSON.parse(stack.pop());
  sheet.areas = s.areas; sheet.openings = s.openings;
  selection.clear(); reassign(); draw(); markDirty();
}

/* ---------- save ---------- */
function markDirty() { dirty = true; $('#dirty').className = 'dirty on'; scheduleSave(); }
function scheduleSave() { clearTimeout(saveTimer); saveTimer = setTimeout(save, 800); }
async function save() {
  if (!sheet || !dirty) return;
  $('#dirty').className = 'dirty saving';
  sheet.time_spent_s = (sheet.time_spent_s || 0) + Math.round((Date.now() - enteredAt) / 1000);
  enteredAt = Date.now();
  const res = await fetch(`/api/ann/${sheet.sheet_key}`, {
    method: 'PUT', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ base_rev: sheet.rev, sheet }),
  });
  if (res.status === 409) {
    banner('This sheet changed on disk (another tab?). Reload to continue.');
    return;
  }
  if (!res.ok) { banner('Save failed — your edits are NOT stored.'); return; }
  const out = await res.json();
  sheet.rev = out.rev; dirty = false;
  $('#dirty').className = 'dirty';
  $('#saved').textContent = 'saved ' + new Date().toLocaleTimeString();
  loadProgress();
}
const banner = m => { $('#banner').className = 'banner'; $('#banner').textContent = m; };
addEventListener('beforeunload', () => {
  if (dirty && sheet) navigator.sendBeacon(`/api/ann/${sheet.sheet_key}`,
    new Blob([JSON.stringify({ base_rev: sheet.rev, sheet })], { type: 'application/json' }));
});

/* ---------- sidebar ---------- */
async function loadProgress() {
  progress = await (await fetch('/api/progress')).json();
  const t = progress.totals;
  $('#prog').textContent = `${t.verified}/${t.sheets} verified · ${t.openings} openings · ${t.flagged} flagged`;
  $('#pbar').style.width = (100 * t.verified / Math.max(t.sheets, 1)) + '%';
  renderList();
}
function renderList() {
  const q = $('#q').value.trim().toLowerCase(), sort = $('#sort').value;
  let rows = progress.sheets.filter(r => !q || r.sheet_key.toLowerCase().includes(q));
  if (sort === 'unverified') rows = rows.filter(r => r.status !== 'verified');
  if (sort === 'hard') rows = [...rows].sort((a, b) => b.difficulty - a.difficulty);
  $('#list').innerHTML = rows.map(r => `
    <div class="row${sheet && sheet.sheet_key === r.sheet_key ? ' on' : ''}" data-k="${r.sheet_key}">
      <b>${r.project_id} · p${r.page}</b>
      <span class="pill ${r.status}">${r.status.replace('_', ' ')}</span>
      <span class="meta">${r.areas} areas · ${r.openings} openings${r.flagged ? ' · ' + r.flagged + ' flagged' : ''}${r.needs_review_areas ? ' · ' + r.needs_review_areas + ' need type' : ''}</span>
    </div>`).join('') || '<div style="padding:20px;color:var(--muted)">none</div>';
  $('#list').querySelectorAll('.row').forEach(el => el.onclick = () => open(el.dataset.k));
}
$('#q').oninput = renderList; $('#sort').onchange = renderList;

/* ---------- sheet ---------- */
async function open(key) {
  if (dirty) await save();
  const res = await fetch('/api/ann/' + key);
  if (!res.ok) { banner('Sheet not seeded'); return; }
  sheet = await res.json();
  selection.clear(); undoStack = []; redoStack = []; enteredAt = Date.now();
  $('#banner').className = ''; $('#banner').textContent = '';
  $('#wrap').innerHTML = `<div class="stage" id="stage" style="width:${sheet.render.w}px">
      <img src="${sheet.render.image}" alt="${key}">
    </div>`;
  reassign(); draw(); fit(); renderList();
}

const toPx = r => r.map(v => v * sheet.render.zoom);
const toPt = r => r.map(v => +(v / sheet.render.zoom).toFixed(2));

function draw() {
  const st = $('#stage'); if (!st) return;
  st.querySelectorAll('.ov,.h').forEach(e => e.remove());

  for (const a of sheet.areas) {
    const [x0, y0, x1, y1] = toPx(a.rect);
    const d = document.createElement('div');
    d.className = `ov area ${a.type}${selection.has(a.id) ? ' sel' : ''}`;
    d.style.cssText = `left:${x0}px;top:${y0}px;width:${x1 - x0}px;height:${y1 - y0}px`;
    d.dataset.id = a.id;
    d.innerHTML = `<span class="tag" style="background:var(--${a.type.replace('_area', '')})">${a.type.replace('_area', '')}</span>`;
    st.appendChild(d);
  }
  for (const o of sheet.openings) {
    if (o.state === 'rejected' && !showRejected) continue;
    const [x0, y0, x1, y1] = toPx(o.rect);
    const d = document.createElement('div');
    d.className = `ov op ${o.cls || 'none'} ${o.state}${selection.has(o.id) ? ' sel' : ''}`;
    d.style.cssText = `left:${x0}px;top:${y0}px;width:${x1 - x0}px;height:${y1 - y0}px`;
    d.dataset.id = o.id;
    st.appendChild(d);
  }
  if (selection.size === 1) handles([...selection][0]);
  renderCtx(); refreshSheetActions();
  const pend = sheet.openings.filter(o => o.state === 'pending').length;
  const need = sheet.areas.filter(a => a.type === 'unknown' || a.type === 'mixed').length;
  $('#sel').textContent = `${sheet.openings.filter(o => o.state === 'accepted').length} accepted`
    + (pend ? ` · ${pend} pending` : '') + (need ? ` · ${need} areas need a type` : '');
}

function find(id) { return sheet.areas.find(a => a.id === id) || sheet.openings.find(o => o.id === id); }

function handles(id) {
  const obj = find(id); if (!obj) return;
  const [x0, y0, x1, y1] = toPx(obj.rect), st = $('#stage');
  const pts = [[x0, y0, 'nw'], [(x0 + x1) / 2, y0, 'n'], [x1, y0, 'ne'], [x1, (y0 + y1) / 2, 'e'],
               [x1, y1, 'se'], [(x0 + x1) / 2, y1, 's'], [x0, y1, 'sw'], [x0, (y0 + y1) / 2, 'w']];
  for (const [x, y, k] of pts) {
    const h = document.createElement('div');
    h.className = 'h'; h.dataset.handle = k; h.dataset.id = id;
    h.style.cssText = `left:${x - 4.5}px;top:${y - 4.5}px;cursor:${k}-resize`;
    st.appendChild(h);
  }
}

/* ---------- pointer: select, move, resize, draw ---------- */
let drag = null;

function stagePoint(e) {
  const st = $('#stage'), r = st.getBoundingClientRect();
  return [(e.clientX - r.left) / zoom / sheet.render.zoom,
          (e.clientY - r.top) / zoom / sheet.render.zoom];   // points
}

document.addEventListener('pointerdown', e => {
  if (!sheet || !$('#stage') || !$('#stage').contains(e.target)) return;
  const [px, py] = stagePoint(e);

  if (e.target.dataset.handle) {
    snapshot();
    drag = { kind: 'resize', id: e.target.dataset.id, handle: e.target.dataset.handle,
             start: [px, py], orig: [...find(e.target.dataset.id).rect] };
  } else if (mode !== 'select') {
    snapshot();
    drag = { kind: 'draw', layer: mode, start: [px, py] };
  } else if (e.target.dataset.id) {
    const id = e.target.dataset.id;
    if (!e.shiftKey && !selection.has(id)) selection.clear();
    selection.add(id);
    snapshot();
    drag = { kind: 'move', start: [px, py],
             orig: new Map([...selection].map(i => [i, [...find(i).rect]])) };
    draw();
  } else {
    selection.clear(); draw();
  }
  if (drag) e.preventDefault();
});

document.addEventListener('pointermove', e => {
  if (!drag) return;
  const [px, py] = stagePoint(e);
  const dx = px - drag.start[0], dy = py - drag.start[1];

  if (drag.kind === 'move') {
    for (const [id, orig] of drag.orig) {
      const obj = find(id);
      obj.rect = [orig[0] + dx, orig[1] + dy, orig[2] + dx, orig[3] + dy];
      obj.edited = true;
    }
  } else if (drag.kind === 'resize') {
    const obj = find(drag.id), r = [...drag.orig], h = drag.handle;
    if (h.includes('n')) r[1] += dy;
    if (h.includes('s')) r[3] += dy;
    if (h.includes('w')) r[0] += dx;
    if (h.includes('e')) r[2] += dx;
    obj.rect = [Math.min(r[0], r[2]), Math.min(r[1], r[3]),
                Math.max(r[0], r[2]), Math.max(r[1], r[3])];
    obj.edited = true;
  } else if (drag.kind === 'draw') {
    drag.rect = [Math.min(drag.start[0], px), Math.min(drag.start[1], py),
                 Math.max(drag.start[0], px), Math.max(drag.start[1], py)];
    preview(drag.rect, drag.layer);
    return;
  }
  reassign(); draw();
});

document.addEventListener('pointerup', () => {
  if (!drag) return;
  if (drag.kind === 'draw' && drag.rect && area(drag.rect) > 4) {
    const id = `${drag.layer === 'area' ? 'a' : 'o'}_h${Date.now().toString(36)}`;
    if (drag.layer === 'area') {
      sheet.areas.push({ id, type: 'window_area', rect: drag.rect, origin: 'human',
                         edited: true, type_source: 'human', seed: {} });
    } else {
      const op = { id, rect: drag.rect, cls: '', cls_source: 'inherited', area_id: '',
                   containment: 0, origin: 'human', edited: true, state: 'accepted',
                   flags: [], confidence: 1.0, detector_source: 'human', group_id: '', seed: {} };
      assign(op); sheet.openings.push(op);
    }
    selection = new Set([id]);
    setMode('select');            // one-shot draw: avoids stray boxes while panning
  }
  document.querySelectorAll('.pv').forEach(e => e.remove());
  drag = null; reassign(); draw(); markDirty();
});

function preview(rect, layer) {
  document.querySelectorAll('.pv').forEach(e => e.remove());
  const [x0, y0, x1, y1] = toPx(rect), d = document.createElement('div');
  d.className = 'ov pv ' + (layer === 'area' ? 'area window_area' : 'op none');
  d.style.cssText = `left:${x0}px;top:${y0}px;width:${x1 - x0}px;height:${y1 - y0}px;opacity:.6`;
  $('#stage').appendChild(d);
}

/* ---------- actions ---------- */
function setMode(m) {
  mode = m;
  const map = { select: 'mSelect', area: 'bArea', opening: 'bOpen' };
  Object.entries(map).forEach(([k, id]) => {
    const el = $('#' + id);
    if (el) el.classList.toggle('on', k === m);
  });
}
function selectedAreas() { return sheet.areas.filter(a => selection.has(a.id)); }
function selectedOps() { return sheet.openings.filter(o => selection.has(o.id)); }

function setAreaType(type) {
  const picked = selectedAreas();
  if (!picked.length) return;
  snapshot();
  picked.forEach(a => { a.type = type; a.type_source = 'human'; });
  reassign(); draw(); markDirty();
}
function setClass(cls) {
  const picked = selectedOps();
  if (!picked.length) return;
  snapshot();
  picked.forEach(o => { o.cls = cls; o.cls_source = 'human'; o.state = 'accepted'; });
  draw(); markDirty();
}
function setState(state) {
  const picked = selectedOps();
  if (!picked.length) return;
  snapshot();
  picked.forEach(o => { o.state = state; });
  draw(); markDirty();
}
function acceptAllPending() {
  snapshot();
  sheet.openings.forEach(o => { if (o.state === 'pending' && o.cls) o.state = 'accepted'; });
  draw(); markDirty();
}
function rejectOutside() {
  snapshot();
  let n = 0;
  sheet.openings.forEach(o => { if (!o.area_id && o.state !== 'rejected') { o.state = 'rejected'; n++; } });
  draw(); markDirty();
  hint(`rejected ${n} box(es) that sit outside every area`);
}
function nudge(dx, dy, resize) {
  const picked = [...selection].map(find).filter(Boolean);
  if (!picked.length) return;
  snapshot();
  picked.forEach(o => {
    o.rect = resize ? [o.rect[0], o.rect[1], o.rect[2] + dx, o.rect[3] + dy]
                    : [o.rect[0] + dx, o.rect[1] + dy, o.rect[2] + dx, o.rect[3] + dy];
    o.edited = true;
  });
  reassign(); draw(); markDirty();
}

async function verify() {
  const unresolvedAreas = sheet.areas.filter(a => a.type === 'unknown' || a.type === 'mixed');
  const unclassed = sheet.openings.filter(o => o.state === 'accepted' && !o.cls);
  const pending = sheet.openings.filter(o => o.state === 'pending');
  if (unresolvedAreas.length || unclassed.length || pending.length) {
    // Refuse rather than record a verification that isn't one.
    hint(`cannot verify: ${unresolvedAreas.length} area(s) need a type, `
       + `${pending.length} pending, ${unclassed.length} accepted box(es) without a class`);
    return;
  }
  snapshot();
  sheet.status = 'verified';
  markDirty(); await save();
  const rows = progress.sheets.filter(r => r.status !== 'verified');
  if (rows.length) open(rows[0].sheet_key); else hint('all sheets verified');
}

function hint(msg) {
  const h = $('#hint'); h.hidden = false; h.textContent = msg;
  clearTimeout(h._t); h._t = setTimeout(() => { h.hidden = true; }, 4000);
}


/* ---------- contextual action panel ----------
 * Everything is a button. The panel shows only the actions that apply to the
 * current selection, so there is nothing to memorise: click a thing, see what
 * you can do to it.
 */
function renderCtx() {
  const ctx = $('#ctx');
  if (!sheet) { ctx.innerHTML = ''; return; }

  const areas = selectedAreas(), ops = selectedOps();

  if (!areas.length && !ops.length) {
    ctx.innerHTML = '<span class="tip">Click an <b>area</b> (big box) to set window/door · '
                  + 'click an <b>opening</b> (small box) to change or reject it</span>';
    return;
  }

  if (areas.length) {
    ctx.innerHTML = `<span class="lbl">${areas.length} area${areas.length > 1 ? 's' : ''} selected — this area contains:</span>`
      + `<button class="win" data-at="window_area">Windows</button>`
      + `<button class="dor" data-at="door_area">Doors</button>`
      + `<button class="gar" data-at="garage_area">Garage doors</button>`
      + `<button class="mix" data-at="mixed">Both (mixed)</button>`
      + `<button class="exc" data-at="exclude">Not a schedule — exclude</button>`
      + `<button class="del" data-del="area">Delete area</button>`;
    ctx.querySelectorAll('[data-at]').forEach(b =>
      b.onclick = () => setAreaType(b.dataset.at));
    ctx.querySelector('[data-del]').onclick = () => {
      snapshot();
      sheet.areas = sheet.areas.filter(a => !selection.has(a.id));
      selection.clear(); reassign(); draw(); markDirty(); renderCtx();
    };
    return;
  }

  const n = ops.length;
  ctx.innerHTML = `<span class="lbl">${n} opening${n > 1 ? 's' : ''} selected:</span>`
    + `<button class="good" data-st="accepted">Keep</button>`
    + `<button class="del" data-st="rejected">Not an opening — reject</button>`
    + `<span class="lbl" style="margin-left:10px">force class:</span>`
    + `<button class="win" data-cl="window">Window</button>`
    + `<button class="dor" data-cl="door">Door</button>`
    + `<button class="gar" data-cl="garage_door">Garage</button>`;
  ctx.querySelectorAll('[data-st]').forEach(b => b.onclick = () => setState(b.dataset.st));
  ctx.querySelectorAll('[data-cl]').forEach(b => b.onclick = () => setClass(b.dataset.cl));
}

/* Verify is disabled, with the reason on the button, rather than failing on click. */
function refreshSheetActions() {
  if (!sheet) return;
  const pending = sheet.openings.filter(o => o.state === 'pending').length;
  const outside = sheet.openings.filter(o => !o.area_id && o.state !== 'rejected').length;
  const needType = sheet.areas.filter(a => a.type === 'unknown' || a.type === 'mixed').length;
  const unclassed = sheet.openings.filter(o => o.state === 'accepted' && !o.cls).length;

  $('#bRejectOut').textContent = `Reject ${outside} box${outside === 1 ? '' : 'es'} outside areas`;
  $('#bRejectOut').disabled = outside === 0;
  $('#bAcceptAll').textContent = `Accept ${pending} pending`;
  $('#bAcceptAll').disabled = pending === 0;

  const blockers = [];
  if (needType) blockers.push(`${needType} area${needType > 1 ? 's need' : ' needs'} a type`);
  if (pending) blockers.push(`${pending} pending`);
  if (unclassed) blockers.push(`${unclassed} without a class`);
  const v = $('#bVerify');
  v.disabled = blockers.length > 0;
  v.textContent = blockers.length ? `Can't verify — ${blockers.join(', ')}` : 'Verify sheet ✓';
  $('#bShowRej').textContent = showRejected ? 'Hide rejected' : 'Show rejected';
  $('#bSkip').textContent = sheet.status === 'skipped' ? 'Un-skip sheet' : 'Skip sheet';
}


/* Skip keeps the sheet and its work but takes it out of the queue and the
 * export -- the right move for a near-duplicate. Delete is for a sheet that
 * should never have been picked up at all (an HVAC schedule, say); the file is
 * copied to .history first so it is recoverable. */
async function skipSheet() {
  if (!sheet) return;
  snapshot();
  sheet.status = sheet.status === 'skipped' ? 'in_progress' : 'skipped';
  markDirty(); await save();
  hint(sheet.status === 'skipped' ? 'sheet skipped — excluded from the dataset'
                                  : 'sheet un-skipped');
  step_sheet(1);
}

async function deleteSheet() {
  if (!sheet) return;
  const key = sheet.sheet_key;
  if (!confirm(`Delete ${key}?\n\nIt is removed from the dataset. A copy is kept in `
             + `annotations/.history/ so it can be restored.`)) return;
  const res = await fetch('/api/ann/' + key, { method: 'DELETE' });
  if (!res.ok) { banner('Delete failed'); return; }
  dirty = false;
  await loadProgress();
  const next = progress.sheets[0];
  if (next) open(next.sheet_key); else $('#wrap').innerHTML = '<div style="padding:40px">No sheets left.</div>';
}

/* ---------- zoom ---------- */
function setZoom(z) {
  zoom = Math.min(6, Math.max(0.05, z));
  const st = $('#stage'); if (!st) return;
  st.style.transform = `scale(${zoom})`;
  st.parentElement.style.height = (sheet.render.h * zoom) + 'px';
  st.parentElement.style.width = (sheet.render.w * zoom) + 'px';
}
function fit() { if (sheet) setZoom(($('main').clientWidth - 30) / sheet.render.w); }

/* ---------- keys ---------- */
addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  // Shortcuts are an optional accelerator on top of the buttons -- they must
  // not fire while the Library tab is showing.
  const view = document.querySelector('#annview');
  if (view && view.hidden) return;
  const k = e.key;
  if ((e.ctrlKey || e.metaKey) && k.toLowerCase() === 'z') {
    e.preventDefault();
    return e.shiftKey ? restore(redoStack, undoStack) : restore(undoStack, redoStack);
  }
  if ((e.ctrlKey || e.metaKey) && k.toLowerCase() === 's') { e.preventDefault(); return save(); }
  if (!sheet) return;
  const step = e.shiftKey ? 10 : 1;
  const map = {
    a: () => setMode('area'), o: () => setMode('opening'), Escape: () => setMode('select'),
    1: () => setAreaType('window_area'), 2: () => setAreaType('door_area'),
    3: () => setAreaType('garage_area'), 4: () => setAreaType('mixed'), 0: () => setAreaType('exclude'),
    w: () => setClass('window'), e: () => setClass('door'), g: () => setClass('garage_door'),
    Enter: () => setState('accepted'), Delete: () => setState('rejected'),
    Backspace: () => setState('rejected'),
    A: acceptAllPending, X: rejectOutside, v: verify,
    p: () => { showRejected = !showRejected; draw(); },
    f: fit, '+': () => setZoom(zoom * 1.25), '-': () => setZoom(zoom / 1.25),
    '?': () => $('#dlg').showModal(),
    j: () => step_sheet(1), k: () => step_sheet(-1),
    ArrowLeft: () => nudge(-step, 0, e.altKey), ArrowRight: () => nudge(step, 0, e.altKey),
    ArrowUp: () => nudge(0, -step, e.altKey), ArrowDown: () => nudge(0, step, e.altKey),
  };
  if (map[k]) { e.preventDefault(); map[k](); }
});

function step_sheet(delta) {
  const rows = progress.sheets;
  const i = rows.findIndex(r => r.sheet_key === sheet.sheet_key);
  const next = rows[Math.min(rows.length - 1, Math.max(0, i + delta))];
  if (next) open(next.sheet_key);
}

$('#bArea').onclick = () => setMode('area');
$('#bOpen').onclick = () => setMode('opening');
$('#mSelect').onclick = () => setMode('select');
$('#bVerify').onclick = verify;
$('#bRejectOut').onclick = rejectOutside;
$('#bAcceptAll').onclick = acceptAllPending;
$('#bUndo').onclick = () => restore(undoStack, redoStack);
$('#bShowRej').onclick = () => { showRejected = !showRejected; draw(); };
$('#bPrev').onclick = () => step_sheet(-1);
$('#bNext').onclick = () => step_sheet(1);
$('#bSkip').onclick = skipSheet;
$('#bDelete').onclick = deleteSheet;
$('#zin').onclick = () => setZoom(zoom * 1.25);
$('#zout').onclick = () => setZoom(zoom / 1.25);
$('#zfit').onclick = fit;
$('#help').onclick = () => $('#dlg').showModal();
addEventListener('resize', fit);

export async function boot(sheetKey) {
  await loadProgress();
  if (sheetKey) return open(sheetKey);
  if (progress.sheets.length) return open(progress.sheets[0].sheet_key);
}
export { open, loadProgress };
