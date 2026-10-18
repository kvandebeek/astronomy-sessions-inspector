// The review page's client. The server embeds the dataset as window.REVIEW_DATA before
// this script; see build_page() in fits_review_server.py.
/* global REVIEW_DATA, rowTotalKey, setRowTotal */
const FILES = REVIEW_DATA.files;
const FOLDERS = REVIEW_DATA.folders;
// One entry per (folder, exposure length). Everything derived below aggregates on these
// rather than on folders: a folder holding 30 s and 45 s subs is two incomparable sets.
const UNITS = REVIEW_DATA.units;
const GROUPS = REVIEW_DATA.groups;
const MATRIX_METRICS = REVIEW_DATA.matrixMetrics;
const FILTER_COLORS = REVIEW_DATA.filterColors;
const DEFAULT_FILTER_COLOR = REVIEW_DATA.defaultFilterColor;
const DASH = '\u2014';

function unitKey(panel, session, filt, exposure) {
  return [panel, session, filt, exposure].join('\u0000');
}

function filterColor(filt) {
  return FILTER_COLORS[filt] || DEFAULT_FILTER_COLOR;
}

const VERDICT_COLORS = {good: 'var(--good)', warn: '#d9a441', reject: 'var(--bad)'};

function verdictColor(f) {
  return VERDICT_COLORS[f.quality_verdict] || filterColor(f.filter);
}

let curId = null;
let curStretch = 'auto';
// Raw or calibrated, shared by the preview and the thumbnail grid so the two never
// disagree about which pixels are on screen. Only offered for frames a scan run with
// --save-calibrated actually wrote a calibrated copy for.
let curSource = 'raw';
let matrixMetric = MATRIX_METRICS[0][0];
let clipMetric = 'signal_adu_per_sec';
let clipSigma = 0;
const CLIP_SLIDER_MAX = 100;
const CLIP_SIGMA_MAX = 5.0;
const CLIP_SIGMA_MIN = 0.1;

// Looked up once. These are read on every slider tick, and querying the DOM per
// frame per tick is what makes the sigma slider feel sticky on a large dataset.
const ROW_BY_ID = new Map();
const BOX_BY_ID = new Map();
const DEV_BY_ID = new Map();
// One summary <div> can now cover several folders (all sessions of one panel+filter share a
// table), so it is indexed by group rather than by a single folder id.
const SUMMARY_GROUPS = [];
const UNIT_BY_KEY = new Map();
const MATRIX_CELLS = [];
const MATRIX_TOTALS = [];
// unit id -> {agg, included, live, active}, rebuilt once per view update. The matrix cells
// and the per-section summary lines both read this instead of each re-aggregating every unit.
const VIEW_BY_UNIT = new Map();
let clippedTotal = 0;

function fmtNum(v, d) {
  if (v === null || v === undefined || Number.isNaN(v)) return DASH;
  return Number(v).toLocaleString(undefined, {minimumFractionDigits:d, maximumFractionDigits:d});
}

function fmtPct(v, d) {
  if (v === null || v === undefined || Number.isNaN(v)) return DASH;
  return (v * 100).toLocaleString(undefined, {minimumFractionDigits:d, maximumFractionDigits:d}) + '%';
}

function fmtMetric(v, digits, pct) {
  return pct ? fmtPct(v, digits) : fmtNum(v, digits);
}

function numeric(v) {
  return typeof v === 'number' && Number.isFinite(v);
}

function mean(values) {
  return values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;
}

function median(values) {
  if (!values.length) return null;
  const sorted = values.slice().sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function stddev(values) {
  if (values.length < 2) return 0;
  const m = mean(values);
  return Math.sqrt(values.reduce((acc, v) => acc + (v - m) * (v - m), 0) / values.length);
}

function aggregate(files) {
  // One pass per metric rather than map+filter chains: this runs for every matrix
  // cell on every slider tick.
  const exposures = [], hfrs = [], fwhms = [], roundnesses = [], starCounts = [],
        adus = [], signals = [], rates = [], uniformRs = [], radialRs = [], qualities = [];
  for (const f of files) {
    if (numeric(f.exposure)) exposures.push(f.exposure);
    if (numeric(f.hfr_avg)) hfrs.push(f.hfr_avg);
    if (numeric(f.fwhm_avg)) fwhms.push(f.fwhm_avg);
    if (numeric(f.roundness_avg)) roundnesses.push(f.roundness_avg);
    if (numeric(f.num_stars)) starCounts.push(f.num_stars);
    if (numeric(f.adu_mean)) adus.push(f.adu_mean);
    if (numeric(f.signal_adu)) signals.push(f.signal_adu);
    if (numeric(f.signal_adu_per_sec)) rates.push(f.signal_adu_per_sec);
    if (numeric(f.elong_uniform_r)) uniformRs.push(f.elong_uniform_r);
    if (numeric(f.elong_radial_r)) radialRs.push(f.elong_radial_r);
    if (numeric(f.quality_score)) qualities.push(f.quality_score);
  }
  return {
    frames: files.length,
    total_exp_min: exposures.length ? exposures.reduce((a, b) => a + b, 0) / 60 : null,
    hfr: median(hfrs),
    fwhm: median(fwhms),
    roundness: median(roundnesses),
    stars: mean(starCounts),
    adu_mean: mean(adus),
    signal_adu: mean(signals),
    signal_adu_per_sec: mean(rates),
    adu_spread: stddev(adus),
    elong_uniform_r: mean(uniformRs),
    elong_radial_r: mean(radialRs),
    quality: mean(qualities),
  };
}

function robustSigma(values, center) {
  // Median absolute deviation, rescaled to the equivalent Gaussian sigma. A plain
  // standard deviation is computed *including* the outliers it is being used to find,
  // so in a short folder a single bad frame inflates sigma enough to hide itself.
  const mad = median(values.map(v => Math.abs(v - center)));
  return mad > 0 ? mad * 1.4826 : stddev(values);
}

function recomputeSigmaClipping() {
  for (const f of FILES) f.sigmaClipped = false;
  if (clipSigma <= 0) return;
  // Per unit, not per folder: clipping ADU or signal across two exposure lengths would
  // measure the difference between the lengths rather than the spread within either.
  for (const folder of UNITS) {
    const candidates = [], values = [];
    for (const id of folder.file_ids) {
      const f = FILES[id];
      if (f.deleted || !numeric(f[clipMetric])) continue;
      candidates.push(f);
      values.push(f[clipMetric]);
    }
    if (candidates.length < 2) continue;
    const center = median(values);
    const sigma = robustSigma(values, center);
    if (!numeric(center) || !(sigma > 0)) continue;
    const cutoff = clipSigma * sigma;
    for (const f of candidates) f.sigmaClipped = Math.abs(f[clipMetric] - center) > cutoff;
  }
}

function recomputeFolderViews() {
  VIEW_BY_UNIT.clear();
  clippedTotal = 0;
  for (const unit of UNITS) {
    const active = [];
    let live = 0;
    for (const id of unit.file_ids) {
      const f = FILES[id];
      if (f.deleted) continue;
      live++;
      if (!f.sigmaClipped) active.push(f);
    }
    clippedTotal += live - active.length;
    VIEW_BY_UNIT.set(unit.id, {agg: aggregate(active), included: active.length, live, active});
  }
}

function renderMatrix() {
  const [, , digits, pct] = MATRIX_METRICS.find(m => m[0] === matrixMetric) || MATRIX_METRICS[0];
  for (const entry of MATRIX_CELLS) {
    const view = entry.unit && VIEW_BY_UNIT.get(entry.unit.id);
    const text = view ? fmtMetric(view.agg[matrixMetric], digits, pct) : DASH;
    if (entry.text === text) continue;   // a slider tick moves only a handful of cells
    entry.text = text;
    entry.span.textContent = text;
    entry.span.className = view ? 'cell-val' : 'empty';
  }
  renderMatrixTotals();
}

function renderMatrixTotals() {
  // Built from the same VIEW_BY_UNIT the cells above read, so a deleted or sigma-clipped
  // frame drops out of the row's frame count and its exposure out of the row's total.
  for (const entry of MATRIX_TOTALS) {
    let frames = 0, minutes = 0, any = false;
    for (const unit of entry.units) {
      const view = VIEW_BY_UNIT.get(unit.id);
      if (!view) continue;
      any = true;
      frames += view.agg.frames;
      if (numeric(view.agg.total_exp_min)) minutes += view.agg.total_exp_min;
    }
    // Compared on a key, since the cell now holds two slotted spans rather than a string.
    const key = rowTotalKey(entry.kind, frames, minutes, any);
    if (entry.text === key) continue;
    entry.text = key;
    setRowTotal(entry.cell, entry.kind, frames, minutes, any);
  }
}

function setMatrixMetric(key) {
  matrixMetric = key;
  document.querySelectorAll('.metric-btn').forEach(b => b.classList.toggle('active', b.dataset.metric === key));
  renderMatrix();
}

function renderFolderSummaries() {
  for (const group of SUMMARY_GROUPS) {
    let included = 0, live = 0;
    // Only the two means below are shown, so gather just those instead of calling
    // aggregate(): the full roll-up also medians HFR, FWHM and roundness, and those
    // three sorts ran for every group on every slider tick. Same values, same order,
    // same mean() - just none of the eleven metrics nothing reads.
    const adus = [], rates = [];
    for (const id of group.ids) {
      const view = VIEW_BY_UNIT.get(id);
      if (!view) continue;
      included += view.included;
      live += view.live;
      for (const f of view.active) {
        if (numeric(f.adu_mean)) adus.push(f.adu_mean);
        if (numeric(f.signal_adu_per_sec)) rates.push(f.signal_adu_per_sec);
      }
    }
    const suffix = group.ids.length > 1 ? ' across ' + group.ids.length + ' session(s)' : '';
    const html = included + '/' + live + ' included' + suffix +
      ' &nbsp;&middot;&nbsp; mean ADU: ' + fmtNum(mean(adus), 1) +
      ' &nbsp;&middot;&nbsp; signal ADU/s: ' + fmtNum(mean(rates), 2);
    if (group.el._html === html) continue;   // most groups are untouched by a given tick
    group.el._html = html;
    group.el.innerHTML = html;
  }
  const total = document.getElementById('clip-total');
  if (total) total.textContent = clippedTotal + ' clipped';
}

function renderRows() {
  for (const f of FILES) {
    const row = ROW_BY_ID.get(f.id);
    if (!row) continue;
    const deleted = !!f.deleted, clipped = !!f.sigmaClipped;
    // Only touch rows whose state actually changed. A slider tick moves a handful of
    // frames, and re-toggling the class on all ~600 rows invalidates style for all of them.
    if (row._deleted !== deleted) {
      row._deleted = deleted;
      row.classList.toggle('deleted', deleted);
    }
    const missing = !!f.missing;
    if (row._missing !== missing) {
      // Deleted frames stay selectable - that is how several are restored at once - but a
      // purged one is gone for good and has nothing left to act on.
      row._missing = missing;
      row.classList.toggle('missing', missing);
      const box = BOX_BY_ID.get(f.id);
      if (box) {
        box.disabled = missing;
        // Deferred: renderRows() refreshes the buttons once when it finishes.
        if (missing && box.checked) setSelected(f.id, false, true);
      }
    }
    if (row._clipped !== clipped) {
      row._clipped = clipped;
      row.classList.toggle('sigma-clipped', clipped);
    }
    updateDeviation(f);
  }
  updateDeleteButtons();
}

function updateDeviation(f) {
  // The scan wrote this against every frame the set had then. Deleting or clipping frames
  // moves the set's mean, so the column has to follow or it keeps reporting a spread
  // around frames that are no longer in the set.
  const cell = DEV_BY_ID.get(f.id);
  if (!cell) return;
  const view = VIEW_BY_UNIT.get(f.unit_id);
  const mean = view && view.agg ? view.agg.adu_mean : null;
  const dev = numeric(mean) && numeric(f.adu_mean) ? f.adu_mean - mean : null;
  const text = dev === null ? DASH
    : (dev > 0 ? '+' : '') + dev.toLocaleString(undefined, {minimumFractionDigits: 1, maximumFractionDigits: 1});
  if (cell._text === text) return;
  cell._text = text;
  cell.textContent = text;
  cell.className = 'dev-cell' + (dev === null || dev === 0 ? '' : (dev > 0 ? ' dev-pos' : ' dev-neg'));
}

function updateClipControls() {
  const value = document.getElementById('clip-value');
  if (value) value.textContent = clipSigma <= 0 ? 'Off' : clipSigma.toFixed(2).replace(/\.00$/, '.0') + '\u03c3';
}

function updateDerivedView() {
  recomputeSigmaClipping();
  recomputeFolderViews();
  renderRows();
  renderFolderSummaries();
  renderMatrix();
  updateClipControls();
  // The grid marks clipped frames, so it has to follow the slider like everything else.
  if (document.getElementById('grid-overlay').classList.contains('open')) refreshGridClipMarks();
}

function refreshGridClipMarks() {
  let clipped = 0;
  for (const [id, item] of GRID_ITEM_BY_ID) {
    const isClipped = !!FILES[id].sigmaClipped;
    if (isClipped) clipped++;
    if (item._clipped === isClipped) continue;
    item._clipped = isClipped;
    item.classList.toggle('clipped', isClipped);
    const wrap = item.querySelector('.thumb-wrap');
    const existing = wrap.querySelector('.clip-badge');
    if (isClipped && !existing) {
      const badge = document.createElement('span');
      badge.className = 'clip-badge';
      badge.title = 'Excluded by the sigma-clipping slider; not counted in this set\u2019s statistics';
      badge.textContent = 'clipped';
      wrap.append(badge);
    } else if (!isClipped && existing) {
      existing.remove();
    }
  }
  setGridCount(GRID_ITEM_BY_ID.size, clipped);
}

let pendingView = 0;
function scheduleDerivedView() {
  // The range input fires ~1000 events across its travel. Collapse them onto one frame
  // so the work happens at most once per repaint.
  if (pendingView) return;
  pendingView = requestAnimationFrame(() => { pendingView = 0; updateDerivedView(); });
}

function sigmaFromSliderPosition(value) {
  const position = Math.max(0, Math.min(CLIP_SLIDER_MAX, Number(value)));
  if (position <= 0) return 0;
  // Clamped at 0: the step is 0.1, so positions between "Off" and the first labelled
  // mark are reachable and a negative t put sigma *above* CLIP_SIGMA_MAX, off the scale
  // the datalist labels describe.
  const t = Math.max(0, (position - 1) / (CLIP_SLIDER_MAX - 1));
  return CLIP_SIGMA_MAX - t * (CLIP_SIGMA_MAX - CLIP_SIGMA_MIN);
}

function setClipSigma(value) {
  const next = sigmaFromSliderPosition(value);
  if (next === clipSigma) return;   // the slider fires ~1000 input events across its range
  clipSigma = next;
  updateClipControls();   // cheap, and keeps the readout exact under the cursor
  scheduleDerivedView();
}

function setClipMetric(value) {
  clipMetric = value;
  updateDerivedView();
}

// The ids the preview's arrows walk: the table rows or grid tiles the frame was opened
// from, in the order they are on screen. Arrowing used to walk every frame of the whole
// dataset in scan order, whatever the table had been sorted by.
let navIds = null;

function visibleRowIds(tbody) {
  const ids = [];
  for (const row of tbody.rows) {
    if (row.dataset.id === undefined || row.offsetParent === null) continue;   // hidden
    ids.push(Number(row.dataset.id));
  }
  return ids;
}

function openModal(id, context) {
  if (context) navIds = context;
  const wasOpen = document.getElementById('modal-overlay').classList.contains('open');
  const changed = curId !== id;
  curId = id;
  if (!wasOpen) { curStretch = 'auto'; zoom100 = false; }
  // Handles are in ADU, and a different frame sits at a different sky level, so a manual
  // adjustment does not carry over - the preset re-fits for the new frame. Zoom does
  // carry over, because a pixel window means the same thing on every frame.
  if (changed) { levels = null; levelsAuto = true; histogram = null; histogramId = null; }
  // Zoom deliberately survives arrowing between frames: holding the same 1:1 window over
  // the same patch of sky is how you compare star shapes frame to frame. loadImage()
  // re-clamps it to each frame's dimensions.
  document.getElementById('modal-overlay').classList.add('open');
  renderModal();
}

function closeModal() {
  document.getElementById('modal-overlay').classList.remove('open');
  curId = null;
}

function setText(id, value) {
  document.getElementById(id).textContent = value;
}

function statSpan(label, value) {
  // Values come from FITS headers, so build the node instead of splicing into innerHTML.
  const span = document.createElement('span');
  const b = document.createElement('b');
  b.textContent = label;
  span.append(b, ' ' + value);
  return span;
}

function elongationSummary(f) {
  // Mirrors fmt_elongation() in fits_light_stats.py: whichever direction hypothesis the
  // frame's elongated stars agree on more strongly - same axis everywhere (tracking/PE)
  // or pointing away from frame center (coma/tilt) - or a dash if too few to tell.
  if (!numeric(f.elong_uniform_r) || !numeric(f.elong_radial_r)) return DASH;
  if (f.elong_uniform_r >= f.elong_radial_r) {
    return 'tracking-like, r=' + f.elong_uniform_r.toFixed(2) + ' @ ' + fmtNum(f.elong_uniform_angle_deg, 0) + '\u00b0';
  }
  return 'radial (coma-like), r=' + f.elong_radial_r.toFixed(2);
}

function renderModal() {
  const f = FILES[curId];
  setText('modal-title', f.name);
  const headline = document.getElementById('modal-headline');
  // Where this frame sits in the set. The filename carries the date and the settings but
  // not which panel or night it belongs to, and that is what you are comparing against
  // when arrowing through a group.
  const folder = FOLDERS[f.folder_id];
  headline.replaceChildren(
    statSpan('Panel', (folder && folder.panel) || DASH),
    statSpan('Session', (folder && folder.session) || DASH),
    statSpan('Filter', f.filter || DASH),
    statSpan('Exposure', numeric(f.exposure) ? f.exposure + 's' : DASH),
    statSpan('Calibration', f.calibration || 'none'),
  );
  const box = document.getElementById('modal-stats');
  const qualityLabel = numeric(f.quality_score)
    ? Math.round(f.quality_score) + ' (' + (f.quality_verdict || DASH) + ')' +
      (f.quality_reasons && f.quality_reasons.length ? ' \u2014 ' + f.quality_reasons.join(', ') : '')
    : DASH;
  box.replaceChildren(
    statSpan('Quality', qualityLabel),
    statSpan('ADU', fmtNum(f.adu_min, 0) + ' \u2013 ' + fmtNum(f.adu_max, 0) + ' (mean ' + fmtNum(f.adu_mean, 1) + ')'),
    statSpan('Signal', fmtNum(f.signal_adu, 1) + ' ADU'),
    statSpan('Signal rate', fmtNum(f.signal_adu_per_sec, 2) + ' ADU/s'),
    statSpan('HFR', fmtNum(f.hfr_avg, 2)),
    statSpan('FWHM', fmtNum(f.fwhm_avg, 2) + ' px' +
      (numeric(f.fwhm_arcsec) ? ' (' + fmtNum(f.fwhm_arcsec, 2) + '\u2033)' : '')),
    statSpan('Stars', String(f.num_stars ?? 0)),
    statSpan('Roundness', fmtPct(f.roundness_avg, 1)),
    statSpan('Elongation dir', elongationSummary(f)),
  );

  // Masters the scan skipped for this frame (another gain, binning...), or a large
  // temperature gap: the calibration column only flags them, this says what they are.
  const warnBox = document.getElementById('modal-warnings');
  const warnings = f.calibration_warnings || [];
  warnBox.hidden = !warnings.length;
  warnBox.textContent = warnings.length ? 'Calibration: ' + warnings.join('; ') : '';

  markStretchButtons('', curStretch);

  const trailBtn = document.getElementById('trail-btn');
  trailBtn.hidden = !f.has_trail;
  trailBtn.classList.toggle('active', showTrail);

  document.getElementById('zoom-btn').classList.toggle('active', zoom100);

  const hasCalibrated = !!f.has_calibrated;
  for (const id of ['raw-btn', 'calibrated-btn']) {
    document.getElementById(id).hidden = !hasCalibrated;
  }
  markSourceButtons();

  const delBtn = document.getElementById('delete-btn');
  delBtn.textContent = f.missing ? 'File missing' : (f.deleted ? 'Restore' : 'Delete');
  delBtn.disabled = !!f.missing;
  delBtn.title = f.missing
    ? 'This frame is at neither its original path nor in _deleted, so it cannot be restored.'
    : '';
  delBtn.classList.toggle('danger', !f.deleted && !f.missing);

  loadImage();
  loadHistogram();
}

// --- levels ---------------------------------------------------------------------------
// black / midtone / white, the three handles on the histogram. The midtone is the input
// level between black and white that comes out mid grey, so its handle position between
// the other two IS its value - dragging it is dragging the curve.
let levels = null;          // {black, white, mid} in ADU / [0,1], or null to follow the preset
let levelsAuto = true;      // true while the preset is placing them for each frame
let histogram = null;       // the payload for the frame on screen

function levelsFor(f) {
  if (levels) return levels;
  if (histogram && histogram.presets && histogram.presets[curStretch]) return histogram.presets[curStretch];
  return null;
}

function levelsQuery() {
  const lv = levelsFor();
  if (!lv) return '';
  return '&black=' + lv.black.toFixed(4) + '&white=' + lv.white.toFixed(4) + '&mid=' + lv.mid.toFixed(5);
}

function sourceQuery() {
  return curSource === 'calibrated' ? '&source=calibrated' : '';
}

function setSource(source) {
  if (source === curSource) return;
  curSource = source;
  markSourceButtons();
  // Dark subtraction removes the pedestal, so the two sources sit at completely different
  // ADU levels - a background near 2900 becomes one near 120. Levels in ADU cannot carry
  // across, and the histogram is a different distribution, so both are refitted.
  levels = null;
  levelsAuto = true;
  histogram = null;
  histogramId = null;
  if (curId !== null) renderModal();
  if (document.getElementById('grid-overlay').classList.contains('open')) renderGroupGrid(true);
}

function markSourceButtons() {
  for (const [id, value] of [['raw-btn', 'raw'], ['calibrated-btn', 'calibrated'],
                             ['grid-raw-btn', 'raw'], ['grid-calibrated-btn', 'calibrated']]) {
    const btn = document.getElementById(id);
    if (btn) btn.classList.toggle('active', curSource === value);
  }
}

function fmtAdu(v) {
  return Number(v).toLocaleString(undefined, {maximumFractionDigits: 0});
}

function renderLevelsRow() {
  const box = document.getElementById('modal-levels');
  const lv = levelsFor();
  if (!lv) { box.replaceChildren(); return; }
  const item = (cls, label, value) => {
    const span = document.createElement('span');
    span.className = cls;
    const b = document.createElement('b');
    b.textContent = value;
    span.append(label + ' ', b);
    return span;
  };
  const offPlot = histogram && lv.white > histogram.max;
  const nodes = [
    item('lv-black', 'Black', fmtAdu(lv.black)),
    // The midtone handle's own value is a fraction, but what it means is "this input level
    // comes out mid grey", so the readout gives that level in ADU as well.
    item('lv-mid', 'Midtone', fmtAdu(lv.black + lv.mid * (lv.white - lv.black)) +
                              ' (' + lv.mid.toFixed(4) + ')'),
    item('lv-white', 'White', fmtAdu(lv.white) + (offPlot ? ' \u00b7 full scale' : '')),
  ];
  const spacer = document.createElement('span');
  spacer.className = 'spacer';
  nodes.push(spacer);
  const note = document.createElement('span');
  note.textContent = levelsAuto ? 'following ' + curStretch : 'adjusted \u2014 drag the handles';
  nodes.push(note);
  // Always present, disabled while a preset is in charge. Adding and removing it made this
  // row 5px taller and shorter again, and since the image area is what absorbs the
  // difference, the preview resized under the cursor every time levels were touched -
  // which also left a 1:1 crop no longer matching its viewport.
  const reset = document.createElement('button');
  reset.textContent = 'Reset to ' + curStretch;
  reset.disabled = levelsAuto;
  reset.addEventListener('click', () => { levels = null; levelsAuto = true; applyLevels(); });
  nodes.push(reset);
  box.replaceChildren(...nodes);
}

function histX(value, w) {
  return (value - histogram.min) / (histogram.max - histogram.min) * w;
}

function histValue(px, w) {
  return histogram.min + (px / w) * (histogram.max - histogram.min);
}

function drawHistogram() {
  const canvas = document.getElementById('modal-histogram');
  const ctx = canvas.getContext('2d');
  const w = canvas.clientWidth || canvas.width, h = canvas.clientHeight || canvas.height;
  canvas.width = w; canvas.height = h;
  const style = getComputedStyle(document.body);
  ctx.clearRect(0, 0, w, h);
  if (!histogram || !histogram.counts || !histogram.counts.length) return;
  const counts = histogram.counts;
  const plotH = h - 16;   // leave a strip at the bottom for the axis labels
  // Log scale: a handful of saturated/background bins otherwise dwarf everything else.
  const logCounts = counts.map(c => Math.log1p(c));
  let maxLog = 1e-9;
  for (const v of logCounts) if (v > maxLog) maxLog = v;
  const barW = w / counts.length;
  ctx.fillStyle = style.getPropertyValue('--accent') || '#4a9eff';
  for (let i = 0; i < counts.length; i++) {
    const barH = (logCounts[i] / maxLog) * (plotH - 2);
    ctx.fillRect(i * barW, plotH - barH, Math.max(1, barW - 0.5), barH);
  }

  // ADU axis. Without it there is no way to tell where the sensor's offset pedestal sits,
  // which is the first thing the shape of this histogram is read for.
  ctx.font = '9px -apple-system, Segoe UI, Helvetica, Arial, sans-serif';
  ctx.fillStyle = style.getPropertyValue('--muted') || '#8a93a6';
  ctx.textBaseline = 'top';
  for (let i = 0; i <= 4; i++) {
    const value = histogram.min + (histogram.max - histogram.min) * (i / 4);
    const x = (i / 4) * w;
    ctx.textAlign = i === 0 ? 'left' : (i === 4 ? 'right' : 'center');
    ctx.fillText(fmtAdu(value), Math.min(w - 1, Math.max(1, x)), plotH + 3);
  }

  const lv = levelsFor();
  if (!lv) return;
  const handles = [
    ['black', lv.black, style.getPropertyValue('--muted') || '#8a93a6'],
    ['mid', lv.black + lv.mid * (lv.white - lv.black), '#d9a441'],
    ['white', lv.white, style.getPropertyValue('--accent') || '#4a9eff'],
  ];
  for (const [name, value, color] of handles) {
    const raw = histX(value, w);
    const offPlot = raw > w + 0.5 || raw < -0.5;
    const x = Math.max(2, Math.min(w - 2, raw));
    ctx.strokeStyle = color;
    ctx.lineWidth = name === 'mid' ? 1 : 1.5;
    // A white point parked at the sensor's full scale sits far past the plotted range.
    // Drawn pinned at the edge and hollow, so it reads as "beyond here" rather than as a
    // white point that happens to sit on the last bin.
    ctx.setLineDash(name === 'mid' ? [3, 2] : (offPlot ? [2, 3] : []));
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, plotH);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(x, plotH);
    ctx.lineTo(x - 4, plotH + 6);
    ctx.lineTo(x + 4, plotH + 6);
    ctx.closePath();
    if (offPlot) { ctx.strokeStyle = color; ctx.lineWidth = 1; ctx.stroke(); }
    else { ctx.fillStyle = color; ctx.fill(); }
  }
}

let levelsRequest = 0, levelsPending = false;

function applyLevels() {
  drawHistogram();
  renderLevelsRow();
  // One render in flight at a time: dragging a handle fires continuously, and queuing a
  // request per pixel would leave the preview lagging seconds behind the cursor.
  if (levelsPending) { levelsRequest++; return; }
  levelsPending = true;
  const issued = ++levelsRequest;
  loadImage().finally(() => {
    levelsPending = false;
    if (levelsRequest !== issued) applyLevels();
  });
}

// Drag the three handles on the histogram.
(function () {
  const canvas = document.getElementById('modal-histogram');
  let holding = null;

  function nearest(px, w) {
    const lv = levelsFor();
    if (!lv) return null;
    const pts = {
      black: histX(lv.black, w),
      mid: histX(lv.black + lv.mid * (lv.white - lv.black), w),
      white: histX(lv.white, w),
    };
    let best = null, bestDist = Infinity;
    for (const name of ['black', 'mid', 'white']) {
      const d = Math.abs(Math.max(2, Math.min(w - 2, pts[name])) - px);
      if (d < bestDist) { bestDist = d; best = name; }
    }
    return best;
  }

  function moveHandle(name, px, w) {
    const lv = Object.assign({}, levelsFor());
    const value = histValue(px, w);
    const span = Math.max(1e-6, histogram.max - histogram.min);
    const minGap = span * 0.005;
    if (name === 'black') {
      lv.black = Math.min(value, lv.white - minGap);
    } else if (name === 'white') {
      lv.white = Math.max(value, lv.black + minGap);
    } else {
      const t = (value - lv.black) / Math.max(1e-9, lv.white - lv.black);
      lv.mid = Math.min(0.999, Math.max(0.001, t));
    }
    levels = lv;
    levelsAuto = false;
    applyLevels();
  }

  canvas.addEventListener('mousedown', (e) => {
    if (!histogram || !levelsFor()) return;
    const rect = canvas.getBoundingClientRect();
    holding = nearest(e.clientX - rect.left, rect.width);
    if (holding) { moveHandle(holding, e.clientX - rect.left, rect.width); e.preventDefault(); }
  });

  window.addEventListener('mousemove', (e) => {
    if (!holding) return;
    const rect = canvas.getBoundingClientRect();
    moveHandle(holding, Math.max(0, Math.min(rect.width, e.clientX - rect.left)), rect.width);
  });

  window.addEventListener('mouseup', () => { holding = null; });
})();

let histogramId = null;

async function loadHistogram() {
  const requestedId = curId;
  // The histogram is a property of the frame, not of the stretch or the zoom, so it is
  // fetched once per frame. Re-fetching on every renderModal() also blanked it first,
  // which raced loadImage(): the render went out with no levels and a second one followed
  // as soon as the payload landed, so every zoom or preset change cost two renders.
  if (histogramId === requestedId && histogram) return;
  histogramId = requestedId;
  histogram = null;
  drawHistogram();
  try {
    const res = await fetch('/api/histogram?id=' + requestedId + sourceQuery());
    if (curId !== requestedId || !res.ok) return;
    histogram = await res.json();
  } catch (e) {
    // Non-critical: the preview image is still usable without a histogram.
    histogramId = null;
    return;
  }
  if (curId !== requestedId) return;
  drawHistogram();
  renderLevelsRow();
}

// --- 1:1 inspection -------------------------------------------------------------------
// The fitted preview is a ~4x decimated frame, so a star is roughly one pixel in it and
// its shape is gone. Zooming serves an undecimated crop of the region instead, stretched
// with the same levels as the preview so only the scale changes.
let zoom100 = false;
let zoomX = 0, zoomY = 0;          // top-left of the VISIBLE region, in full-frame pixels
let zoomW = 0, zoomH = 0;          // size of that region (the viewport), in full-frame pixels
let zoomCrop = null;               // {x,y,w,h} of the crop actually loaded, which is larger
const FRAME_INFO = new Map();      // id -> {width, height}
// Pixels fetched beyond each edge of the viewport. Panning inside this margin is a
// reposition of pixels already on screen rather than a round trip, which is what makes
// dragging feel continuous instead of stepping one viewport at a time.
const ZOOM_PAN_MARGIN = 320;

async function frameInfo(id) {
  if (FRAME_INFO.has(id)) return FRAME_INFO.get(id);
  let info = null;
  try {
    const res = await fetch('/api/frameinfo?id=' + id + sourceQuery());
    if (res.ok) info = await res.json();
  } catch (e) { /* zoom just stays unavailable */ }
  FRAME_INFO.set(id, info);
  return info;
}

function clampZoom(info) {
  zoomW = Math.max(1, Math.min(zoomW, info.width));
  zoomH = Math.max(1, Math.min(zoomH, info.height));
  zoomX = Math.max(0, Math.min(Math.round(zoomX), Math.max(0, info.width - zoomW)));
  zoomY = Math.max(0, Math.min(Math.round(zoomY), Math.max(0, info.height - zoomH)));
}

function cropForView(info) {
  const m = ZOOM_PAN_MARGIN;
  const w = Math.min(zoomW + 2 * m, info.width);
  const h = Math.min(zoomH + 2 * m, info.height);
  const x = Math.max(0, Math.min(Math.round(zoomX - (w - zoomW) / 2), info.width - w));
  const y = Math.max(0, Math.min(Math.round(zoomY - (h - zoomH) / 2), info.height - h));
  return {x: x, y: y, w: w, h: h};
}

function viewInsideCrop() {
  const c = zoomCrop;
  return !!c && zoomX >= c.x && zoomY >= c.y &&
         zoomX + zoomW <= c.x + c.w && zoomY + zoomH <= c.y + c.h;
}

// The crop is positioned explicitly rather than centred, because near a frame edge it
// gets clamped and is no longer symmetric around the viewport.
function placeZoomImage() {
  const img = document.getElementById('modal-img-wrap').querySelector('img');
  if (!img || !zoomCrop) return;
  img.style.left = (zoomCrop.x - zoomX) + 'px';
  img.style.top = (zoomCrop.y - zoomY) + 'px';
  img.style.transform = '';
  const label = document.querySelector('#modal-img-wrap .preview-status:not(.trail-status)');
  if (label) label.textContent = zoomLabel();
  retargetTrailOverlay();
}

function zoomLabel() {
  return '100% · x ' + zoomX + ', y ' + zoomY + ' · drag to pan';
}

// Where a click landed in frame pixels. The fitted image is letterboxed by
// object-fit:contain, so the drawn rectangle has to be reconstructed from the aspect
// ratios rather than read off the element box.
function framePointFromEvent(e, wrap, info) {
  const box = wrap.getBoundingClientRect();
  const scale = Math.min(box.width / info.width, box.height / info.height);
  const drawnW = info.width * scale, drawnH = info.height * scale;
  const originX = box.left + (box.width - drawnW) / 2;
  const originY = box.top + (box.height - drawnH) / 2;
  return {
    x: Math.max(0, Math.min(info.width, (e.clientX - originX) / scale)),
    y: Math.max(0, Math.min(info.height, (e.clientY - originY) / scale)),
  };
}

async function enterZoom(centreX, centreY) {
  const id = curId;
  const info = await frameInfo(id);
  if (!info || curId !== id) return;
  const box = document.getElementById('modal-img-wrap').getBoundingClientRect();
  zoomW = Math.max(1, Math.min(Math.round(box.width), info.width));
  zoomH = Math.max(1, Math.min(Math.round(box.height), info.height));
  zoomX = Math.round((centreX === undefined ? info.width / 2 : centreX) - zoomW / 2);
  zoomY = Math.round((centreY === undefined ? info.height / 2 : centreY) - zoomH / 2);
  clampZoom(info);
  zoomCrop = null;
  zoom100 = true;
  renderModal();
}

function exitZoom() {
  if (!zoom100) return;
  zoom100 = false;
  renderModal();
}

function toggleZoom() {
  if (zoom100) exitZoom(); else enterZoom();
}

// zoomW/zoomH are the viewport measured when zoom was entered, and the crop is fetched to
// match. Anything that changes the wrap's size afterwards - resizing the window, or a row
// below it changing height - would leave the 1:1 crop no longer covering it. Re-fit and
// refetch instead, keeping the same top-left so the view does not jump.
if (typeof ResizeObserver !== 'undefined') {
  let lastW = 0, lastH = 0, queued = false;
  new ResizeObserver(entries => {
    const box = entries[0].contentRect;
    const w = Math.round(box.width), h = Math.round(box.height);
    if (w === lastW && h === lastH) return;
    lastW = w; lastH = h;
    if (!zoom100 || curId === null || queued) return;
    queued = true;
    requestAnimationFrame(() => {
      queued = false;
      const info = FRAME_INFO.get(curId);
      if (!zoom100 || !info) return;
      zoomW = w; zoomH = h;
      clampZoom(info);
      loadImage();
    });
  }).observe(document.getElementById('modal-img-wrap'));
}

async function loadImage() {
  const wrap = document.getElementById('modal-img-wrap');
  const requestedId = curId, requestedStretch = curStretch, requestedZoom = zoom100;
  if (requestedZoom) {
    // Zoom survives arrowing to the next frame, so the window has to be re-fitted to
    // whatever that frame's dimensions are before anything is requested for it.
    const info = await frameInfo(requestedId);
    if (curId !== requestedId || curStretch !== requestedStretch || zoom100 !== requestedZoom) return;
    if (!info) { exitZoom(); return; }
    clampZoom(info);
    zoomCrop = cropForView(info);
  }
  const crop = zoomCrop;
  wrap.classList.add('loading');
  wrap.classList.toggle('zoomed', requestedZoom);
  wrap.classList.toggle('zoomable', !requestedZoom);
  wrap.querySelectorAll('.trail-badge, .preview-status').forEach(node => node.remove());
  const status = Object.assign(document.createElement('span'), {className: 'preview-status',
    textContent: requestedZoom ? 'Loading 1:1\u2026' : 'Rendering preview\u2026'});
  wrap.append(status);
  const img = new Image();
  img.draggable = false;
  img.style.border = '1px solid ' + filterColor(FILES[requestedId].filter);
  const stale = () => curId !== requestedId || curStretch !== requestedStretch ||
                      zoom100 !== requestedZoom || (requestedZoom && zoomCrop !== crop);
  img.onload = () => {
    if (stale()) return;  // arrowed on, panned again, or the handles moved
    wrap.replaceChildren(img);
    wrap.classList.remove('loading');
    if (requestedZoom) {
      const label = document.createElement('span');
      label.className = 'preview-status';
      label.textContent = zoomLabel();
      wrap.append(label);
      placeZoomImage();
      refreshTrailOverlay(requestedId, requestedStretch);
      return;
    }
    refreshTrailOverlay(requestedId, requestedStretch);
    if (FILES[requestedId].has_trail) {
      const badge = document.createElement('span');
      badge.className = 'trail-badge';
      badge.title = 'Possible satellite trail detected';
      badge.textContent = '\u{1F6F0}\uFE0F';
      wrap.append(badge);
    }
  };
  img.onerror = () => {
    if (stale()) return;
    wrap.replaceChildren(Object.assign(document.createElement('span'), {className: 'preview-status', textContent: 'Failed to load preview.'}));
    wrap.classList.remove('loading');
  };
  // Resolves either way, so the levels single-flight guard always clears.
  const settled = new Promise(resolve => {
    img.addEventListener('load', resolve, {once: true});
    img.addEventListener('error', resolve, {once: true});
  });
  // No cache-buster: the server sends an ETag keyed on the file's mtime and the levels,
  // so an already-seen render comes back from the browser cache as a 304.
  const lv = levelsQuery();
  img.src = requestedZoom
    ? '/api/crop?id=' + requestedId + '&stretch=' + encodeURIComponent(requestedStretch) + lv +
      '&x=' + crop.x + '&y=' + crop.y + '&w=' + crop.w + '&h=' + crop.h + sourceQuery()
    : '/api/preview?id=' + requestedId + '&stretch=' + encodeURIComponent(requestedStretch) + lv + sourceQuery();
  return settled;
}

// Drag to pan at 1:1, and click the fitted preview to zoom straight to that spot.
(function () {
  const wrap = document.getElementById('modal-img-wrap');
  let dragging = false, moved = false, startX = 0, startY = 0, originX = 0, originY = 0;

  wrap.addEventListener('mousedown', (e) => {
    if (!zoom100 || e.button !== 0) return;
    dragging = true; moved = false;
    startX = e.clientX; startY = e.clientY;
    originX = zoomX; originY = zoomY;
    wrap.classList.add('panning');
    e.preventDefault();
  });

  window.addEventListener('mousemove', (e) => {
    if (!dragging) return;
    const dx = e.clientX - startX, dy = e.clientY - startY;
    if (Math.abs(dx) > 2 || Math.abs(dy) > 2) moved = true;
    const img = wrap.querySelector('img');
    if (img) img.style.transform = 'translate(' + dx + 'px,' + dy + 'px)';
  });

  window.addEventListener('mouseup', (e) => {
    if (!dragging) return;
    dragging = false;
    wrap.classList.remove('panning');
    if (!moved) return;
    const info = FRAME_INFO.get(curId);
    if (!info) return;
    zoomX = originX - (e.clientX - startX);
    zoomY = originY - (e.clientY - startY);
    clampZoom(info);
    // The loaded crop reaches a margin past the viewport, so a short drag lands inside
    // pixels that are already here: reposition them and skip the round trip entirely.
    // Only a drag past the margin needs a new crop - and the dragged offset stays on
    // screen until it arrives, rather than snapping back first.
    if (viewInsideCrop()) placeZoomImage(); else loadImage();
  });

  wrap.addEventListener('click', async (e) => {
    if (curId === null || zoom100 || moved) return;
    if (!e.target.closest('img')) return;
    const info = await frameInfo(curId);
    if (!info) return;
    const pt = framePointFromEvent(e, wrap, info);
    enterZoom(pt.x, pt.y);
  });

  // A wheel over the zoomed view pans it too, which is the natural gesture on a trackpad.
  wrap.addEventListener('wheel', (e) => {
    if (!zoom100) return;
    const info = FRAME_INFO.get(curId);
    if (!info) return;
    e.preventDefault();
    zoomX += e.deltaX;
    zoomY += e.deltaY;
    clampZoom(info);
    if (viewInsideCrop()) placeZoomImage(); else loadImage();
  }, {passive: false});
})();

const TRAIL_BY_ID = new Map();   // id -> geometry, or null once we know there is none
let showTrail = true;

async function trailGeometry(id) {
  if (TRAIL_BY_ID.has(id)) return TRAIL_BY_ID.get(id);
  let geometry = null;
  try {
    const res = await fetch('/api/trail?id=' + id);
    if (res.ok) geometry = (await res.json()).trail;
  } catch (e) {
    // The overlay is an extra; the preview underneath is unaffected.
  }
  TRAIL_BY_ID.set(id, geometry);
  return geometry;
}

function trailAngleDeg(g) {
  // As seen on screen: counterclockwise from horizontal, and undirected - a single frame
  // cannot say which way along the line the satellite was travelling.
  return ((-Math.atan2(g.y1 - g.y0, g.x1 - g.x0) * 180 / Math.PI) % 180 + 180) % 180;
}

function buildTrailOverlay(g) {
  const NS = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('class', 'trail-overlay');
  applyTrailViewBox(svg, g);

  // The box is padded past the trail's ends, which for a trail that reaches the frame edge
  // would otherwise draw over the letterboxing beside the image.
  const clip = document.createElementNS(NS, 'clipPath');
  clip.id = 'trail-clip';
  const clipRect = document.createElementNS(NS, 'rect');
  clipRect.setAttribute('width', g.frame_w);
  clipRect.setAttribute('height', g.frame_h);
  clip.append(clipRect);
  const marks = document.createElementNS(NS, 'g');
  marks.setAttribute('clip-path', 'url(#trail-clip)');
  svg.append(clip, marks);

  const dx = g.x1 - g.x0, dy = g.y1 - g.y0;
  const len = Math.hypot(dx, dy) || 1;
  const ux = dx / len, uy = dy / len;
  // Padded well past the measured ridge width: at preview scale the trail itself is about one
  // pixel, and a box drawn tight to it would hide what it is pointing at.
  const half = Math.max(g.width * 3, g.frame_w / 90);
  const px = -uy * half, py = ux * half, ex = ux * half, ey = uy * half;
  const corners = [[g.x0 - ex + px, g.y0 - ey + py], [g.x1 + ex + px, g.y1 + ey + py],
                   [g.x1 + ex - px, g.y1 + ey - py], [g.x0 - ex - px, g.y0 - ey - py]];
  const box = document.createElementNS(NS, 'polygon');
  box.setAttribute('points', corners.map(c => c[0].toFixed(1) + ',' + c[1].toFixed(1)).join(' '));
  box.setAttribute('class', 'trail-box');
  marks.append(box);

  // Ticks just outside each end, so the extent is readable where the box sides converge.
  for (const [x, y, dir] of [[g.x0, g.y0, -1], [g.x1, g.y1, 1]]) {
    const tick = document.createElementNS(NS, 'line');
    tick.setAttribute('x1', (x + dir * ex * 2 + px).toFixed(1));
    tick.setAttribute('y1', (y + dir * ey * 2 + py).toFixed(1));
    tick.setAttribute('x2', (x + dir * ex * 2 - px).toFixed(1));
    tick.setAttribute('y2', (y + dir * ey * 2 - py).toFixed(1));
    tick.setAttribute('class', 'trail-box');
    marks.append(tick);
  }
  return svg;
}

// Fitted, the viewBox is the whole frame and "meet" letterboxes exactly the way the
// image's object-fit:contain does, so the box lands on the trail at any window size. At
// 1:1 the image is a crop, not the whole frame, so the same viewBox drew the box in the
// wrong place entirely - there the visible region IS the viewBox, and one frame pixel is
// one screen pixel, so it maps exactly.
function applyTrailViewBox(svg, g) {
  if (zoom100) {
    svg.setAttribute('viewBox', zoomX + ' ' + zoomY + ' ' + zoomW + ' ' + zoomH);
    svg.setAttribute('preserveAspectRatio', 'none');
  } else {
    svg.setAttribute('viewBox', '0 0 ' + g.frame_w + ' ' + g.frame_h);
    svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
  }
}

// Panning at 1:1 moves the visible region, so the overlay has to follow it.
function retargetTrailOverlay() {
  const svg = document.querySelector('#modal-img-wrap .trail-overlay');
  if (svg && svg.__geometry) applyTrailViewBox(svg, svg.__geometry);
}

function refreshTrailOverlay(requestedId, requestedStretch) {
  const wrap = document.getElementById('modal-img-wrap');
  wrap.querySelectorAll('.trail-overlay, .trail-status').forEach(node => node.remove());
  if (!showTrail || !FILES[requestedId].has_trail) return;
  const requestedZoom = zoom100;
  trailGeometry(requestedId).then(geometry => {
    // Locating a trail takes about a second, so re-check everything it was drawn for -
    // including the zoom, or an overlay asked for while fitted lands after a zoom and is
    // drawn against the wrong viewBox.
    if (!geometry || !showTrail || curId !== requestedId ||
        curStretch !== requestedStretch || zoom100 !== requestedZoom) return;
    if (wrap.querySelector('.trail-overlay')) return;
    // The readout is HTML rather than SVG text: inside the viewBox it would be scaled by
    // whatever the window size makes of a 3000 px frame.
    const status = document.createElement('span');
    status.className = 'preview-status trail-status';
    status.textContent = '\u{1F6F0}\uFE0F trail ' + trailAngleDeg(geometry).toFixed(0) +
      '\u00b0 \u00b7 ' + geometry.width.toFixed(1) + ' px wide';
    const svg = buildTrailOverlay(geometry);
    svg.__geometry = geometry;   // so a pan can retarget it without refetching
    wrap.append(svg, status);
  });
}

function toggleTrail() {
  showTrail = !showTrail;
  document.getElementById('trail-btn').classList.toggle('active', showTrail);
  refreshTrailOverlay(curId, curStretch);
}

const STRETCH_MODES = ['auto', 'super', 'linear'];

function markStretchButtons(prefix, active) {
  for (const mode of STRETCH_MODES) {
    const btn = document.getElementById(prefix + mode + '-btn');
    if (btn) btn.classList.toggle('active', active === mode);
  }
}

function setStretch(mode) {
  // Clicking a preset re-fits its three handles to this frame, even when it is already
  // the selected mode - that is how you get back after adjusting them by hand.
  curStretch = mode;
  levels = null;
  levelsAuto = true;
  markStretchButtons('', curStretch);
  // The histogram itself is the same pixel distribution whatever the preset, so it is not
  // refetched; only the handle positions drawn over it change.
  drawHistogram();
  renderLevelsRow();
  loadImage();   // keeps the 1:1 window where it is, just restretched
}

async function postAction(action, payload) {
  // Every write names the dataset version the page was built from: after a rescan the
  // same ids mean other frames, and the server refuses rather than moving those.
  const res = await fetch('/api/' + action, {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(Object.assign({version: REVIEW_DATA.version}, payload))
  });
  const detail = await res.json().catch(() => ({}));
  if (res.status === 409 && detail.stale) showReloadBanner();
  if (!res.ok) {
    alert('Action failed: ' + (detail.error || res.status));
    return null;
  }
  return detail;
}

async function applyMove(action, ids) {
  // One request for the whole batch: the server moves them under a single lock and
  // rewrites the recovery state file once, rather than once per frame.
  const detail = await postAction(action, {ids});
  if (detail === null) return null;
  const failed = detail.failed || {};
  const nowDeleted = action === 'delete';
  for (const id of ids) {
    if (failed[id] !== undefined) continue;
    FILES[id].deleted = nowDeleted;
    setSelected(id, false, true);
  }
  updateDeleteButtons();
  const failedIds = Object.keys(failed);
  if (failedIds.length) {
    alert(failedIds.length + ' frame(s) could not be moved:\n' +
          failedIds.map(id => FILES[id].name + ': ' + failed[id]).join('\n'));
  }
  updateDerivedView();
  // The group's shared stretch is fitted from the frames that survive, so a delete or
  // restore changes it. Re-render the open grid so every thumbnail on screen is scaled
  // the same way again - once for the whole batch, since applyMove is one call per
  // selection rather than one per frame.
  if (ids.length > failedIds.length &&
      document.getElementById('grid-overlay').classList.contains('open')) {
    renderGroupGrid(true);
  }
  return failed;
}

async function toggleDelete(skipConfirm) {
  const f = FILES[curId];
  if (f.missing) return null;
  const action = f.deleted ? 'restore' : 'delete';
  if (action === 'delete' && !skipConfirm && !confirm('Move this frame to _deleted?\n' + f.name + '\n\nThis is reversible (Restore), the file is moved not erased.')) {
    return;
  }
  const delBtn = document.getElementById('delete-btn');
  delBtn.disabled = true;
  try {
    const failed = await applyMove(action, [f.id]);
    if (failed !== null) renderModal();
    return failed;
  } finally {
    delBtn.disabled = false;
  }
}

async function deleteAndAdvance() {
  // Keyboard triage: Del moves the frame on screen to _deleted - no dialog, it is one
  // click to undo - and steps on to the next one.
  const f = FILES[curId];
  if (!f || f.deleted || f.missing) { nextFile(1); return; }
  const failed = await toggleDelete(true);
  if (failed && failed[f.id] === undefined) nextFile(1);
}

function nextFile(delta) {
  if (curId === null || !FILES.length) return;
  const list = navIds && navIds.includes(curId) ? navIds : FILES.map(f => f.id);
  const n = list.length;
  const at = list.indexOf(curId);
  const hideDeleted = document.body.classList.contains('hide-deleted');
  // Skip over frames the current view hides, so arrowing follows what is on screen.
  for (let step = 1; step <= n; step++) {
    const id = list[((at + delta * step) % n + n) % n];
    const f = FILES[id];
    if (id === curId || (!f.sigmaClipped && !(hideDeleted && f.deleted))) { openModal(id); return; }
  }
}

let groupPanel = null;
let groupFilter = null;
let groupExposure = null;   // stats.exposure_key of the set on screen, or null for all
let groupStretch = 'auto';
const GRID_ITEM_BY_ID = new Map();
const GRID_BOX_BY_ID = new Map();

function groupFileIds(panel, filt, exposure) {
  const ids = [];
  for (const unit of UNITS) {
    if (unit.panel !== panel || unit.filter !== filt || unit.exposure !== exposure) continue;
    for (const fid of unit.file_ids) {
      if (!FILES[fid].deleted) ids.push(fid);
    }
  }
  return ids;
}

function exposureSuffix(exposure) {
  return exposure ? exposure + 's' : 'no exposure';
}

function groupLabel(g) {
  return g.panel + ' \u00b7 Filter ' + g.filter + ' \u00b7 ' + exposureSuffix(g.exposure);
}

function currentGroup() {
  return GROUPS.find(g => g.panel === groupPanel && g.filter === groupFilter &&
                          g.exposure === groupExposure);
}

function showGroup(panel, filt, exposure) {
  groupPanel = panel;
  groupFilter = filt;
  groupExposure = exposure === undefined || exposure === null ? '' : exposure;
  // The length is always named, even when a filter has only one: it is what the shared
  // stretch was fitted to, so it belongs on the grid that stretch is showing.
  document.getElementById('grid-title').textContent =
    panel + ' \u2014 Filter ' + filt + ' \u00b7 ' + exposureSuffix(groupExposure) + ' (all sessions)';
  renderGroupGrid();
}

function navigateGroup(delta) {
  // Wraps, like arrowing through frames in the preview. GROUPS is in the same order as the
  // sections down the page, so this walks every filter of a panel before moving on.
  if (GROUPS.length < 2) return;
  const i = GROUPS.indexOf(currentGroup());
  if (i < 0) return;
  const g = GROUPS[((i + delta) % GROUPS.length + GROUPS.length) % GROUPS.length];
  showGroup(g.panel, g.filter, g.exposure);   // keeps the chosen stretch mode across groups
}

function updateGroupNav() {
  const i = GROUPS.indexOf(currentGroup());
  for (const [id, delta] of [['grid-prev-btn', -1], ['grid-next-btn', 1]]) {
    const btn = document.getElementById(id);
    btn.disabled = GROUPS.length < 2 || i < 0;
    btn.title = btn.disabled ? ''
      : 'Go to ' + groupLabel(GROUPS[((i + delta) % GROUPS.length + GROUPS.length) % GROUPS.length]);
  }
  return i;
}

function openGroupGrid(panel, filt, exposure) {
  groupStretch = 'auto';
  showGroup(panel, filt, exposure);
  document.getElementById('grid-overlay').classList.add('open');
}

function closeGroupGrid() {
  cancelGridLoads();
  document.getElementById('grid-overlay').classList.remove('open');
  GRID_ITEM_BY_ID.clear();
  GRID_BOX_BY_ID.clear();
  updateDeleteButtons();
}

function setGroupStretch(mode) {
  if (mode === groupStretch) return;
  groupStretch = mode;
  renderGroupGrid(true);
}

let gridGroupIndex = -1;

// --- thumbnail loading ----------------------------------------------------------------
// Assigning every tile's src up front queued a whole group's renders the moment the grid
// was built, and arrowing through groups stacked them: the tiles of every group passed
// through kept their place in a six-deep browser queue, and the server rendered all of
// them. Several hundred requests would pile up and a thumbnail could take forty seconds
// to arrive because it was behind groups nobody was looking at any more.
//
// Instead a tile only asks for its image once it comes near the viewport, at most a few
// at a time, and moving to another group cancels whatever is still outstanding.
const THUMB_CONCURRENCY = 6;
const THUMB_PRELOAD_MARGIN = '600px';
let gridToken = 0;
const gridQueue = [];
const gridInFlight = new Set();

const gridObserver = typeof IntersectionObserver === 'undefined' ? null :
  new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      gridObserver.unobserve(entry.target);
      gridQueue.push(entry.target);
    }
    pumpGrid();
  }, {root: document.getElementById('grid-wrap'), rootMargin: THUMB_PRELOAD_MARGIN});

function gridObserve(img) {
  img.dataset.token = String(gridToken);
  // Without IntersectionObserver every tile simply queues; the concurrency limit and the
  // cancellation below still apply, which is what keeps navigation responsive.
  if (gridObserver) gridObserver.observe(img); else { gridQueue.push(img); pumpGrid(); }
}

function pumpGrid() {
  while (gridInFlight.size < THUMB_CONCURRENCY && gridQueue.length) {
    const img = gridQueue.shift();
    if (img.dataset.token !== String(gridToken) || !img.isConnected || img.src) continue;
    gridInFlight.add(img);
    const done = () => { gridInFlight.delete(img); pumpGrid(); };
    img.addEventListener('load', done, {once: true});
    img.addEventListener('error', done, {once: true});
    img.src = img.dataset.src;
  }
}

function cancelGridLoads() {
  gridToken++;
  gridQueue.length = 0;
  for (const img of gridInFlight) {
    // Removing the attribute aborts the fetch; dropping the element alone does not, which
    // is why the abandoned groups' requests kept their place in the queue.
    img.removeAttribute('src');
  }
  gridInFlight.clear();
}

function setGridCount(total, clipped) {
  document.getElementById('grid-count').textContent =
    (total - clipped) + ' of ' + total + ' included' +
    (clipped ? ' \u00b7 ' + clipped + ' clipped' : '') + ' \u00b7 same stretch for all' +
    (gridGroupIndex >= 0 ? ' \u00b7 group ' + (gridGroupIndex + 1) + '/' + GROUPS.length : '');
}

function renderGroupGrid(keepScroll = false) {
  markStretchButtons('grid-', groupStretch);
  const groupIndex = updateGroupNav();

  const ids = groupFileIds(groupPanel, groupFilter, groupExposure);
  gridGroupIndex = groupIndex;
  const anyCalibrated = ids.some(id => FILES[id].has_calibrated);
  for (const id of ['grid-raw-btn', 'grid-calibrated-btn']) {
    document.getElementById(id).hidden = !anyCalibrated;
  }
  markSourceButtons();
  setGridCount(ids.length, ids.filter(id => FILES[id].sigmaClipped).length);
  const wrap = document.getElementById('grid-wrap');
  // A delete or a stretch toggle rebuilds every tile, so those keep your place; moving to
  // another group is a different set of frames and starts at the top.
  const scrollTop = keepScroll ? wrap.scrollTop : 0;
  cancelGridLoads();
  wrap.replaceChildren();
  GRID_ITEM_BY_ID.clear();
  GRID_BOX_BY_ID.clear();
  for (const id of ids) {
    const f = FILES[id];
    const folder = FOLDERS[f.folder_id];
    const item = document.createElement('div');
    // The sigma slider drops these from every statistic and hides their table row, but the
    // grid used to show them as ordinary members of the set - so the tile count disagreed
    // with the "n/m included" line right above it with nothing to explain the difference.
    item.className = 'grid-item' + (f.sigmaClipped ? ' clipped' : '');
    const thumbWrap = document.createElement('div');
    thumbWrap.className = 'thumb-wrap';
    const img = document.createElement('img');
    img.alt = f.name;
    // Deliberately not img.src, and not loading="lazy": see the loader below.
    img.dataset.src = '/api/thumb?id=' + id + '&panel=' + encodeURIComponent(groupPanel) +
      '&filter=' + encodeURIComponent(groupFilter) + '&stretch=' + groupStretch +
      '&exp=' + encodeURIComponent(groupExposure) + (FILES[id].has_calibrated ? sourceQuery() : '');
    gridObserve(img);
    // Every frame in this grid shares one filter, so a filter-coloured border would say
    // nothing. Spend it on the quality verdict instead - bad frames stand out while you
    // are looking at the images rather than reading the captions.
    img.style.border = '2px solid ' + verdictColor(f);
    img.addEventListener('click', () => openModal(id, Array.from(GRID_ITEM_BY_ID.keys())));
    const box = document.createElement('input');
    box.type = 'checkbox';
    box.className = 'grid-select';
    box.title = 'Mark for deletion';
    box.checked = selectedIds.has(id);
    box.addEventListener('change', () => setSelected(id, box.checked));
    thumbWrap.append(img, box);
    GRID_BOX_BY_ID.set(id, box);
    if (f.has_trail) {
      const badge = document.createElement('span');
      badge.className = 'trail-badge';
      badge.title = 'Possible satellite trail detected';
      badge.textContent = '\u{1F6F0}\uFE0F';
      thumbWrap.append(badge);
    }
    if (f.sigmaClipped) {
      const badge = document.createElement('span');
      badge.className = 'clip-badge';
      badge.title = 'Excluded by the sigma-clipping slider; not counted in this set\u2019s statistics';
      badge.textContent = 'clipped';
      thumbWrap.append(badge);
    }
    // The filename is 90 characters of camera settings and repeats across the whole grid,
    // so it moves to the hover title and the caption carries what you actually compare
    // thumbnails on.
    const cap = document.createElement('div');
    cap.className = 'grid-cap';
    cap.title = f.name + (f.quality_reasons && f.quality_reasons.length
      ? '\n\n' + f.quality_reasons.join('; ') : '');
    const q = document.createElement('span');
    q.className = 'grid-q q-' + (f.quality_verdict || '');
    q.textContent = numeric(f.quality_score) ? 'Q' + Math.round(f.quality_score) : 'Q' + DASH;
    const rest = document.createElement('span');
    rest.className = 'grid-metrics';
    rest.textContent = 'A' + fmtNum(f.adu_mean, 0) + ' \u00b7 \u2605' + (f.num_stars ?? DASH);
    const sess = document.createElement('span');
    sess.className = 'grid-sess';
    sess.textContent = folder ? folder.session : '';
    cap.append(q, rest, sess);
    item.append(thumbWrap, cap);
    wrap.append(item);
    GRID_ITEM_BY_ID.set(id, item);
  }
  wrap.scrollTop = scrollTop;
  updateDeleteButtons();
}

function goToStats(anchorId) {
  const heading = document.getElementById(anchorId);
  if (!heading) return;
  if (document.getElementById('grid-overlay').classList.contains('open')) closeGroupGrid();
  // The panel name lives in the <h2> above the filter heading, so scroll that into view
  // too when it is the first filter of its panel - otherwise you land on a bare "Filter R".
  const prev = heading.previousElementSibling;
  const target = prev && prev.tagName === 'H2' ? prev : heading;
  target.scrollIntoView({behavior: 'smooth', block: 'start'});
  heading.classList.remove('flash');
  void heading.offsetWidth;   // restart the animation when the same section is revisited
  heading.classList.add('flash');
}

let gridPressTarget = null;
document.getElementById('grid-overlay').addEventListener('mousedown', (e) => {
  gridPressTarget = e.target;
});
document.getElementById('grid-overlay').addEventListener('click', (e) => {
  if (e.target.id === 'grid-overlay' && gridPressTarget &&
      gridPressTarget.id === 'grid-overlay') closeGroupGrid();
  gridPressTarget = null;
});

document.addEventListener('keydown', (e) => {
  if (document.getElementById('modal-overlay').classList.contains('open')) {
    if (e.key === 'Escape') { if (zoom100) exitZoom(); else closeModal(); }
    else if (e.key === 'ArrowRight') nextFile(1);
    else if (e.key === 'ArrowLeft') nextFile(-1);
    else if (e.key === 'Delete' || e.key === 'Backspace') deleteAndAdvance();
    else return;
    e.preventDefault();
  } else if (document.getElementById('timeline-overlay').classList.contains('open')) {
    if (e.key === 'Escape') { closeTimeline(); e.preventDefault(); }
  } else if (document.getElementById('grid-overlay').classList.contains('open')) {
    if (e.key === 'Escape') closeGroupGrid();
    else if (e.key === 'ArrowRight') navigateGroup(1);
    else if (e.key === 'ArrowLeft') navigateGroup(-1);
    else return;
    e.preventDefault();
  }
});

// A click event fires on the nearest common ancestor of where the button went down and
// where it came up. Dragging a histogram handle or panning the image past the edge of the
// modal therefore delivered a click on the overlay itself, and the frame closed mid-drag.
// Only treat it as a dismissal when the press started on the backdrop too.
let modalPressTarget = null;
document.getElementById('modal-overlay').addEventListener('mousedown', (e) => {
  modalPressTarget = e.target;
});
document.getElementById('modal-overlay').addEventListener('click', (e) => {
  if (e.target.id === 'modal-overlay' && modalPressTarget &&
      modalPressTarget.id === 'modal-overlay') closeModal();
  modalPressTarget = null;
});

// One delegated listener instead of an inline onclick on every row.
document.querySelector('.data-pane').addEventListener('click', (e) => {
  const row = e.target.closest('tr[data-id]');
  if (row) openModal(Number(row.dataset.id), visibleRowIds(row.parentElement));
});

const selectedIds = new Set();

function selectedFor(action) {
  // Selected frames the action applies to: live ones for delete, deleted-but-not-purged
  // ones for restore. One selection can hold both, and each button counts its own.
  return Array.from(selectedIds).filter(id => {
    const f = FILES[id];
    return !f.missing && (action === 'delete' ? !f.deleted : f.deleted);
  });
}

function updateDeleteButtons() {
  const btn = document.getElementById('bulk-delete-btn');
  const toDelete = selectedFor('delete').length, toRestore = selectedFor('restore').length;
  btn.disabled = toDelete === 0;
  btn.textContent = 'Delete selected (' + toDelete + ')';
  const restoreBtn = document.getElementById('bulk-restore-btn');
  restoreBtn.disabled = toRestore === 0;
  restoreBtn.textContent = 'Restore selected (' + toRestore + ')';
  const purgeable = FILES.filter(f => f.deleted && !f.missing).length;
  const purgeBtn = document.getElementById('purge-btn');
  purgeBtn.disabled = purgeable === 0;
  purgeBtn.textContent = 'Empty _deleted (' + purgeable + ')';
  const gridBtn = document.getElementById('grid-delete-btn');
  const inGrid = gridSelectedIds().length;
  gridBtn.disabled = inGrid === 0;
  gridBtn.textContent = 'Delete selected (' + inGrid + ')';
}

function setSelected(id, checked, defer) {
  // One selection shared by the table rows and the thumbnail grid, so a frame ticked in
  // the grid is still ticked in the table after the grid is closed, and vice versa.
  // `defer` skips the button refresh for callers clearing a whole batch, which otherwise
  // rescanned the grid's selection once per frame; they refresh once when the loop ends.
  if (checked) selectedIds.add(id); else selectedIds.delete(id);
  const row = BOX_BY_ID.get(id);
  if (row && row.checked !== checked) row.checked = checked;
  const thumb = GRID_BOX_BY_ID.get(id);
  if (thumb && thumb.checked !== checked) thumb.checked = checked;
  if (!defer) updateDeleteButtons();
}

document.querySelector('.data-pane').addEventListener('change', (e) => {
  const box = e.target.closest('input.row-select');
  if (!box) return;
  setSelected(Number(box.dataset.id), box.checked);
});

function gridSelectedIds() {
  // Only the frames this grid is showing. The selection itself is global, so the grid's
  // own delete must not touch frames ticked in another panel/filter's table or grid.
  const ids = [];
  for (const id of GRID_BOX_BY_ID.keys()) {
    if (selectedIds.has(id) && !FILES[id].deleted) ids.push(id);
  }
  return ids;
}

async function deleteGridSelected() {
  const ids = gridSelectedIds();
  if (!ids.length) return;
  if (!confirm('Move ' + ids.length + ' frame(s) selected in this view to _deleted?\n\n' +
               'Only frames shown here are affected. This is reversible (Restore in the preview), files are moved not erased.')) {
    return;
  }
  const btn = document.getElementById('grid-delete-btn');
  btn.disabled = true;
  try {
    await applyMove('delete', ids);
  } finally {
    updateDeleteButtons();
  }
}

async function bulkDeleteSelected() {
  const ids = selectedFor('delete');
  if (!ids.length) return;
  if (!confirm('Move ' + ids.length + ' selected frame(s) to _deleted?\n\nThis is reversible (Restore in the preview), files are moved not erased.')) {
    return;
  }
  const btn = document.getElementById('bulk-delete-btn');
  btn.disabled = true;
  try {
    await applyMove('delete', ids);
  } finally {
    updateDeleteButtons();
  }
}

async function bulkRestoreSelected() {
  const ids = selectedFor('restore');
  if (!ids.length) return;
  const btn = document.getElementById('bulk-restore-btn');
  btn.disabled = true;
  try {
    await applyMove('restore', ids);
  } finally {
    updateDeleteButtons();
  }
}

async function purgeDeleted() {
  const n = FILES.filter(f => f.deleted && !f.missing).length;
  if (!n) return;
  // Typed, not clicked: this is the one action in the reviewer that cannot be undone.
  const answer = prompt('Permanently erase the ' + n + ' frame(s) in _deleted?\n\n' +
                        'This cannot be undone. Type ' + n + ' to confirm.');
  if (answer === null || answer.trim() !== String(n)) return;
  const detail = await postAction('purge', {confirm: n});
  if (detail === null) return;
  for (const f of FILES) if (f.deleted && !(detail.failed || {})[f.id]) f.missing = true;
  const failedIds = Object.keys(detail.failed || {});
  if (failedIds.length) alert(failedIds.length + ' frame(s) could not be erased.');
  updateDerivedView();
}

function setShowDeleted(show) {
  document.body.classList.toggle('hide-deleted', !show);
}

const SELECT_RULES = {
  reject: f => f.quality_verdict === 'reject',
  warn: f => f.quality_verdict === 'reject' || f.quality_verdict === 'warn',
  trail: f => !!f.has_trail,
  clipped: f => !!f.sigmaClipped,
};

function selectBy(kind) {
  if (!kind) return;
  if (kind === 'none') {
    for (const id of Array.from(selectedIds)) setSelected(id, false, true);
  } else {
    const rule = SELECT_RULES[kind];
    for (const f of FILES) if (!f.deleted && !f.missing && rule(f)) setSelected(f.id, true, true);
  }
  updateDeleteButtons();
}

function selectGridWhere(kind) {
  const rule = SELECT_RULES[kind];
  for (const id of GRID_ITEM_BY_ID.keys()) if (rule(FILES[id])) setSelected(id, true, true);
  updateDeleteButtons();
}

function download(name, text, type) {
  const url = URL.createObjectURL(new Blob([text], {type: type}));
  const a = Object.assign(document.createElement('a'), {href: url, download: name});
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function csvCell(v) {
  if (v === null || v === undefined) return '';
  const s = Array.isArray(v) ? v.join('; ') : String(v);
  return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
}

const CSV_COLUMNS = ['path', 'name', 'session', 'panel', 'filter', 'exposure', 'date_obs',
  'quality_score', 'quality_verdict', 'quality_reasons', 'hfr_avg', 'fwhm_avg', 'fwhm_arcsec',
  'num_stars', 'roundness_avg', 'signal_adu_per_sec', 'has_trail', 'calibration',
  'solved_ra', 'solved_dec', 'solved_rotation', 'pointing_error_arcsec',
  'deleted', 'sigma_clipped', 'selected'];

function exportCsv() {
  const lines = [CSV_COLUMNS.join(',')];
  for (const f of FILES) {
    const folder = FOLDERS[f.folder_id] || {};
    const row = Object.assign({}, f, {session: folder.session, panel: folder.panel,
      deleted: !!f.deleted, sigma_clipped: !!f.sigmaClipped, selected: selectedIds.has(f.id)});
    lines.push(CSV_COLUMNS.map(c => csvCell(row[c])).join(','));
  }
  download('frames.csv', lines.join('\n') + '\n', 'text/csv');
}

function exportRejectList() {
  const paths = FILES.filter(f => f.deleted || selectedIds.has(f.id)).map(f => f.path);
  if (!paths.length) { alert('No frames are deleted or selected.'); return; }
  download('rejected_frames.txt', paths.join('\n') + '\n', 'text/plain');
}

// --- dataset reload -------------------------------------------------------------------
function showReloadBanner() {
  document.getElementById('reload-banner').hidden = false;
}

setInterval(async () => {
  try {
    const res = await fetch('/api/version', {cache: 'no-store'});
    if (res.ok && (await res.json()).version !== REVIEW_DATA.version) showReloadBanner();
  } catch (e) { /* server stopped; nothing to reload from */ }
}, 10000);

function indexDom() {
  document.querySelectorAll('tr[data-id]').forEach(row => ROW_BY_ID.set(Number(row.dataset.id), row));
  document.querySelectorAll('input.row-select').forEach(box => BOX_BY_ID.set(Number(box.dataset.id), box));
  document.querySelectorAll('tr[data-id]').forEach(row => {
    const cell = row.querySelector('td.dev-cell');
    if (cell) DEV_BY_ID.set(Number(row.dataset.id), cell);
  });
  document.querySelectorAll('[data-unit-summary]').forEach(el =>
    SUMMARY_GROUPS.push({ids: el.dataset.unitSummary.split(',').filter(Boolean).map(Number), el}));
  for (const unit of UNITS) UNIT_BY_KEY.set(unitKey(unit.panel, unit.session, unit.filter, unit.exposure), unit);
  document.querySelectorAll('.matrix-cell').forEach(cell => {
    const key = unitKey(cell.dataset.panel, cell.dataset.session, cell.dataset.filter, cell.dataset.exposure);
    // Each cell keeps one span for its lifetime, so an update is a textContent write
    // rather than re-parsing HTML for every cell of the matrix on every slider tick.
    const span = document.createElement('span');
    cell.replaceChildren(span);
    MATRIX_CELLS.push({span, unit: UNIT_BY_KEY.get(key) || null, text: null});
  });
  // One row is one panel+filter at one exposure, so a row total sums that row's units
  // over every session. Resolved once here, the same way the session cells are.
  document.querySelectorAll('.matrix-total[data-total]').forEach(cell => {
    const units = UNITS.filter(u => u.panel === cell.dataset.panel &&
                                    u.filter === cell.dataset.filter &&
                                    u.exposure === cell.dataset.exposure);
    MATRIX_TOTALS.push({cell, units, kind: cell.dataset.total, text: null});
  });
}

indexDom();
updateDerivedView();

// --- per-night timeline ----------------------------------------------------------------
// Three small multiples sharing one time axis - FWHM, sky brightness, star count - so focus
// drift, a rising moon or passing cloud read at a glance. One y-scale per chart (never two
// on one), points coloured by filter with the marker shape as a second cue: circles for
// broadband, diamonds for narrowband, since R and H share a hue.
const TIMELINE_METRICS = [
  ['fwhm_avg', 'FWHM (px)', 2],
  ['signal_adu_per_sec', 'Sky (ADU/s)', 2],
  ['num_stars', 'Stars', 0],
];
const NARROWBAND = new Set(REVIEW_DATA.narrowband || []);
const SVG_NS = 'http://www.w3.org/2000/svg';

function svgEl(name, attrs) {
  const el = document.createElementNS(SVG_NS, name);
  for (const [k, v] of Object.entries(attrs || {})) el.setAttribute(k, v);
  return el;
}

function frameTime(f) {
  const t = f.date_obs ? Date.parse(f.date_obs) : NaN;
  return Number.isFinite(t) ? t : null;
}

function fmtClock(t) {
  const d = new Date(t);
  return String(d.getHours()).padStart(2, '0') + ':' + String(d.getMinutes()).padStart(2, '0');
}

function niceTicks(lo, hi, count) {
  const span = hi - lo || 1;
  const step0 = span / count;
  const mag = Math.pow(10, Math.floor(Math.log10(step0)));
  let step = [1, 2, 5, 10].map(m => m * mag).find(st => st >= step0) || step0;
  // At least two gridlines, or a narrow range has no scale to read against.
  while (Math.floor(hi / step) - Math.ceil(lo / step) < 1 && step > span / 50) step /= 2;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) {
    out.push(Math.abs(v) < step * 1e-9 ? 0 : v);   // never a "-0" label
  }
  // Enough decimals to tell neighbouring ticks apart - a fixed count collapsed a narrow
  // FWHM range (3.75, 3.80, 3.85) into one repeated label.
  out.decimals = Math.max(0, -Math.floor(Math.log10(step) + 1e-9));
  return out;
}

function closeTimeline() {
  document.getElementById('timeline-overlay').classList.remove('open');
  document.getElementById('timeline-tip').hidden = true;
}

function openTimeline(session) {
  const frames = FILES.filter(f => !f.deleted && (FOLDERS[f.folder_id] || {}).session === session &&
                                   frameTime(f) !== null)
    .sort((a, b) => frameTime(a) - frameTime(b));
  document.getElementById('timeline-title').textContent = session + ' · ' + frames.length + ' frame(s)';
  const charts = document.getElementById('timeline-charts');
  charts.replaceChildren();
  const legend = document.getElementById('timeline-legend');
  legend.replaceChildren();
  document.getElementById('timeline-overlay').classList.add('open');
  if (!frames.length) {
    charts.textContent = 'None of these frames carries a DATE-OBS/DATE-LOC time.';
    return;
  }
  const filters = Array.from(new Set(frames.map(f => f.filter || '?')));
  for (const filt of filters) {
    const item = document.createElement('span');
    item.className = 'tl-legend-item';
    const key = svgEl('svg', {width: 14, height: 14, viewBox: '-7 -7 14 14'});
    key.append(timelineMarker(filt, 0, 0));
    item.append(key, document.createTextNode(filt));
    legend.append(item);
  }
  const ids = frames.map(f => f.id);
  const t0 = frameTime(frames[0]), t1 = frameTime(frames[frames.length - 1]);
  const pad = Math.max(60000, (t1 - t0) * 0.03);
  const width = Math.max(320, charts.clientWidth || 900);
  TIMELINE_METRICS.forEach(([key, label, digits], i) => {
    charts.append(timelineChart(frames, ids, key, label, digits, t0 - pad, t1 + pad, width,
                                i === TIMELINE_METRICS.length - 1));
  });
}

function timelineMarker(filt, x, y) {
  const attrs = {fill: filterColor(filt), stroke: 'var(--panel)', 'stroke-width': 2, class: 'tl-mark'};
  if (NARROWBAND.has(filt)) {
    return svgEl('path', Object.assign(attrs, {d: `M${x},${y - 5.5}L${x + 5.5},${y}L${x},${y + 5.5}L${x - 5.5},${y}Z`}));
  }
  return svgEl('circle', Object.assign(attrs, {cx: x, cy: y, r: 4.5}));
}

function timelineChart(frames, ids, key, label, digits, tmin, tmax, width, withTimeAxis) {
  const wrap = document.createElement('div');
  wrap.className = 'tl-chart';
  const title = document.createElement('div');
  title.className = 'tl-chart-title';
  title.textContent = label;
  wrap.append(title);
  const pts = frames.filter(f => numeric(f[key]));
  const H = withTimeAxis ? 150 : 128, M = {l: 52, r: 12, t: 8, b: withTimeAxis ? 26 : 8};
  const svg = svgEl('svg', {width: width, height: H, class: 'tl-svg'});
  wrap.append(svg);
  if (!pts.length) {
    const none = svgEl('text', {x: M.l, y: 30, class: 'tl-axis'});
    none.textContent = 'not measured';
    svg.append(none);
    return wrap;
  }
  let lo = Math.min(...pts.map(f => f[key])), hi = Math.max(...pts.map(f => f[key]));
  if (key === 'num_stars') lo = 0;
  const span = (hi - lo) || Math.abs(hi) || 1;
  lo = key === 'num_stars' ? 0 : lo - span * 0.08;
  hi += span * 0.08;
  const x = t => M.l + (t - tmin) / (tmax - tmin) * (width - M.l - M.r);
  const y = v => M.t + (1 - (v - lo) / (hi - lo)) * (H - M.t - M.b);

  const ticks = niceTicks(lo, hi, 3);
  for (const v of ticks) {
    svg.append(svgEl('line', {x1: M.l, x2: width - M.r, y1: y(v), y2: y(v), class: 'tl-grid'}));
    const t = svgEl('text', {x: M.l - 6, y: y(v) + 3.5, 'text-anchor': 'end', class: 'tl-axis'});
    t.textContent = fmtNum(v, Math.min(digits + 1, ticks.decimals));
    svg.append(t);
  }
  if (withTimeAxis) {
    const hour = 3600000;
    const step = (tmax - tmin) > 8 * hour ? 2 * hour : ((tmax - tmin) > 3 * hour ? hour : hour / 4);
    for (let t = Math.ceil(tmin / step) * step; t <= tmax; t += step) {
      const lab = svgEl('text', {x: x(t), y: H - 8, 'text-anchor': 'middle', class: 'tl-axis'});
      lab.textContent = fmtClock(t);
      svg.append(lab);
    }
  }
  // A thin line per filter, in time order, so each filter's trend reads through the gaps
  // where the sequence switched to another filter.
  const byFilter = new Map();
  for (const f of pts) {
    const k = f.filter || '?';
    if (!byFilter.has(k)) byFilter.set(k, []);
    byFilter.get(k).push(f);
  }
  // Broken where the sequence paused (a gap of several frame intervals): a line across
  // an hour of clouds or a meridian flip would draw data that was never taken.
  const gaps = [];
  for (let i = 1; i < frames.length; i++) gaps.push(frameTime(frames[i]) - frameTime(frames[i - 1]));
  const typical = median(gaps.filter(g => g > 0)) || 0;
  const maxGap = Math.max(3 * typical, 5 * 60000);
  for (const [filt, list] of byFilter) {
    let run = [];
    const flush = () => {
      if (run.length > 1) {
        svg.append(svgEl('polyline', {
          points: run.map(f => x(frameTime(f)).toFixed(1) + ',' + y(f[key]).toFixed(1)).join(' '),
          fill: 'none', stroke: filterColor(filt), 'stroke-width': 1.5, 'stroke-opacity': 0.45}));
      }
      run = [];
    };
    for (const f of list) {
      if (run.length && frameTime(f) - frameTime(run[run.length - 1]) > maxGap) flush();
      run.push(f);
    }
    flush();
  }
  const tip = document.getElementById('timeline-tip');
  const modal = document.getElementById('timeline-modal');
  for (const f of pts) {
    const cx = x(frameTime(f)), cy = y(f[key]);
    const g = svgEl('g', {class: 'tl-point', tabindex: 0});
    // The hit target is bigger than the mark: a 9 px marker is hard to land on.
    g.append(svgEl('circle', {cx: cx, cy: cy, r: 12, fill: 'transparent'}),
             timelineMarker(f.filter || '?', cx, cy));
    const show = (e) => {
      tip.replaceChildren();
      const v = document.createElement('b');
      v.textContent = fmtNum(f[key], digits);
      tip.append(v, ' ' + label + ' · ' + (f.filter || '?') + ' · ' + fmtClock(frameTime(f)),
                 document.createElement('br'), f.name);
      tip.hidden = false;
      const box = modal.getBoundingClientRect();
      const at = (e && e.clientX !== undefined) ? {x: e.clientX, y: e.clientY}
        : (() => { const r = g.getBoundingClientRect(); return {x: r.right, y: r.bottom}; })();
      tip.style.left = Math.min(box.width - 260, at.x - box.left + modal.scrollLeft + 14) + 'px';
      tip.style.top = (at.y - box.top + modal.scrollTop + 14) + 'px';
    };
    g.addEventListener('pointermove', show);
    g.addEventListener('focus', show);
    g.addEventListener('pointerleave', () => { tip.hidden = true; });
    g.addEventListener('blur', () => { tip.hidden = true; });
    g.addEventListener('click', () => { closeTimeline(); openModal(f.id, ids); });
    g.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') { closeTimeline(); openModal(f.id, ids); }
    });
    svg.append(g);
  }
  return wrap;
}

document.getElementById('timeline-overlay').addEventListener('click', (e) => {
  if (e.target.id === 'timeline-overlay') closeTimeline();
});
