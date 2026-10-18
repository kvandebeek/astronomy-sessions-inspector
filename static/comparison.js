// The static report's comparison matrix. Expects COMP and COMP_METRICS, which the report
// embeds before this script; see build_comparison_section() in fits_light_stats.py.
/* global COMP, COMP_METRICS, setRowTotal */
let matrixMetric = COMP_METRICS[0][0];

function fmtMetric(v, digits, pct) {
  if (v === null || v === undefined || Number.isNaN(v)) return '&mdash;';
  const opts = {minimumFractionDigits: digits, maximumFractionDigits: digits};
  return pct ? (v * 100).toLocaleString(undefined, opts) + '%' : Number(v).toLocaleString(undefined, opts);
}

function renderMatrix() {
  const [, , digits, pct] = COMP_METRICS.find(m => m[0] === matrixMetric) || COMP_METRICS[0];
  document.querySelectorAll('.matrix-cell').forEach(cell => {
    const byFilter = (COMP[cell.dataset.panel] || {})[cell.dataset.session] || {};
    const agg = (byFilter[cell.dataset.filter] || {})[cell.dataset.exposure];
    cell.innerHTML = agg
      ? '<span class="cell-val">' + fmtMetric(agg[matrixMetric], digits, pct) + '</span>'
      : '<span class="empty">&mdash;</span>';
  });
  renderMatrixTotals();
}

function renderMatrixTotals() {
  // One row is one panel+filter at one exposure; the total is that row read across
  // every session column.
  document.querySelectorAll('.matrix-total[data-total]').forEach(cell => {
    const bySession = COMP[cell.dataset.panel] || {};
    let frames = 0, minutes = 0, any = false;
    for (const session of Object.keys(bySession)) {
      const agg = ((bySession[session] || {})[cell.dataset.filter] || {})[cell.dataset.exposure];
      if (!agg) continue;
      any = true;
      if (typeof agg.frames === 'number') frames += agg.frames;
      if (typeof agg.total_exp_min === 'number') minutes += agg.total_exp_min;
    }
    setRowTotal(cell, cell.dataset.total, frames, minutes, any);
  });
}

function setMatrixMetric(key) {
  matrixMetric = key;
  document.querySelectorAll('.metric-btn').forEach(b => b.classList.toggle('active', b.dataset.metric === key));
  renderMatrix();
}

renderMatrix();
