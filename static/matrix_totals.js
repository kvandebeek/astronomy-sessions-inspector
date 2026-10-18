function splitExposure(minutes) {
  if (minutes === null || minutes === undefined || Number.isNaN(minutes)) return null;
  let h = Math.floor(minutes / 60);
  let m = Math.round(minutes - h * 60);
  // A value like 119.7 min rounds to 60 on the minute part; carry it rather than show "1h 60m".
  if (m === 60) { h += 1; m = 0; }
  return {h: h, m: m};
}

function fmtTotalExposure(minutes) {
  const t = splitExposure(minutes);
  if (!t) return '\u2014';
  return t.h ? t.h + 'h ' + t.m + 'm' : t.m + 'm';
}

// "20m" and "1h 9m" right-aligned as one string still do not line up, because they are a
// different number of tokens. Each part gets its own fixed-width slot instead, so the
// minutes share an edge down the column whether or not there is an hour beside them.
function setRowTotal(cell, kind, frames, minutes, any) {
  cell.replaceChildren();
  if (!any) { cell.textContent = '\u2014'; return; }
  if (kind === 'frames') { cell.textContent = String(frames); return; }
  const t = splitExposure(minutes);
  if (!t) { cell.textContent = '\u2014'; return; }
  const hours = document.createElement('span');
  hours.className = 'exp-h';
  hours.textContent = t.h ? t.h + 'h' : '';
  const mins = document.createElement('span');
  mins.className = 'exp-m';
  mins.textContent = t.m + 'm';
  cell.append(hours, mins);
}

function rowTotalKey(kind, frames, minutes, any) {
  return !any ? '-' : (kind === 'frames' ? String(frames) : fmtTotalExposure(minutes));
}
