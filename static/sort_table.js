(function () {
  // An em-dash is how every formatter here writes "no value", so it is missing data
  // rather than text. Counting it as text made any metric column with one unmeasured
  // frame sort as strings, putting "11.40" above "2.05" in HFR, ADU, quality and the
  // rest. Blank cells were already treated this way; the dash just never was.
  var MISSING = /^[\s\u2013\u2014-]*$/;
  // The whole cell has to read as a number ("1,234.5", "+12.3", "85.0%"). Stripping every
  // non-digit and parsing what was left turned a filename like "Light_M31_30.00s_0001.fits"
  // into 30, so the File column sorted as numbers that were all the same.
  var NUMBER = /^[+\-\u2212]?[\d.,]*\d\s*%?$/;

  function cellSortValue(td) {
    var raw = td.textContent.trim();
    if (MISSING.test(raw)) return { raw: raw, num: null, missing: true };
    if (!NUMBER.test(raw)) return { raw: raw, num: null, missing: false };
    var num = parseFloat(raw.replace(/\u2212/, '-').replace(/[^0-9.\-]+/g, ''));
    return { raw: raw, num: isNaN(num) ? null : num, missing: false };
  }

  function columnIsNumeric(rows, idx) {
    var seen = 0, numeric = 0;
    for (var i = 0; i < rows.length; i++) {
      var cell = rows[i].cells[idx];
      if (!cell) continue;
      var v = cellSortValue(cell);
      if (v.missing) continue;
      seen++;
      if (v.num !== null) numeric++;
    }
    return seen > 0 && numeric === seen;
  }

  function sortTable(table, th) {
    var idx = th.cellIndex;
    var tbody = table.tBodies[0];
    var rows = Array.prototype.slice.call(tbody.rows);
    var errorRows = rows.filter(function (r) { return r.cells.length <= idx || r.querySelector('.err'); });
    var dataRows = rows.filter(function (r) { return errorRows.indexOf(r) === -1; });

    var asc = th.classList.contains('sort-asc') ? false : true;
    Array.prototype.forEach.call(table.querySelectorAll('th.sortable'), function (h) {
      h.classList.remove('sort-asc', 'sort-desc');
    });
    th.classList.add(asc ? 'sort-asc' : 'sort-desc');

    var numeric = columnIsNumeric(dataRows, idx);
    dataRows.sort(function (a, b) {
      var av = cellSortValue(a.cells[idx]);
      var bv = cellSortValue(b.cells[idx]);
      // Missing values go last in either direction. As -Infinity they compared NaN against
      // each other (-Infinity - -Infinity), which leaves the sort order undefined.
      if (av.missing || bv.missing) return av.missing === bv.missing ? 0 : (av.missing ? 1 : -1);
      var cmp;
      if (numeric) {
        cmp = av.num - bv.num;
      } else {
        // numeric collation: "session 2" before "session 10", "_0002" before "_0010".
        cmp = av.raw.localeCompare(bv.raw, undefined, { numeric: true, sensitivity: 'base' });
      }
      return asc ? cmp : -cmp;
    });

    dataRows.concat(errorRows).forEach(function (r) { tbody.appendChild(r); });
  }

  document.querySelectorAll('table.sortable-table').forEach(function (table) {
    table.querySelectorAll('th.sortable').forEach(function (th) {
      th.addEventListener('click', function () { sortTable(table, th); });
    });
  });
})();
