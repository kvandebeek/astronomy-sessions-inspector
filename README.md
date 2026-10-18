# astronomy-sessions-inspector

Scripts for quality-checking astrophotography LIGHT frames across one or more
imaging sessions: per-frame star/background statistics, calibration (bias +
dark + flat), an HTML report, and an interactive local review server for
inspecting and discarding bad frames.

## Layout this expects

```
<root>/
  darks/                                   master bias + dark XISF files
    masterBias_BIN-1_3008x3008.xisf
    masterDark_BIN-1_3008x3008_EXPOSURE-30.00s.xisf
    ...
  <session>/<target>/<filter>/LIGHT/...    light frames
  <session>/<target>/<filter>/FLAT/...     flats (see organize_flats.py)
  <session>/.../DARKFLAT/...               dark flats (optional)
```

The root may also be a single session (`<target>/<filter>/LIGHT`) or sit above
the sessions (`<year>/<session>/...`); the two folders above `LIGHT` are always
read as target and filter.

Master bias and darks live together in one folder and are told apart by
filename (`masterBias*.xisf` vs `masterDark*_EXPOSURE-<n>s.xisf`). A dark is
matched to a light by exact exposure and must agree with it on gain, offset
and binning wherever both say (from the master's FITS keywords or file name,
e.g. `GAIN-100`, `BIN-1`); a mismatched master is skipped and the reason shown
in the report's Cal column. Every master bias is used: each light gets the one
matching its settings. Uncompressed and zlib-compressed XISF are read directly;
lz4/zstd need the optional `lz4`/`zstandard` packages.

## Scripts

### `fits_light_stats.py`
Scans a root folder tree for `LIGHT` subfolders, reads every `.fit`/`.fits`
file, calibrates it (bias/dark + flat, unless `--no-calibrate`), computes ADU
statistics plus star-based quality metrics (FWHM, HFR, star count, roundness,
orientation — fit per star via batched Levenberg-Marquardt, not just image
moments), flags satellite trails, scores each frame against the session's own
median (`rel_stars`/`rel_fwhm`/`rel_sky`), and writes an HTML report plus a
JSON cache.

```
python3 fits_light_stats.py [root] -o report.html -j cache.json --darks darks
```

- **Rescans are incremental.** `--rescan` re-measures only frames whose file,
  masters or options changed; `--full-rescan` re-measures everything.
- **Interrupting is safe.** Progress is checkpointed to `<cache>.partial` every
  minute and on Ctrl+C; running the same command again continues from there.
- **Plate solving** with ASTAP: `--solve` during a scan, or `--solve-only` to
  add solutions to an existing cache without re-measuring (`--resolve` redoes
  frames that already have one). `--write-wcs` also writes the WCS into each
  frame's header, keeping the original header in `<root>/.wcs_backup`;
  `--undo-wcs` puts them all back.
- `--save-calibrated [dir]` writes calibrated float32 FITS alongside; a copy is
  rewritten whenever its source or masters change.
- `--full-scale ADU` for 12/14-bit data not scaled to 16 bit (default: the
  `SATURATE` card, else 65535).

Quality score: fixed thresholds plus the frame against its own set. A frame
with a satellite trail loses 15 points and is never rated better than "warn".
HFR/FWHM thresholds are in pixels; set `FITS_QUALITY_REFERENCE_SCALE` (the
arcsec/px they were tuned at) to score on angular size instead, so they carry
over to another camera or telescope.

Master calibration frames are cached in `<root>/.calibration`; masters a run no
longer uses (e.g. a master flat rebuilt after adding flats) are removed at the
end of each calibrated run.

#### What the JSON holds per frame
Besides the measurements: `calibration` and `calibration_warnings`,
`settings` (gain/offset/binning/temperature), `pixel_scale_arcsec`,
`hfr_arcsec`/`fwhm_arcsec`, the plate solve (`solved_ra`, `solved_dec`,
`pointing_error_arcsec`, ...), `object` and the mount's `mount_ra`/`mount_dec`,
and a **`wcs`** block for mosaic planning:

| key | |
|---|---|
| `cards` | every WCS card verbatim, SIP distortion included — `astropy.wcs.WCS(cards)` rebuilds the exact solution |
| `center_ra`, `center_dec` | field centre, degrees |
| `corners` | RA/Dec of pixels (0,0), (w-1,0), (w-1,h-1), (0,h-1) |
| `pixel_scale_arcsec`, `field_deg` | per axis |
| `rotation_deg` | position angle of the image's +y axis, east of north |
| `flipped` | true when the image is mirrored against the sky |
| `source` | `astap` (plate solve) or `header` (written by the capture software) |

### `fits_review_server.py`
Serves the same per-folder tables as a local, interactive web UI
(`http://127.0.0.1:8765` by default). Each row is clickable and opens a
live-rendered preview (linear or auto-stretch, 1:1 zoom, histogram levels).
Deleting moves the file into a `_deleted/` folder next to the report
(mirroring its relative path) instead of erasing it, so it's recoverable.

- In the preview, ←/→ step through the frames in the order the table or grid
  shows them; **Del** moves the frame to `_deleted` and steps on.
- Select frames by verdict (reject / warn / trail / sigma-clipped), delete or
  **restore** a whole selection, hide deleted frames, and — with a typed
  confirmation — permanently empty `_deleted`.
- Export a CSV of every frame, or a reject list (paths of deleted + selected
  frames) for a stacker's exclusion list.
- **Nights**: a per-session timeline of FWHM, sky brightness and star count.
- Picks up a rescan automatically (the page offers a reload); frames the scan
  could not read are listed at the top.

Only binds to localhost by default and only ever touches files that were part
of the scanned dataset. `--host 0.0.0.0` (with `--allow-host NAME` for the
names you browse to it by) serves it to the network — anyone who can reach it
can delete frames. `--open` opens the browser.

```
python3 fits_review_server.py -j fits_light_data.json -r . -p 8765 [--open]
```

### `organize_flats.py`
Moves flats shot into one dumping folder into the per-filter `FLAT/` layout
`fits_light_stats.py` expects, matching session/target/filter depth to how
the lights are already laid out. Dry-run by default; `--apply` moves files
(also across drives, never overwriting) and logs every move so `--undo <log>`
can reverse it.

```
python3 organize_flats.py [root]          # preview
python3 organize_flats.py [root] --apply  # do it
```

### `run_light_stats.sh` / `run_review_server.sh` (and `.bat` on Windows)
Convenience wrappers meant to live next to a project's own `darks/` and
session folders. They auto-detect a sibling `../.venv` for the Python
interpreter, and `run_review_server` runs the scan first if no JSON cache
exists yet (or with `--rescan` / `--full-rescan`, anywhere in its arguments).

```
./run_light_stats.sh [--rescan] [--solve] ...
./run_review_server.sh [--rescan] [-p 9000] [--open]
```

## Setup

Python 3.11 or newer.

```
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

(The run scripts expect this venv one directory above the project folder, at
`../.venv`.)

## Development

```
pip install -r requirements-dev.txt
ruff check .
python -m pytest            # synthetic data, ~20 s
npx eslint@9 static         # the pages' JavaScript
```

The pages' JavaScript and CSS live in `static/` and are inlined into the
report and the review page. CI runs all three checks on every push.
