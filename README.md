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
```

Master bias and darks live together in one folder and are told apart by
filename (`masterBias*.xisf` vs `masterDark*_EXPOSURE-<n>s.xisf`); darks are
matched to a light frame by exact exposure, falling back to the bias.

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

Also supports `--solve` (plate solve via ASTAP, recording pointing error),
`--write-wcs`, and `--save-calibrated [dir]` to dump calibrated float32 FITS
alongside the originals. `--solve-only` plate solves the frames already in the
JSON cache and adds the results to it without re-measuring anything (frames
that already have a solution are skipped unless `--resolve` is given):

```
python3 fits_light_stats.py -j cache.json --solve-only [--write-wcs]
```

A frame with a satellite trail loses 15 quality points and is never rated
better than "warn". Master calibration frames are cached in `<root>/.calibration`;
masters a run no longer uses (e.g. a master flat rebuilt after adding flats)
are removed at the end of each calibrated run.

Requires: `numpy`, `astropy`, `photutils`, `scipy`, `scikit-image`.

### `fits_review_server.py`
Serves the same per-folder tables as a local, interactive web UI
(`http://127.0.0.1:8765` by default). Each row is clickable and opens a
live-rendered preview (linear or auto-stretch); you can delete a frame
straight from there. Deleting moves the file into a `_deleted/` folder next
to the report (mirroring its relative path) instead of erasing it, so it's
recoverable. Only binds to localhost and only ever touches files that were
part of the scanned dataset. Additionally requires `pillow`.

```
python3 fits_review_server.py -j fits_light_data.json -r . -p 8765
```

### `organize_flats.py`
Moves flats shot into one dumping folder into the per-filter `FLAT/` layout
`fits_light_stats.py` expects, matching session/target/filter depth to how
the lights are already laid out. Dry-run by default; `--apply` moves files
and logs every move so `--undo <log>` can reverse it.

```
python3 organize_flats.py [root]          # preview
python3 organize_flats.py [root] --apply  # do it
```

### `run_light_stats.sh` / `run_review_server.sh`
Convenience wrappers meant to live next to a project's own `darks/` and
session folders. They auto-detect a sibling `../.venv` for the Python
interpreter, and `run_review_server.sh` runs the scan first if no JSON cache
exists yet (or always, with `--rescan`).

```
./run_light_stats.sh [--rescan] [--solve] ...
./run_review_server.sh [--rescan] [-p 9000]
```

## Setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install numpy astropy photutils scipy scikit-image pillow
```

(`run_light_stats.sh`/`run_review_server.sh` expect this venv one directory
above the project folder, at `../.venv`.)
