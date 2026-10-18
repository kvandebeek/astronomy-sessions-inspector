#!/usr/bin/env python3
"""
FITS LIGHT-frame quality report.

Scans a root folder tree for "LIGHT" subfolders (case-insensitive), reads every
.fit/.fits file found inside each one, computes basic ADU statistics plus
star-based quality metrics (HFR, FWHM, star count, roundness), and writes
an HTML report with one table per LIGHT folder.

Usage:
    python3 fits_light_stats.py [root_dir] [-o output.html] [-j cache.json] [-w workers]

Requires: numpy, astropy, photutils, scipy, scikit-image
    (pip install numpy astropy photutils scipy scikit-image)
"""
import argparse
import hashlib
import html
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import sys
import time
from typing import Any
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.segmentation import make_2dgaussian_kernel, detect_sources, SourceCatalog
from scipy import ndimage
from skimage.measure import block_reduce
from skimage.transform import hough_line

FITS_EXTS = {".fit", ".fits", ".fts"}
EXCLUDED_DIR_NAMES = {"_deleted", ".calibration", ".wcs_backup", "calibrated"}
BIAS_ADU = 2796.0        # fallback bias pedestal for signal estimates when no master bias is known
SATURATION_ADU = 60000    # peak ADU above which a star is considered saturated and skipped
SATURATION_FRAC = SATURATION_ADU / 65535.0  # the same, as a fraction of the sensor's full scale
# Bumped whenever a change here alters any measured number: rows from an older version are
# then re-measured by an incremental rescan instead of being kept.
MEASURE_VERSION = 2
WORKER_BYTES_PER_PIXEL = 40   # a scan worker's peak memory per frame pixel (see process_paths)
CHECKPOINT_INTERVAL_S = 60    # measured rows are flushed to the checkpoint this often
FWHM_GUESS_PX = 4.0       # matched-filter kernel FWHM used for source detection
DETECT_SIGMA = 5.0        # detection threshold = DETECT_SIGMA * background sigma
MIN_NPIXELS = 9           # minimum connected above-threshold pixels to count as a source
MIN_SEP_FROM_EDGE = 15    # exclude sources whose centroid is closer than this to the frame border
TRAIL_MASK_SIGMA = 3.0      # per-pixel threshold (x background sigma) for the trail candidate mask
TRAIL_STAR_DILATE_ITERS = 6  # dilation applied to the star segmentation before excluding it from the mask
TRAIL_HOUGH_DOWNSAMPLE = 4   # candidate mask is pooled by this factor before the Hough transform
TRAIL_MIN_BLOCK_HITS = 2     # mask pixels needed in a pooled block for it to vote
TRAIL_HOUGH_N_THETA = 1200   # angle steps over 180 deg; 0.15 deg keeps a frame-long line in one bin
TRAIL_MIN_CHORD_FRAC = 0.55  # ignore lines crossing less than this fraction of the short frame axis
TRAIL_N_CANDIDATES = 6       # leading accumulator peaks re-measured on the pixels
TRAIL_PEAK_RHO_PAD = 12      # accumulator bins suppressed around a peak before taking the next
TRAIL_PEAK_THETA_PAD = 25
TRAIL_LINEAR_MIN_EXTENT_PX = 100  # a source at least this long, and mostly empty inside its bounding
TRAIL_LINEAR_MAX_FILL = 0.25      # box, is line-like: kept in the mask instead of blanked as a star
TRAIL_DETECTOR_LINE_FRAC = 0.25      # a column/row this full of mask pixels, whose neighbours are
TRAIL_DETECTOR_LINE_ISOLATION = 0.25  # near-empty, is a hot column/row rather than a trail
TRAIL_PROFILE_HALF_PX = 22   # perpendicular half-window sampled across a candidate line
TRAIL_PROFILE_WING_PX = 14   # |offset| >= this sets the local baseline of the cross-section
TRAIL_PROFILE_CORE_PX = 5    # the ridge peak is looked for within this many px of the line
TRAIL_PROFILE_MIN_SAMPLES = 50   # too few samples along the line to average meaningfully
TRAIL_MIN_LINE_SAMPLES = 200  # too little of the line inside the frame to average over
TRAIL_MIN_RIDGE_SIGMA = 20.0  # cross-section ridge significance needed to confirm a candidate;
#                               over 637 real subs, clean frames top out at 13.7 and the faintest
#                               real trail scores 35.4, so this sits mid-gap
TRAIL_MIN_WIDTH_PX = 2.0      # a ridge narrower than this is a detector artefact, not the sky
TRAIL_MIN_WIDTH_PSF_FRAC = 0.55  # ...and no trail is much narrower than the frame's own PSF
TRAIL_MAX_WIDTH_PX = 14.0     # a ridge broader than this is nebulosity, not a trail
TRAIL_EDGE_MARGIN_PX = 20     # a line running this close to the frame border, along its whole
#                               length, is a sensor edge artefact (amp glow, overscan rolloff),
#                               not a trail: a real one crosses the field, it doesn't hug a side
TRAIL_EXTENT_SMOOTH_PX = 31   # along-line smoothing used to find where a confirmed trail starts and
TRAIL_EXTENT_SIGMA = 3.0      # ends; the lit run is what the review server draws a box around
HOT_PIXEL_SECOND_PEAK_FRAC = 0.2  # a source whose 2nd-highest raw pixel is below this fraction of
#                                    its peak is a single-pixel spike (hot/warm pixel), not a star
ELONGATION_ROUND_THRESH = 0.8  # roundness below this counts as "elongated" for direction stats
MIN_ELONGATED_FOR_DIRECTION = 5  # need at least this many elongated stars for a direction verdict
STATS_SUBSAMPLE_TARGET_PX = 1_000_000  # background level/noise are fitted to at most this many pixels
PSF_FIT_HALF_PX = 7          # star shapes are fitted on (2*half+1)^2 stamps around each centroid
PSF_FIT_MAX_STARS = 300      # brightest unsaturated stars fitted per frame - plenty for a median
PSF_FIT_MIN_PEAK_SIGMA = 15.0  # fainter stars' shapes are noise-dominated; they still count as stars
PSF_FIT_MIN_STARS = 5        # fewer good fits than this -> fall back to (unconvolved) moments
PSF_FIT_ITERS = 30           # Levenberg-Marquardt iterations, run for all stars at once

# --- Quality score ------------------------------------------------------------------------
# Fixed thresholds, split narrowband (H/O/S: intrinsically fewer/softer stars) vs
# broadband (L/R/G/B), since a star count or HFR that is fine on one is not on the other.
NARROWBAND_FILTERS = {"H", "O", "S"}

QUALITY_THRESHOLDS = {
    # metric: (good, bad, max_penalty) - linear ramp from 0 penalty at `good` to
    # `max_penalty` at `bad` and beyond. Values are illustrative starting points;
    # tune to taste once scored frames can be eyeballed against these numbers.
    "hfr": (2.5, 6.0, 30),
    "fwhm": (3.5, 8.0, 20),
    # inverted: penalty grows as roundness falls. Read off fitted PSFs, which on this data sit
    # around 0.75-0.85 on good subs; the old convolved moments read 0.90-0.97 for the same stars.
    "roundness": (0.8, 0.5, 15),
    "elong": (0.35, 0.75, 15),      # max(uniform_r, radial_r); penalty grows as it rises
}
QUALITY_STARS = {
    # narrowband: (good, bad, max_penalty), broadband: (good, bad, max_penalty)
    True: (15, 3, 25),
    False: (120, 15, 25),
}
# Relative to the frame's own set (same panel, filter and exposure, every session): fixed
# thresholds cannot see a sub that lost half its stars to haze but still clears the absolute
# star count, or one shot into moonlight. Each is the ratio to the set's median.
#   metric: (good, bad, max_penalty); a set smaller than QUALITY_RELATIVE_MIN_SET is not trusted
QUALITY_RELATIVE = {
    "rel_stars": (0.75, 0.35, 25),   # transparency: clouds, haze, dew
    "rel_fwhm": (1.15, 1.6, 20),     # seeing, focus drift
    "rel_sky": (1.3, 3.0, 15),       # sky brightness: moon, twilight, lit clouds
}
QUALITY_RELATIVE_MIN_SET = 5
# The pixel scale (arcsec/px) the HFR/FWHM thresholds above were tuned at. Set it - here or
# through the FITS_QUALITY_REFERENCE_SCALE environment variable - and frames whose scale is
# known (plate solve, or FOCALLEN + XPIXSZ) are scored on angular size instead, so the
# thresholds carry over to another camera or telescope. Unset, they stay plain pixels.
REFERENCE_PIXEL_SCALE_ARCSEC = float(os.environ.get("FITS_QUALITY_REFERENCE_SCALE") or 0) or None
QUALITY_WARN_MAX = 70    # score < this -> "warn"
QUALITY_REJECT_MAX = 40  # score < this -> "reject"
# A trail is one line through an otherwise fine frame, and a stacker's pixel rejection
# usually removes it - so it costs a fixed amount rather than sinking the frame. It does
# cap the verdict at "warn", though: green in the review grid should mean nothing to look at.
QUALITY_TRAIL_PENALTY = 15


def _ramp_penalty(value, good, bad, max_penalty):
    """0 at `good`, `max_penalty` at/beyond `bad`; `good`/`bad` order sets the direction."""
    span = bad - good
    if span == 0:
        return 0.0
    frac = (value - good) / span
    return max_penalty * min(1.0, max(0.0, frac))


def compute_quality(r):
    """Composite 0-100 quality score from the metrics already measured for a frame, plus
    the reasons behind any deduction. Frames with too little data to score (e.g. zero
    stars detected) fall through to a low score with an explicit reason rather than a
    misleadingly neutral one."""
    hfr, fwhm = r.get("hfr_avg"), r.get("fwhm_avg")
    # Scored in "reference pixels" when the frame's own scale is known (see above); the
    # reasons still quote the frame's own pixels, which is what its numbers are in.
    scale = r.get("solved_scale_arcsec") or r.get("pixel_scale_arcsec")
    to_ref = (scale / REFERENCE_PIXEL_SCALE_ARCSEC) if REFERENCE_PIXEL_SCALE_ARCSEC and scale else 1.0
    stars, roundness = r.get("num_stars"), r.get("roundness_avg")
    uniform_r, radial_r = r.get("elong_uniform_r"), r.get("elong_radial_r")
    trail = bool(r.get("has_trail"))
    narrowband = (r.get("filter") or "").strip().upper() in NARROWBAND_FILTERS

    score, reasons = 100.0, []

    if hfr is not None:
        good, bad, mx = QUALITY_THRESHOLDS["hfr"]
        p = _ramp_penalty(hfr * to_ref, good, bad, mx)
        score -= p
        if p >= mx * 0.3:
            reasons.append(f"high HFR ({hfr:.1f}px)")

    if fwhm is not None:
        good, bad, mx = QUALITY_THRESHOLDS["fwhm"]
        p = _ramp_penalty(fwhm * to_ref, good, bad, mx)
        score -= p
        if p >= mx * 0.3:
            reasons.append(f"high FWHM ({fwhm:.1f}px)")

    if roundness is not None:
        good, bad, mx = QUALITY_THRESHOLDS["roundness"]
        p = _ramp_penalty(roundness, good, bad, mx)  # good > bad: ramp runs downward
        score -= p
        if p >= mx * 0.3:
            reasons.append(f"elongated stars ({roundness * 100:.0f}% round)")

    if stars is not None:
        good, bad, mx = QUALITY_STARS[narrowband]
        p = _ramp_penalty(stars, good, bad, mx)
        score -= p
        if p >= mx * 0.3:
            reasons.append(f"few stars ({int(stars)})")

    if uniform_r is not None or radial_r is not None:
        worst = max(uniform_r or 0.0, radial_r or 0.0)
        good, bad, mx = QUALITY_THRESHOLDS["elong"]
        p = _ramp_penalty(worst, good, bad, mx)
        score -= p
        if p >= mx * 0.3:
            reasons.append("tracking drift" if (uniform_r or 0) >= (radial_r or 0) else "coma/tilt")

    for key, describe in (
            ("rel_stars", lambda v: f"star count {v * 100:.0f}% of set (clouds/haze?)"),
            ("rel_fwhm", lambda v: f"FWHM {v:.2f}x set (seeing/focus?)"),
            ("rel_sky", lambda v: f"sky {v:.1f}x set (moon/twilight/cloud glow?)")):
        value = r.get(key)
        if value is None:
            continue
        good, bad, mx = QUALITY_RELATIVE[key]
        p = _ramp_penalty(value, good, bad, mx)
        score -= p
        if p >= mx * 0.3:
            reasons.append(describe(value))

    if trail:
        score -= QUALITY_TRAIL_PENALTY
        reasons.insert(0, "satellite trail")

    if stars == 0 or (hfr is None and fwhm is None and stars is None):
        score = min(score, QUALITY_REJECT_MAX - 5)
        reasons.insert(0, "no stars detected")

    score = max(0.0, min(100.0, score))
    verdict = "reject" if score < QUALITY_REJECT_MAX else "warn" if score < QUALITY_WARN_MAX else "good"
    if trail and verdict == "good":
        verdict = "warn"
    return score, reasons, verdict

_HOUGH_NORM_CACHE = {}   # (pooled shape, n_theta) -> per-bin pixel counts, built once

EM_DASH = "&mdash;"


def _enclosing_light_dir(root: Path, directory: Path):
    """The outermost "LIGHT" ancestor of `directory` (inclusive), or None if it has none."""
    parts = directory.relative_to(root).parts
    for i, part in enumerate(parts):
        if part.upper() == "LIGHT":
            return root.joinpath(*parts[:i + 1])
    return None


def scan_light_dirs(root: Path, exclude=()):
    """One filesystem walk -> {light_dir: [fits files]}, excluded subtrees pruned.

    Files nested below a LIGHT folder (e.g. LIGHT/30.00/*.fits) are attributed to
    that LIGHT folder, matching the previous per-folder rglob behaviour.
    """
    # Resolved paths rather than names: the calibrated output mirrors the source tree, so
    # if it sits inside the root under any name but the default one, the next scan walks
    # straight into it and measures every calibrated frame a second time as a light.
    pruned = {Path(p).resolve() for p in exclude}
    found = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIR_NAMES
                       and not d.startswith(".")
                       and (Path(dirpath) / d).resolve() not in pruned]
        directory = Path(dirpath)
        light_dir = _enclosing_light_dir(root, directory)
        if light_dir is None:
            continue
        fits_files = [directory / n for n in filenames if Path(n).suffix.lower() in FITS_EXTS]
        if fits_files:
            found.setdefault(light_dir, []).extend(fits_files)
    return {d: sorted(files) for d, files in sorted(found.items())}


# --- Plate solving -------------------------------------------------------------------------
# ASTAP is used rather than a built-in solver: it ships its own compact star database, solves
# a 9 Mpx frame in about a tenth of a second, and needs no network. The field of view it is
# given comes from the frame's own optics keywords, so no pointing information is assumed.
ASTAP_CANDIDATES = (
    "/Volumes/Astro/Astap CLI/astap_cli",
    "/Applications/ASTAP.app/Contents/MacOS/astap",
    "/usr/local/bin/astap_cli", "/usr/local/bin/astap", "/opt/homebrew/bin/astap",
)
ASTAP_DB_CANDIDATES = ("/usr/local/opt/astap", "/usr/local/share/astap", "/opt/homebrew/share/astap")
ASTAP_SEARCH_RADIUS_DEG = 30.0   # generous: far beyond any plausible mount error, still fast
# ASTAP is multithreaded itself, so a few at a time already fill the machine.
ASTAP_CONCURRENCY = max(1, min(4, (os.cpu_count() or 4) // 4))
WCS_BACKUP_DIRNAME = ".wcs_backup"   # original headers, saved before --write-wcs touches them


def find_astap(explicit=None):
    """Path to the ASTAP command-line binary, or None."""
    for candidate in ([explicit] if explicit else []) + list(ASTAP_CANDIDATES):
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return shutil.which("astap_cli") or shutil.which("astap")


def find_astap_db(explicit=None):
    for candidate in ([explicit] if explicit else []) + list(ASTAP_DB_CANDIDATES):
        if candidate and Path(candidate).is_dir() and any(Path(candidate).glob("*.1476")):
            return str(candidate)
        if candidate and Path(candidate).is_dir() and any(Path(candidate).glob("*.290")):
            return str(candidate)
    return None


def frame_fov_deg(header, height):
    """Field height in degrees from the optics, not from any assumed pointing."""
    pixel_um = header.get("XPIXSZ") or header.get("PIXSIZE1")
    focal_mm = header.get("FOCALLEN")
    binning = header.get("XBINNING", 1) or 1
    if not (isinstance(pixel_um, (int, float)) and isinstance(focal_mm, (int, float)) and focal_mm > 0):
        return None
    return height * (206.265 * pixel_um * binning / focal_mm) / 3600.0


def _wcs_value(text, key):
    m = re.search(key + r"\s*=\s*([-+0-9.eE]+)", text)
    return float(m.group(1)) if m else None


def solve_frame(solver, path: Path, header, height):
    """Plate solve one frame with ASTAP.

    Returns {ra, dec, rotation, scale_arcsec, error_arcsec} or None. With
    solver["update_header"] the WCS is written into the frame's own header as well - ASTAP
    appends the cards in place and leaves the pixel data untouched.
    """
    fov = frame_fov_deg(header, height)
    if not solver or not fov:
        return None
    cmd = [solver["astap"], "-f", str(path), "-fov", f"{fov:.4f}",
           "-r", str(ASTAP_SEARCH_RADIUS_DEG)]
    if solver.get("db"):
        cmd += ["-d", solver["db"]]

    tmpdir = None
    if solver.get("update_header"):
        backup_header(solver, path, header)
        cmd += ["-update"]
    else:
        # Solve without touching the frame: send ASTAP's sidecars to a scratch directory.
        tmpdir = tempfile.mkdtemp(prefix="astap-")
        cmd += ["-o", os.path.join(tmpdir, "s"), "-wcs"]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if solver.get("update_header"):
            # Header only: the old path went through first_image_hdu, which reads the
            # whole image just to find out the HDU has one.
            solved = first_image_header(path)
            if not solved.get("PLTSOLVD"):
                return None
            ra, dec = solved.get("CRVAL1"), solved.get("CRVAL2")
            rot, cdelt = solved.get("CROTA2"), solved.get("CDELT1")
        else:
            wcs_path = os.path.join(tmpdir, "s.wcs")
            if not os.path.exists(wcs_path):
                return None
            text = open(wcs_path, "r", errors="replace").read()
            ra, dec = _wcs_value(text, "CRVAL1"), _wcs_value(text, "CRVAL2")
            rot, cdelt = _wcs_value(text, "CROTA2"), _wcs_value(text, "CDELT1")
        if ra is None or dec is None:
            return None
        result = {"ra": float(ra), "dec": float(dec),
                  "rotation": float(rot) if rot is not None else None,
                  "scale_arcsec": abs(float(cdelt)) * 3600.0 if cdelt is not None else None}
        # How far the mount's own idea of where it was pointing turned out to be off.
        hra, hdec = header.get("RA"), header.get("DEC")
        if isinstance(hra, (int, float)) and isinstance(hdec, (int, float)):
            d0 = math.radians(result["dec"])
            result["error_arcsec"] = math.hypot((result["ra"] - hra) * 3600.0 * math.cos(d0),
                                                (result["dec"] - hdec) * 3600.0)
        return result
    except (subprocess.TimeoutExpired, OSError, ValueError):
        return None
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


def _backup_path(solver, path: Path):
    root = Path(solver["root"])
    try:
        relative = Path(path).resolve().relative_to(root)
    except ValueError:
        relative = Path(hashlib.sha1(str(path).encode()).hexdigest()[:12]) / Path(path).name
    return root / WCS_BACKUP_DIRNAME / (str(relative) + ".hdr")


def backup_header(solver, path: Path, header):
    """Keep the frame's original header before ASTAP rewrites it in place.

    Only the first backup is ever kept - a re-solve must not overwrite the true original
    with an already-solved header. --undo-wcs puts them back.
    """
    if not solver.get("root"):
        return
    out = _backup_path(solver, path)
    if out.exists():
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(header.tostring(), encoding="ascii", errors="replace")
    os.replace(tmp, out)


def undo_wcs(root: Path):
    """Restore every header saved by backup_header(), leaving the pixel data alone."""
    backups = sorted((root / WCS_BACKUP_DIRNAME).rglob("*.hdr"))
    restored = 0
    for backup in backups:
        relative = backup.relative_to(root / WCS_BACKUP_DIRNAME)
        target = root / str(relative)[:-len(".hdr")]
        if not target.exists():
            print(f"  skipped (frame no longer there): {target}", file=sys.stderr)
            continue
        original = fits.Header.fromstring(backup.read_text(encoding="ascii"))
        # do_not_scale_image_data: the raw integers go back exactly as stored, with the
        # original BZERO/BSCALE, instead of being rescaled through floats on the way.
        with fits.open(str(target), mode="update", do_not_scale_image_data=True) as hdul:
            for hdu in hdul:
                if int(hdu.header.get("NAXIS", 0)) >= 2:
                    # Rewritten card by card: assigning a freshly parsed Header leaves it
                    # flagged unmodified, and flush() then silently keeps the old one.
                    hdu.header.clear()
                    hdu.header.extend(original, strip=False, unique=False)
                    break
        backup.unlink()
        restored += 1
    print(f"restored {restored} original header(s)", file=sys.stderr)


def prepare_solver(astap=None, db=None, update_header=False, progress=True, root=None):
    """Picklable solver config for the workers, or None if ASTAP is unavailable."""
    binary = find_astap(astap)
    if not binary:
        if progress:
            print("plate solving: ASTAP not found; skipping (pass --astap PATH)", file=sys.stderr)
        return None
    database = find_astap_db(db)
    if progress:
        print(f"plate solving: {binary}"
              f"{' with ' + database if database else ' (database auto-detected by ASTAP)'}"
              f"{', writing WCS into each header' if update_header else ''}", file=sys.stderr)
    return {"astap": binary, "db": database, "update_header": bool(update_header),
            "root": str(root) if root else None}


# --- Calibration ---------------------------------------------------------------------------
# Masters are prepared once in the parent process and written to .npy in 16-bit ADU scale;
# the workers memory-map them read-only, so the operating system shares one copy of each
# master across every worker instead of giving each its own ~36 MB.
CALIB_DIRNAME = ".calibration"      # cache of prepared masters, under the scan root
CALIB_INDEX_NAME = "masters.json"   # what each cached bias/dark is, for runs without --darks
XISF_SIGNATURE = b"XISF0100"
FLAT_DIR_NAME = "FLAT"
LIGHT_DIR_NAME = "LIGHT"
# Folders capture software puts dark flats in (compared upper-cased, separators ignored).
DARKFLAT_DIR_NAMES = {"DARKFLAT", "DARKFLATS", "FLATDARK", "FLATDARKS"}
# Settings that must agree between a light and the master calibrating it. Any one missing
# on either side is not held against the match - plenty of masters only carry some.
MATCH_SETTINGS = ("gain", "offset", "binning")
DARK_TEMP_TOLERANCE_C = 2.0          # sensor temperature difference worth a warning
DARKFLAT_EXPOSURE_TOLERANCE = 0.10   # a dark flat within 10% of a flat's exposure matches it
MASTER_FLAT_READ_THREADS = 4         # flats are read concurrently; the combine stays single

_MASTER_CACHE = {}   # path -> memmapped array, per worker process


def _number(value):
    """A header value as a float, or None when it is missing or not a number."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        v = float(str(value).strip().strip("'").strip())
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def frame_settings(header):
    """{gain, offset, binning, temp} of a frame or master, from whichever cards it has."""
    out = {}
    gain = _number(header.get("GAIN"))
    if gain is not None:
        out["gain"] = gain
    offset = _number(header.get("OFFSET", header.get("BLKLEVEL")))
    if offset is not None:
        out["offset"] = offset
    xb = _number(header.get("XBINNING"))
    if xb is not None:
        yb = _number(header.get("YBINNING")) or xb
        out["binning"] = f"{int(xb)}x{int(yb)}"
    temp = _number(header.get("CCD-TEMP", header.get("SET-TEMP")))
    if temp is not None:
        out["temp"] = temp
    return out


def _settings_from_name(name):
    """Settings PixInsight's WBPP and most capture tools put in a master's file name."""
    out = {}
    for key, pattern in (("gain", r"GAIN[-_]?(\d+(?:\.\d+)?)"),
                         ("offset", r"OFFSET[-_]?(\d+(?:\.\d+)?)"),
                         ("temp", r"TEMP(?:ERATURE)?[-_]?(-?\d+(?:\.\d+)?)")):
        m = re.search(pattern, name, re.IGNORECASE)
        if m:
            out[key] = float(m.group(1))
    m = re.search(r"BIN(?:NING)?[-_]?(\d+)(?:x(\d+))?", name, re.IGNORECASE)
    if m:
        out["binning"] = f"{m.group(1)}x{m.group(2) or m.group(1)}"
    return out


def settings_conflicts(master_settings, frame):
    """The MATCH_SETTINGS on which a master and a frame disagree, as readable strings."""
    conflicts = []
    for key in MATCH_SETTINGS:
        a, b = (master_settings or {}).get(key), (frame or {}).get(key)
        if a is None or b is None:
            continue
        same = a == b if key == "binning" else abs(float(a) - float(b)) < 0.5
        if not same:
            conflicts.append(f"{key} {a:g} vs {b:g}" if key != "binning" else f"{key} {a} vs {b}")
    return conflicts


def _temp_gap(master_settings, frame):
    a, b = (master_settings or {}).get("temp"), (frame or {}).get("temp")
    return None if a is None or b is None else abs(a - b)


def _xisf_header(path: Path):
    with open(path, "rb") as f:
        if f.read(8) != XISF_SIGNATURE:
            raise ValueError(f"{path.name} is not an XISF 1.0 file")
        header_len = int.from_bytes(f.read(4), "little")
        f.read(4)                       # reserved
        return f.read(header_len).decode("utf-8", "replace")


def _xml_attrs(tag):
    return dict(re.findall(r'([A-Za-z_:][\w:.-]*)="([^"]*)"', tag))


def _xisf_image_spec(path: Path):
    """Where and how the single image of a monolithic XISF file is stored.

    PixInsight writes masters as an XML header followed by the samples at a byte offset the
    header states, so a master can be read - or, uncompressed, memory-mapped - without an
    XISF library. Returns {offset, length, shape, dtype, scale, compression, keywords};
    `scale` is None for a float image that declares no bounds (see read_xisf).
    """
    xml = _xisf_header(path)
    image = re.search(r"<Image\b[^>]*>", xml)
    if image is None:
        raise ValueError(f"{path.name} contains no Image element")
    attrs = _xml_attrs(image.group(0))

    m = re.match(r"attachment:(\d+):(\d+)", attrs.get("location", ""))
    if not m:
        raise ValueError(f"{path.name} does not store its samples as an attachment")
    geometry = attrs.get("geometry", "").split(":")
    if len(geometry) < 3 or geometry[2] != "1":
        raise ValueError(f"{path.name} is not a single-channel image")
    width, height = int(geometry[0]), int(geometry[1])

    sample_format = attrs.get("sampleFormat", "")
    dtype = {"Float32": np.float32, "Float64": np.float64,
             "UInt16": np.uint16, "UInt8": np.uint8, "UInt32": np.uint32}.get(sample_format)
    if dtype is None:
        raise ValueError(f"{path.name} has an unsupported sample format {sample_format!r}")
    # Float masters are normalised to the bounds the header declares (PixInsight writes
    # 0:1); integer ones are already in native ADU.
    scale = 1.0
    if np.issubdtype(dtype, np.floating):
        bounds = attrs.get("bounds")
        scale = None
        if bounds:
            lo, hi = (float(v) for v in bounds.split(":"))
            scale = 65535.0 / (hi - lo) if hi > lo else None

    keywords = {}
    for tag in re.findall(r"<FITSKeyword\b[^>]*>", xml):
        kw = _xml_attrs(tag)
        if kw.get("name"):
            keywords[kw["name"].upper()] = kw.get("value", "").strip().strip("'").strip()
    return {"offset": int(m.group(1)), "length": int(m.group(2)), "shape": (height, width),
            "dtype": dtype, "scale": scale, "compression": attrs.get("compression"),
            "keywords": keywords}


def _xisf_decompress(raw: bytes, compression: str, name: str) -> bytes:
    """Undo an XISF block compression: zlib / lz4 / lz4hc / zstd, optionally byte-shuffled.

    zlib is in the standard library; lz4 and zstd need the `lz4` / `zstandard` packages and
    are refused with a clear message when those are not installed.
    """
    parts = compression.split(":")
    codec, size = parts[0].lower(), int(parts[1])
    shuffled = codec.endswith("+sh")
    codec = codec.removesuffix("+sh")
    if codec == "zlib":
        import zlib
        data = zlib.decompress(raw)
    elif codec in ("lz4", "lz4hc"):
        try:
            import lz4.block  # type: ignore[import-not-found]
        except ImportError:
            raise ValueError(f"{name} is lz4-compressed; pip install lz4 to read it")
        data = lz4.block.decompress(raw, uncompressed_size=size)
    elif codec == "zstd":
        try:
            import zstandard  # type: ignore[import-not-found]
        except ImportError:
            raise ValueError(f"{name} is zstd-compressed; pip install zstandard to read it")
        data = zstandard.ZstdDecompressor().decompress(raw, max_output_size=size)
    else:
        raise ValueError(f"{name} uses unsupported XISF compression {codec!r}")
    if shuffled:
        item = int(parts[2]) if len(parts) > 2 else 1
        n = len(data) // item
        body = np.frombuffer(data, dtype=np.uint8, count=n * item).reshape(item, n).T
        data = body.tobytes() + data[n * item:]
    return data


def read_xisf(path: Path) -> np.ndarray:
    """One XISF master as float32 in 16-bit ADU, matching the lights' scale."""
    spec = _xisf_image_spec(path)
    if spec["compression"]:
        with open(path, "rb") as f:
            f.seek(spec["offset"])
            raw = f.read(spec["length"])
        data = np.frombuffer(_xisf_decompress(raw, spec["compression"], path.name),
                             dtype=spec["dtype"]).reshape(spec["shape"])
    else:
        data = np.memmap(path, dtype=spec["dtype"], mode="r", offset=spec["offset"],
                         shape=spec["shape"])
    # A copy: a float32 memmap would come back as itself - read-only - from asarray.
    out = np.array(data, dtype=np.float32)
    scale = spec["scale"]
    if scale is None:
        # A float image with no declared bounds. PixInsight always writes them, so this is
        # another tool's file: read 0..1 data as normalised and anything larger as ADU,
        # and say so - silently assuming either one is off by a factor of 65535.
        normalised = float(np.nanmax(out[::8, ::8])) <= 1.5
        scale = 65535.0 if normalised else 1.0
        print(f"calibration: WARNING - {path.name} declares no sample bounds; reading it as "
              f"{'0..1 normalised' if normalised else 'ADU'} data", file=sys.stderr)
    if scale != 1.0:
        out *= np.float32(scale)
    return out


def xisf_master_settings(path: Path):
    """Settings of an XISF master, from its FITS keywords and failing that its file name."""
    settings = _settings_from_name(path.name)
    try:
        keywords = _xisf_image_spec(path)["keywords"]
    except (OSError, ValueError):
        return settings
    settings.update(frame_settings(keywords))
    return settings


def master(path):
    """A prepared master, memory-mapped and cached for the life of this process."""
    arr = _MASTER_CACHE.get(path)
    if arr is None:
        arr = _MASTER_CACHE[path] = np.load(path, mmap_mode="r")
    return arr


def session_dir_of(path: Path, marker: str):
    """The night's folder for a frame, as an absolute path.

    Both trees are laid out <session>/<group>/<filter>/<LIGHT|FLAT>/..., so the session is
    three levels above the marker folder. Derived from the absolute path rather than from
    the position within the scan root: keying on "first component below the root" silently
    stopped matching as soon as the root was a single session, because the session level
    was then above the root and lights and flats ended up under different keys.
    """
    for parent in path.parents:
        if parent.name.upper() == marker:
            parents = parent.parents
            return parents[2] if len(parents) > 2 else parent.parent
    return None


def _is_darkflat_dir(name):
    return re.sub(r"[^A-Z]", "", name.upper()) in DARKFLAT_DIR_NAMES


def scan_flat_dirs(root: Path):
    """({(session, filter): [flat files]}, {darkflat folder: [files]}).

    Flats are taken per session and per filter. The filter is the folder above FLAT
    (".../<session>/<whatever>/<filter>/FLAT/<exp>/"), which is how the capture software
    lays them out, and the session is three levels above that, so a night's flats only
    ever calibrate that night's lights. Dark flats are collected by folder name and matched
    to sessions afterwards (see _darkflat_sessions), since tools put them at varying depths.
    """
    found, darkflats = {}, {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d not in EXCLUDED_DIR_NAMES and not d.startswith(".")]
        directory = Path(dirpath)
        try:
            parts = directory.relative_to(root).parts
        except ValueError:
            continue
        files = [directory / n for n in filenames if Path(n).suffix.lower() in FITS_EXTS]
        if not files:
            continue
        darkflat_at = [i for i, p in enumerate(parts) if _is_darkflat_dir(p)]
        if darkflat_at:
            folder = root.joinpath(*parts[:darkflat_at[0] + 1])
            darkflats.setdefault(folder, []).extend(files)
            continue
        upper = [p.upper() for p in parts]
        if FLAT_DIR_NAME not in upper:
            continue
        i = upper.index(FLAT_DIR_NAME)
        if i == 0:
            continue
        session = session_dir_of(directory / "x", FLAT_DIR_NAME)
        if session is None:
            continue
        # Upper-cased, as organize_flats.py compares them: a "Ha" folder and an "HA" FILTER
        # card are one filter, and calibration_plan() folds case the same way.
        found.setdefault((str(session), parts[i - 1].upper()), []).extend(files)
    return ({k: sorted(v) for k, v in sorted(found.items())},
            {k: sorted(v) for k, v in sorted(darkflats.items())})


def _darkflat_sessions(darkflats, sessions):
    """{session or "": [files]}: each dark-flat folder goes to the flat session it sits in,
    or to "" - usable by every session - when it sits in none of them."""
    out = {}
    for folder, files in darkflats.items():
        owners = [s for s in sessions if Path(s) in folder.parents or Path(s) == folder]
        key = max(owners, key=len) if owners else ""
        out.setdefault(key, []).extend(files)
    return out


def _read_frame(path):
    """(float32 data, header) of one calibration frame."""
    with fits.open(str(path), memmap=False) as hdul:
        hdu = first_image_hdu(hdul)
        return np.asarray(hdu.data, dtype=np.float32), hdu.header.copy()


def _majority_settings(files, label):
    """The files sharing the most common gain/binning, and a warning for any left out.

    A flat set mixing gains or binnings would median-combine frames that do not share a
    pattern (and, across binnings, not even a shape), so only the larger group is used.
    """
    groups = {}
    for path in files:
        try:
            s = frame_settings(fits.getheader(str(path)))
        except OSError:
            continue
        key = (s.get("gain"), s.get("binning"))
        groups.setdefault(key, []).append(path)
    if len(groups) <= 1:
        return files, None
    best = max(groups.values(), key=len)
    dropped = len(files) - len(best)
    return best, (f"{label}: flats mix {len(groups)} gain/binning settings; "
                  f"using the {len(best)} that agree, ignoring {dropped}")


def _darkflat_groups(files):
    """{exposure: [dark-flat files]} - one master per distinct exposure."""
    by_exposure = {}
    for path in files:
        try:
            header = fits.getheader(str(path))
        except OSError:
            continue
        exp = _number(header.get("EXPTIME", header.get("EXPOSURE")))
        if exp is not None:
            by_exposure.setdefault(round(exp, 3), []).append(path)
    return dict(sorted(by_exposure.items()))


def _combine_darkflats(paths, limit=30):
    stack = np.stack([_read_frame(p)[0] for p in paths[:limit]])
    return np.median(stack, axis=0, overwrite_input=True)


def _build_master_flat(files, bias, darkflats=(), limit=40):
    """Normalised flat field from one session+filter's flats.

    Each frame has its dark flat - or, without one, the bias - subtracted and is divided by
    its own median before they are combined, because these are shot to a target level
    rather than a fixed exposure - one set here spans 0.85 s to 0.99 s - and combining them
    raw would weight the brighter ones. Frames are read on a few threads: on a first run
    building the masters is minutes of disk reads, which the combine itself never was.
    """
    step = max(1, len(files) // limit)
    chosen = files[::step][:limit]

    def load(path):
        frame, header = _read_frame(path)
        exp = _number(header.get("EXPTIME", header.get("EXPOSURE")))
        pedestal = bias
        if darkflats and exp is not None:
            exp_df, df = min(darkflats, key=lambda d: abs(d[0] - exp))
            if abs(exp_df - exp) <= DARKFLAT_EXPOSURE_TOLERANCE * max(exp, 1e-6) \
                    and df.shape == frame.shape:
                pedestal = df
        if pedestal is not None and pedestal.shape == frame.shape:
            frame -= pedestal
        level = float(np.median(frame[::2, ::2]))
        return (frame / level) if level > 0 else None

    # Filled one frame at a time as reads complete, into one preallocated cube rather than
    # a list that is then stacked - which held two full copies at once, about 2.2 GB for
    # thirty 9 Mpx flats. overwrite_input lets the median reuse the cube as well.
    cube, used = None, 0
    with ThreadPoolExecutor(max_workers=MASTER_FLAT_READ_THREADS) as pool:
        for frame in pool.map(load, chosen):
            if frame is None:
                continue
            if cube is None:
                cube = np.empty((len(chosen),) + frame.shape, dtype=np.float32)
            if frame.shape != cube.shape[1:]:
                continue
            cube[used] = frame
            used += 1
    if cube is None or used == 0:
        return None
    combined = np.median(cube[:used], axis=0, overwrite_input=True)
    del cube
    level = float(np.median(combined))
    if level <= 0:
        return None
    combined /= level
    # A flat that reaches zero anywhere would divide a light to infinity there.
    np.clip(combined, 0.05, None, out=combined)
    return combined


def _fingerprint(paths):
    """Identity of a master's inputs, so a changed input rebuilds it.

    Reusing a cached master purely because the file existed meant that shooting more
    flats for a session and rescanning silently kept the old one - the very kind of stale
    calibration that is invisible in the output.
    """
    h = hashlib.sha1()
    for q in sorted(paths):
        st_ = Path(q).stat()
        h.update(f"{q}|{st_.st_mtime_ns}|{st_.st_size}|".encode())
    return h.hexdigest()[:10]


def _name_tag(*parts):
    """A readable, collision-free file-name fragment: "night 1" and "night_1" both read
    as night_1, so a hash of the exact inputs keeps them apart."""
    readable = "_".join(re.sub(r"[^A-Za-z0-9]+", "_", str(p)) for p in parts)
    return readable + "_" + hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:6]


def prepare_calibration(root: Path, darks_dir, progress=True, calibrated_dir=None):
    """Build every master the scan will need, once, and return a picklable manifest.

    Returns {"root", "biases": [{path, settings, level}], "darks": [{path, exposure,
    settings}], "flats": {(session, filter): {path, settings}}, ...} naming .npy files the
    workers memory-map. None if there is nothing to calibrate with.
    """
    darks_dir = Path(darks_dir).expanduser().resolve() if darks_dir else None
    if darks_dir is not None and not darks_dir.is_dir():
        # A mistyped --darks used to glob an empty list and carry on, so the whole scan ran
        # bias-free with nothing on the terminal to say why.
        print(f"calibration: WARNING - darks folder {darks_dir} does not exist; "
              f"no master bias or darks will be applied", file=sys.stderr)
        darks_dir = None
    flat_sets, darkflat_dirs = scan_flat_dirs(root)
    cache = root / CALIB_DIRNAME
    # Without --darks the cache's own bias/darks from an earlier run are reused below, so
    # only give up when there is neither anything new to build nor anything cached.
    if darks_dir is None and not flat_sets and not any(cache.glob("*.npy")):
        return None

    cache.mkdir(exist_ok=True)
    manifest = {"root": str(root), "biases": [], "darks": [], "flats": {},
                "calibrated_dir": str(calibrated_dir) if calibrated_dir else None}
    index_path = cache / CALIB_INDEX_NAME
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        index = {}
    built = [0]

    def store(name, build, label=None):
        out = cache / name
        if not out.exists():
            # Building these is minutes of disk reads on a first run, so say what is
            # happening rather than leaving the terminal silent before the scan starts.
            if progress and label:
                print(f"  {label}", end="", flush=True, file=sys.stderr)
            t0 = time.time()
            arr = build()
            if arr is None:
                if progress and label:
                    print(" - skipped", file=sys.stderr)
                return None
            # Via a temp name: a master is reused purely because its file exists, so one
            # cut short by Ctrl+C or a full disk would otherwise be loaded on every later run.
            # The temp name keeps the .npy suffix, or np.save would append a second one.
            tmp = out.with_name(out.stem + ".part.npy")
            np.save(tmp, np.ascontiguousarray(arr, dtype=np.float32))
            os.replace(tmp, out)
            built[0] += 1
            if progress and label:
                print(f" - {time.time() - t0:.1f}s", file=sys.stderr)
        return str(out)

    if darks_dir is not None:
        bias_files = sorted(darks_dir.glob("masterBias*.xisf"))
        dark_files = sorted(darks_dir.glob("masterDark*.xisf"))
        candidates = bias_files + dark_files
        if progress:
            print(f"calibration: {len(candidates)} masters from {darks_dir}", file=sys.stderr)
        for n, candidate in enumerate(candidates, 1):
            m = re.search(r"EXPOSURE-([0-9.]+)s", candidate.name, re.IGNORECASE)
            settings = xisf_master_settings(candidate)
            exposure = float(m.group(1)) if m else None
            if candidate in dark_files and exposure is None:
                keyword_exp = _number(_xisf_image_spec(candidate)["keywords"].get("EXPTIME"))
                exposure = keyword_exp
            is_bias = candidate in bias_files
            if not is_bias and exposure is None:
                print(f"calibration: {candidate.name} names no exposure; ignored",
                      file=sys.stderr)
                continue
            label = f"[{n}/{len(candidates)}] {'bias' if is_bias else f'dark {exposure:g}s'}"
            stamp = _fingerprint([candidate])
            name = (f"bias-{stamp}.npy" if is_bias else f"dark_{exposure:g}s-{stamp}.npy")
            path = store(name, lambda c=candidate: read_xisf(c), label)
            if not path:
                continue
            entry = {"path": path, "settings": settings, "source": candidate.name}
            if is_bias:
                entry["level"] = bias_level(master(path))
                manifest["biases"].append(entry)
            else:
                entry["exposure"] = exposure
                manifest["darks"].append(entry)
        if len(manifest["biases"]) > 1 and progress:
            print(f"calibration: {len(manifest['biases'])} master biases; each light gets "
                  f"the one matching its gain/offset/binning", file=sys.stderr)
        if progress:
            print(f"calibration: {len(manifest['darks'])} master darks + "
                  f"{len(manifest['biases'])} bias ready", file=sys.stderr)
    else:
        # No raw XISF masters given: reuse whatever bias/dark .npy the cache already holds
        # from a previous run, instead of silently calibrating with neither. The index says
        # what each one is; without it (an older cache) only the newest bias is trusted.
        # Oldest first, so the newest master wins. The names end in a content hash, so
        # sorting by name picked an arbitrary one once a master had been rebuilt.
        def by_age(pattern):
            return sorted(cache.glob(pattern), key=lambda q: q.stat().st_mtime)

        for kind, pattern in (("bias", "bias-*.npy"), ("dark", "dark_*s-*.npy")):
            for q in by_age(pattern):
                info = index.get(q.name)
                m = re.search(r"dark_([0-9.]+)s-", q.name)
                entry = {"path": str(q), "settings": (info or {}).get("settings", {}),
                         "source": (info or {}).get("source", q.name)}
                if kind == "bias":
                    entry["level"] = (info or {}).get("level") or bias_level(master(str(q)))
                    if info is None:
                        manifest["biases"] = []     # unindexed: newest one only
                    manifest["biases"].append(entry)
                elif m:
                    entry["exposure"] = float(m.group(1))
                    manifest["darks"].append(entry)
        if progress and (manifest["biases"] or manifest["darks"]):
            print(f"calibration: reusing {len(manifest['darks'])} cached master darks + "
                  f"{len(manifest['biases'])} bias from {cache}", file=sys.stderr)

    # The bias used for flats, and the default pedestal for frames that match none.
    bias = master(manifest["biases"][-1]["path"]) if manifest["biases"] else None
    manifest["bias_level"] = manifest["biases"][-1]["level"] if manifest["biases"] else None

    darkflats_by_session = _darkflat_sessions(darkflat_dirs, {s for s, _f in flat_sets})
    master_darkflats = {}
    for session, files in darkflats_by_session.items():
        built_df = []
        for exp, group in _darkflat_groups(files).items():
            name = (f"darkflat_{_name_tag(Path(session).name or 'all', exp)}"
                    f"-{_fingerprint(group)}.npy")
            path = store(name, lambda g=group: _combine_darkflats(g))
            if path:
                built_df.append((exp, master(path)))
                manifest.setdefault("darkflats", []).append(path)
        master_darkflats[session] = built_df
        if progress and built_df:
            print(f"calibration: {len(built_df)} master dark flat(s) for "
                  f"{Path(session).name or 'every session'}", file=sys.stderr)

    if progress and flat_sets:
        if bias is None and not master_darkflats:
            # The pedestal is part of every flat, so leaving it in flattens the measured
            # vignetting profile. Worth saying out loud rather than quietly going ahead.
            print("calibration: WARNING - no master bias or dark flats, so flats keep their "
                  "pedestal and the vignetting they model will be understated", file=sys.stderr)
        print(f"calibration: {len(flat_sets)} master flats to prepare "
              f"(one per session and filter)", file=sys.stderr)
    for n, ((session, filt), files) in enumerate(flat_sets.items(), 1):
        label = f"[{n}/{len(flat_sets)}] flat {Path(session).name} / {filt} ({len(files)} frames)"
        files, warning = _majority_settings(files, f"{Path(session).name} / {filt}")
        if warning:
            print(f"calibration: WARNING - {warning}", file=sys.stderr)
        darkflats = master_darkflats.get(session, []) + master_darkflats.get("", [])
        try:
            settings = frame_settings(fits.getheader(str(files[0])))
        except OSError:
            settings = {}
        # The bias these flats were shot at: with masters for several gains, the default
        # one can carry another gain's pedestal entirely.
        flat_bias = next((master(b["path"]) for b in reversed(manifest["biases"])
                          if not settings_conflicts(b["settings"], settings)), None)
        path = store(f"flat_{_name_tag(Path(session).name, filt)}-"
                     f"{_fingerprint(files)}{'' if flat_bias is None else 'b'}.npy",
                     lambda f=files, d=darkflats, b=flat_bias: _build_master_flat(f, b, d),
                     label)
        if path:
            manifest["flats"][(session, filt)] = {"path": path, "settings": settings}

    used = ([b["path"] for b in manifest["biases"]] + [d["path"] for d in manifest["darks"]]
            + [f["path"] for f in manifest["flats"].values()] + manifest.get("darkflats", []))
    manifest["masters_mtime"] = max((Path(q).stat().st_mtime for q in used), default=0.0)
    prune_calibration_cache(cache, used, progress)
    new_index = {Path(e["path"]).name: {k: e.get(k) for k in ("settings", "level", "source",
                                                                "exposure")}
                 for e in manifest["biases"] + manifest["darks"]}
    write_json_atomic(index_path, new_index)
    if progress:
        print(f"calibration: {len(manifest['darks'])} darks + {len(manifest['flats'])} flats "
              f"ready ({built[0]} built, rest reused) in {cache}", file=sys.stderr)
    return manifest


def bias_level(bias):
    """The master bias' pedestal in ADU, for estimating sky signal where no dark or bias
    could be subtracted. A strided median: the pedestal is flat to a fraction of an ADU."""
    return float(np.median(np.asarray(bias[::4, ::4], dtype=np.float32)))


def master_bias_level(darks_dir):
    """bias_level() straight from the masterBias XISF, for runs that do not calibrate."""
    darks_dir = Path(darks_dir).expanduser() if darks_dir else None
    if darks_dir is None or not darks_dir.is_dir():
        return None
    for candidate in sorted(darks_dir.glob("masterBias*.xisf"))[:1]:
        try:
            return bias_level(read_xisf(candidate))
        except (OSError, ValueError) as e:
            print(f"calibration: cannot read {candidate.name} for the bias level ({e})",
                  file=sys.stderr)
    return None


# Everything prepare_calibration() writes. Only these are ever pruned from the cache.
_CACHE_FILE_PATTERNS = ("bias-*.npy", "dark_*s-*.npy", "flat_*-*.npy", "darkflat_*-*.npy",
                        "*.part.npy")


def prune_calibration_cache(cache: Path, used, progress=True):
    """Delete cached masters this run did not use.

    Every rebuilt master flat gets a new name (its inputs' fingerprint is part of it), so
    without this the old one stayed behind forever - ~36 MB per full-size frame, per
    session and filter, per rescan after new flats. Anything removed is rebuilt from the
    source frames if it is ever needed again. Only files matching the names this module
    writes are touched.
    """
    keep = {Path(q).resolve() for q in used if q}
    freed, removed = 0, 0
    for pattern in _CACHE_FILE_PATTERNS:
        for stale in cache.glob(pattern):
            if stale.resolve() in keep:
                continue
            try:
                size = stale.stat().st_size
                stale.unlink()
            except OSError:
                continue
            freed += size
            removed += 1
    if progress and removed:
        print(f"calibration: removed {removed} unused cached master(s), "
              f"{freed / 1e6:.0f} MB freed", file=sys.stderr)


def calibration_plan(manifest, path: Path, exposure, filt, settings):
    """Which masters calibrate one light frame - paths only, nothing loaded.

    Returns {dark, flat, label, pedestal, warnings}. Pure data from the manifest and the
    frame's own exposure/filter/settings, so the parent can work out whether a cached row
    was measured against the same masters without opening the file again.

    The dark has to match the exposure exactly - an interpolated one would invent dark
    current that was never measured - and falls back to the bias, which is the part of the
    pedestal that does not depend on exposure at all. Either has to agree with the frame on
    gain, offset and binning where both say: a dark at another gain carries another
    pattern, and subtracting it is worse than subtracting none.
    """
    plan = {"dark": None, "flat": None, "label": "", "pedestal": None, "warnings": []}
    if not manifest:
        return plan
    warnings = plan["warnings"]
    parts = []

    exp = _number(exposure)
    if exp is not None:
        same_exp = [d for d in manifest["darks"] if abs(d["exposure"] - exp) < 1e-6 * max(1.0, exp)]
        matching = [d for d in same_exp if not settings_conflicts(d["settings"], settings)]
        if same_exp and not matching:
            warnings.append(f"dark {exp:g}s not used: "
                            + "; ".join(settings_conflicts(same_exp[0]["settings"], settings)))
        if matching:
            best = min(matching, key=lambda d: _temp_gap(d["settings"], settings) or 0.0)
            gap = _temp_gap(best["settings"], settings)
            if gap is not None and gap > DARK_TEMP_TOLERANCE_C:
                warnings.append(f"dark {exp:g}s is {gap:.1f}\N{DEGREE SIGN}C off the frame's "
                                f"sensor temperature")
            plan["dark"] = best["path"]
            parts.append(f"dark {exp:g}s")

    biases = manifest.get("biases", [])
    good_bias = [b for b in biases if not settings_conflicts(b["settings"], settings)]
    if biases and not good_bias:
        warnings.append("bias not used: " + "; ".join(settings_conflicts(biases[0]["settings"],
                                                                          settings)))
    if good_bias:
        plan["pedestal"] = good_bias[-1].get("level")
        if plan["dark"] is None:
            plan["dark"] = good_bias[-1]["path"]
            parts.append("bias")

    session = session_dir_of(path, LIGHT_DIR_NAME)
    flat = manifest["flats"].get((str(session) if session else "", (filt or "").strip().upper()))
    if flat:
        conflict = [c for c in settings_conflicts(flat["settings"], settings)
                    if c.startswith("binning")]
        if conflict:
            warnings.append("flat not used: " + conflict[0])
        else:
            plan["flat"] = flat["path"]
            parts.append("flat")

    plan["label"] = " + ".join(parts)
    return plan


def write_calibrated(manifest, path: Path, data, header, calibration, signature):
    """Save the calibrated frame, mirroring its place in the source tree.

    float32 and not uint16: dark subtraction leaves genuine negative pixels wherever the
    sky sat below the pedestal, and rounding those away is exactly the information a
    stacker needs to weight the frame. An existing file is left alone only when its CALSIG
    card says it came from this very source file and these very masters - an mtime check
    kept stale copies whenever the masters changed in any way other than getting newer
    (a flat that stopped applying, a different dark) or the light itself was replaced.
    """
    out_root = manifest.get("calibrated_dir") if manifest else None
    if not out_root or not calibration:
        return None
    root = Path(manifest["root"])
    try:
        relative = path.resolve().relative_to(root)
    except ValueError:
        relative = Path(path.name)
    out = Path(out_root) / relative
    if out.exists():
        try:
            if fits.getheader(str(out)).get("CALSIG") == signature:
                return str(out)
        except OSError:
            pass
    out.parent.mkdir(parents=True, exist_ok=True)

    head = header.copy()
    # The frame is float32 now; the integer scaling the raw file carried would be applied
    # to it again on read.
    for key in ("BZERO", "BSCALE", "BITPIX"):
        head.pop(key, None)
    head["CALSTAT"] = (("D" if "dark" in calibration or "bias" in calibration else "")
                       + ("F" if "flat" in calibration else ""),
                       "calibration applied: D=dark/bias, F=flat")
    head["CALSIG"] = (signature, "source + masters this was calibrated from")
    head.add_history(f"calibrated by fits_light_stats: {calibration}")
    tmp = out.with_suffix(out.suffix + ".part")
    fits.PrimaryHDU(np.ascontiguousarray(data, dtype=np.float32), header=head).writeto(
        tmp, overwrite=True)
    os.replace(tmp, out)     # never leave a half-written frame behind under the real name
    return str(out)


def first_image_hdu(hdul) -> Any:
    """The first HDU in the file that carries image data.

    Annotated Any deliberately: what comes back is a PrimaryHDU, an ImageHDU or a
    CompImageHDU depending on the file, and astropy exposes no public base type covering
    the .data/.header/.shape those share. Without it a type checker infers the HDUList
    itself and flags every attribute read on the result.
    """
    for hdu in hdul:
        if hdu.data is not None:
            return hdu
    raise ValueError("file contains no image data")


def _circular_r_axial(angles_deg):
    """Concentration (0=scattered, 1=all-aligned) and mean direction of a set of axial
    (180-degree-periodic) angles, e.g. elongation position angles - a line and its reverse are
    the same direction, so plain circular stats would cancel opposite-pointing samples that
    actually agree. Doubling the angle before averaging fixes that (standard trick for axial data).

    The raw mean resultant length is badly biased upward on small samples: n randomly
    oriented stars land at r ~ 1/sqrt(n), so five noise stars average r = 0.40 and reach
    0.66 one time in ten - identical to the values this frame's real "tracking" verdicts
    were being read off. What is returned instead is the Rayleigh-debiased concentration
    sqrt((n*r^2 - 1)/(n - 1)), which sits at ~0 for scattered angles at any n while
    leaving a genuinely aligned sample essentially untouched.
    """
    n = len(angles_deg)
    if n == 0:
        return None, None
    theta = np.deg2rad(np.asarray(angles_deg, dtype=float)) * 2.0
    c, s = float(np.mean(np.cos(theta))), float(np.mean(np.sin(theta)))
    r = math.hypot(c, s)
    mean_angle = math.degrees(math.atan2(s, c)) / 2.0 % 180.0
    if n < 2:
        return 0.0, mean_angle
    return math.sqrt(max(0.0, (n * r * r - 1.0) / (n - 1.0))), mean_angle


def _background_stats(data):
    """Sigma-clipped background level and noise, estimated from a strided subsample.

    Every clip iteration re-medians (so effectively re-sorts) the array, which over a
    ~9 Mpx frame was half the total cost of measuring it. Fitting the same clip to a
    ~1 Mpx subsample of that frame reproduces the median to well under 0.1 ADU and the
    sigma to ~0.1% - far finer than either consumer needs, since one becomes a constant
    pedestal to subtract and the other only sets a 5-sigma detection threshold.
    """
    step = math.isqrt(max(1, data.size // STATS_SUBSAMPLE_TARGET_PX))
    sample = data[::step, ::step] if step > 1 else data
    _mean, median, std = sigma_clipped_stats(sample, sigma=3.0, maxiters=5)
    return float(median), max(float(std), 1e-6)


UPSAMPLE_CHUNK_PX = 1 << 20   # output pixels interpolated per pass; see _lerp_upsample


def _lerp_upsample(coarse, h, w):
    """Bilinear upsample of a small 2D array to (h, w), align-corners.

    Produces bit-for-bit what ndimage.zoom(coarse, ..., order=1) produces - same mapping
    (output index i reads input coordinate i*(n_in-1)/(n_out-1)), same float64 weights,
    same rounding to the input dtype on the way out - but as two 1D interpolations rather
    than zoom's general N-D spline machinery, which was costing ~80 ms per 9 Mpx frame.

    The row pass runs on a (h, wb) array barely wider than the input. The column pass is
    the only full-frame work and is done a slab at a time: keeping the float64 working
    buffers inside cache is both faster than one 9 Mpx pass and keeps a scan worker's peak
    memory flat, which matters when the scan runs one of these per core.
    """
    hb, wb = coarse.shape

    def axis_weights(n_in, n_out):
        if n_in == 1:            # nothing to interpolate between: zoom broadcasts the row
            return np.zeros(n_out, dtype=np.intp), np.zeros(n_out)
        x = np.arange(n_out) * ((n_in - 1) / (n_out - 1))
        i0 = np.minimum(np.floor(x).astype(np.intp), n_in - 2)
        return i0, x - i0     # float64 weights: what zoom interpolates in internally

    i0, fy = axis_weights(hb, h)
    rows = coarse[i0] * (1.0 - fy)[:, None]
    rows += coarse[np.minimum(i0 + 1, hb - 1)] * fy[:, None]

    j0, fx = axis_weights(wb, w)
    j1 = np.minimum(j0 + 1, wb - 1)
    left_w, right_w = (1.0 - fx)[None, :], fx[None, :]
    out = np.empty((h, w), dtype=coarse.dtype)
    step = max(1, UPSAMPLE_CHUNK_PX // max(w, 1))
    for start in range(0, h, step):
        block = rows[start:start + step]
        left, right = block[:, j0], block[:, j1]
        left *= left_w
        right *= right_w
        left += right
        out[start:start + step] = left
    return out


def _flatten_background(data_sub, cell=32):
    """Estimate and remove large-scale background structure (nebulosity, gradients) that a plain
    global sigma-clip leaves behind. A diffuse nebula edge is smooth but can still be many sigma
    above the frame's global background level, so it reads as a long, line-like feature to a naive
    per-pixel threshold - a real problem for a target this nebulous. Building the background from
    the median of coarse (cell x cell) blocks - cheap, since it runs on a ~100x smaller array - and
    upsampling captures that slow variation without being dragged around by stars or a thin trail,
    both of which are near-zero-area compared to one block.
    """
    h, w = data_sub.shape
    hb, wb = h // cell, w // cell
    if hb < 1 or wb < 1:
        # Smaller than a single cell: there is no large-scale structure to model, so a
        # global level is the only background this can honestly estimate.
        return data_sub - np.median(data_sub)
    trimmed = data_sub[:hb * cell, :wb * cell].reshape(hb, cell, wb, cell)
    coarse = np.median(trimmed, axis=(1, 3))
    coarse = ndimage.median_filter(coarse, size=5)
    # Sized exactly (h, w) by construction, so unlike zoom - which rounds a float product
    # and can land a pixel short - this needs no edge padding to broadcast.
    bg = _lerp_upsample(coarse, h, w)
    # Subtract into the upsample buffer rather than allocating a third full-size float array.
    return np.subtract(data_sub, bg, out=bg)


def _hough_bin_pixel_counts(shape, n_theta):
    """Number of image pixels that land in each (rho, theta) accumulator bin.

    This is the normalisation the accumulator has to be measured against, and a plain chord
    length is not it: rho is binned to whole pixels, so near +-45 deg two pixel diagonals fall
    in one bin and the count runs up to sqrt(2)x the chord. Dividing by the chord therefore
    manufactures a peak at exactly +-45 deg in every frame. Depends only on the frame shape and
    the theta grid, so it is built once per process and reused for every frame.
    """
    key = (shape, n_theta)
    cached = _HOUGH_NORM_CACHE.get(key)
    if cached is None:
        theta = np.linspace(-np.pi / 2, np.pi / 2, n_theta, endpoint=False)
        counts, angles, dists = hough_line(np.ones(shape, dtype=bool), theta=theta)
        cached = (counts.astype(np.float64), angles, dists, theta)
        _HOUGH_NORM_CACHE[key] = cached
    return cached


def _compact_source_mask(segm):
    """Pixels belonging to compact (star-like) sources only.

    Blanking every detected source would erase exactly what this is looking for: a bright trail
    is detected as a source in its own right. Sources that are long and mostly empty inside their
    bounding box are line-like and stay in the mask; stars and nebula knots fill 50-80% of theirs
    and are blanked, along with the diffraction spikes that would otherwise feed the transform
    short straight lines.
    """
    if segm is None:
        return None
    labels = segm.data
    # The bounding boxes and areas this needs are already on the segmentation - the source
    # catalog and the hot-pixel check have normally built them by the time a frame gets
    # here - so read them off it instead of running a second find_objects and a second
    # full-frame bincount, which together cost about as much as the mask itself.
    linear = [
        lab for lab, sl, area in zip(segm.labels, segm.slices, segm.areas)
        if max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start) >= TRAIL_LINEAR_MIN_EXTENT_PX
        and area / float((sl[0].stop - sl[0].start) * (sl[1].stop - sl[1].start))
        <= TRAIL_LINEAR_MAX_FILL
    ]
    if not linear:
        return labels > 0
    is_compact = np.ones(int(segm.max_label) + 1, dtype=bool)
    is_compact[0] = False
    is_compact[np.array(linear)] = False
    return is_compact[labels]


def _drop_detector_lines(mask):
    """Blank hot columns/rows in place: perfectly straight, and not from the sky.

    A hot column is a flawless full-height line and outscores any real trail in the transform.
    It is one pixel wide, though, while a trail is at least as wide as the PSF, so its immediate
    neighbours sit at the background rate where a real trail's are lit up too.
    """
    for axis in (0, 1):
        counts = mask.sum(axis=axis).astype(np.float64)
        if counts.size < 3:
            continue
        neighbour = np.maximum(
            np.concatenate(([0.0], counts[:-1])),
            np.concatenate((counts[1:], [0.0])),
        )
        hot = (counts > TRAIL_DETECTOR_LINE_FRAC * mask.shape[axis]) & \
              (neighbour < TRAIL_DETECTOR_LINE_ISOLATION * counts)
        if hot.any():
            if axis == 0:
                mask[:, hot] = False
            else:
                mask[hot, :] = False
    return mask


def _profile_width(profile, offsets):
    """(FWHM, peak height, peak index) of the ridge at the centre of a perpendicular profile."""
    wings = profile[np.abs(offsets) >= TRAIL_PROFILE_WING_PX]
    base = float(np.nanmedian(wings)) if np.isfinite(wings).any() else 0.0
    core = np.nan_to_num(profile - base, nan=-np.inf)
    inner = np.abs(offsets) <= TRAIL_PROFILE_CORE_PX
    if not np.isfinite(core[inner]).any():
        return None
    peak_i = int(np.argmax(np.where(inner, core, -np.inf)))
    peak = float(core[peak_i])
    if peak <= 0.0:
        return None

    def half_max_crossing(step):
        i = peak_i
        while 0 <= i + step < core.size and core[i + step] > peak / 2.0:
            i += step
        j = i + step
        if not (0 <= j < core.size):
            return float(offsets[i])
        # interpolate the crossing so a 1 px spike measures ~1 px rather than 2
        return float(offsets[i] + step * (core[i] - peak / 2.0) / max(core[i] - core[j], 1e-12))

    return half_max_crossing(1) - half_max_crossing(-1), peak, peak_i, base


def _line_cross_section(flat, std, valid, theta, rho):
    """Measure the frame along and across one candidate line.

    Along the line, averaging every sample beats the per-pixel noise down by sqrt(n), so a trail
    that only grazes the mask threshold still stands up as a clean ridge - the 2-3 sigma streaks
    in this data come out at 35-890 while the worst clean frame reaches 13.7.

    Across the line, the ridge's *width* says what drew it: ~1 px is a detector artefact, a
    PSF-ish few px is a trail, tens of px is leftover nebulosity that happened to line up. Star
    pixels are dropped throughout, or one bright star on the line would dominate the average.
    """
    h, w = flat.shape
    ct, st = float(np.cos(theta)), float(np.sin(theta))
    ts = np.arange(-float(math.hypot(h, w)), float(math.hypot(h, w)), 1.0)
    xs, ys = rho * ct - ts * st, rho * st + ts * ct
    offsets = np.arange(-TRAIL_PROFILE_HALF_PX, TRAIL_PROFILE_HALF_PX + 1, dtype=float)
    # samples[offset, position along the line]; NaN where off-frame or on a star
    samples = np.full((offsets.size, ts.size), np.nan, dtype=np.float32)
    for i, off in enumerate(offsets):
        xo = np.rint(xs + off * ct).astype(np.int64)
        yo = np.rint(ys + off * st).astype(np.int64)
        inside = (xo >= 0) & (xo < w) & (yo >= 0) & (yo < h)
        if not inside.any():
            continue
        yy, xx = yo[inside], xo[inside]
        vals = flat[yy, xx].astype(np.float32)
        if valid is not None:
            vals = np.where(valid[yy, xx], vals, np.nan)
        samples[i, inside] = vals

    finite = np.isfinite(samples)
    if int(finite.sum()) < TRAIL_PROFILE_MIN_SAMPLES:
        return None
    filled = np.where(finite, samples, np.float32(0.0))
    counts = finite.sum(axis=1)
    profile = np.where(counts > 0, filled.sum(axis=1) / np.maximum(counts, 1), np.nan)
    found = _profile_width(profile, offsets)
    if found is None:
        return None
    _w0, _peak, peak_i, base = found

    # Along-line series at the ridge, averaged over the PSF core for a sqrt(3) noise gain.
    lo, hi = max(0, peak_i - 1), min(offsets.size, peak_i + 2)
    rows = finite[lo:hi].sum(axis=0)
    series = filled[lo:hi].sum(axis=0) / np.maximum(rows, 1) - base
    ok = rows > 0
    total = int(ok.sum())
    if total < TRAIL_MIN_LINE_SAMPLES:
        return None
    signal = np.where(ok, series, 0.0)
    per_sample = std / math.sqrt(hi - lo)

    # One figure for the whole line. A freely sliding window would in principle recover a
    # satellite that only clips a corner, but measured over this set it latches onto local
    # nebulosity just as readily: it lifted the worst clean frame from 13.7 to 28.9 while the
    # faintest real trail sat at 35.4, so it costs far more separation than it buys.
    ridge = float(signal.sum() / (per_sample * math.sqrt(total)))

    # Where along the line the ridge is actually lit. Only used to draw the trail once it has
    # been confirmed - as a *detection* statistic a sliding window latches onto nebulosity
    # (see above), but for showing extent that risk is already behind us.
    smooth = ndimage.uniform_filter1d(signal, TRAIL_EXTENT_SMOOTH_PX, mode="constant")
    lit = np.flatnonzero(ok & (smooth > TRAIL_EXTENT_SIGMA * per_sample /
                               math.sqrt(TRAIL_EXTENT_SMOOTH_PX)))
    # Nothing stands out above the smoothed noise: fall back to the in-frame chord.
    span = lit if lit.size else np.flatnonzero(ok)
    return {"width": _w0, "ridge": ridge,
            "t_lo": float(ts[span[0]]), "t_hi": float(ts[span[-1]])}


def _flat_noise(flat):
    """Per-pixel noise of a background-flattened frame.

    Measured after flattening rather than reusing the whole-frame sigma: that one is fitted
    before the nebula is taken out, so on a nebulous frame it carries the nebula's spread as
    well and can run several times the true pixel noise - pushing both the star detection
    threshold and the trail mask far past the faint end of what they are looking for.
    """
    _mean, _median, noise = sigma_clipped_stats(flat[::3, ::3], sigma=3.0, maxiters=5)
    return max(float(noise), 1e-6)


def find_trail(flat, noise, segm, psf_fwhm=None):
    """Locate a satellite trail: find candidate lines, then confirm them on the pixels.

    Returns None, or the confirmed line as {theta, rho, width, ridge, x0, y0, x1, y1} with the
    endpoints in full-frame pixels - what the review server draws a box around.

    Trails in this data run 2-3 sigma per pixel, so only a quarter to a half of the pixels along
    one clear a 3-sigma cut and the survivors are a dashed line with gaps of tens of pixels. That
    rules out asking a probabilistic Hough transform for long *unbroken* segments. What survives
    the gaps is counting: every mask pixel on a line votes for it however scattered they are, so
    the whole accumulator is a matched filter for a dashed line. Counts are measured against the
    exact per-bin pixel count (see _hough_bin_pixel_counts) as a Poisson excess, which makes the
    score a significance that means the same thing on a sparse 30 s sub and a nebula-filled 180 s
    one. Pooling 4x4 blocks and keeping those with >=2 mask pixels first cuts isolated noise ~50x
    while a trail, which leaves several hits per block it crosses, comes through.

    A count alone still can't tell a trail from a hot column or a nebula filament, so the leading
    candidates are re-measured on the pixels: a real trail is a ridge whose cross-section is about
    as wide as the stars (it is the PSF, smeared along a line) - narrower is a detector artefact,
    much wider is nebulosity.

    `flat` is the background-flattened frame (_flatten_background) and `noise` its own pixel
    noise (_flat_noise): the star detection already needs both, so they are built once.
    """
    compact = _compact_source_mask(segm)
    valid = None if compact is None else \
        ~ndimage.binary_dilation(compact, iterations=TRAIL_STAR_DILATE_ITERS)  # type: ignore[operator]
    trail_mask = flat > TRAIL_MASK_SIGMA * noise
    if valid is not None:
        trail_mask &= valid
    _drop_detector_lines(trail_mask)

    ds = TRAIL_HOUGH_DOWNSAMPLE
    # block_size takes a per-axis tuple; skimage's type stub declares it int.
    pooled = block_reduce(trail_mask.astype(np.uint8), (ds, ds), np.sum) >= TRAIL_MIN_BLOCK_HITS  # type: ignore[arg-type]
    bin_pixels, angles, dists, theta = _hough_bin_pixel_counts(pooled.shape, TRAIL_HOUGH_N_THETA)
    accumulator, _angles, _dists = hough_line(pooled, theta=theta)

    hit_rate = float(pooled.mean())
    expected = hit_rate * bin_pixels
    score = (accumulator.astype(np.float64) - expected) / \
        np.sqrt(np.maximum(expected * (1.0 - hit_rate), 1e-9))
    # Short chords through a corner are noisy and can't hold a frame-crossing trail anyway.
    score = np.where(bin_pixels >= TRAIL_MIN_CHORD_FRAC * min(pooled.shape), score, -np.inf)

    min_width = TRAIL_MIN_WIDTH_PX if not psf_fwhm else \
        max(TRAIL_MIN_WIDTH_PX, TRAIL_MIN_WIDTH_PSF_FRAC * psf_fwhm)
    centre = (ds - 1) / 2.0
    h, w = flat.shape
    corners = [(0.0, 0.0), (w - 1.0, 0.0), (0.0, h - 1.0), (w - 1.0, h - 1.0)]
    for _ in range(TRAIL_N_CANDIDATES):
        i, j = np.unravel_index(np.argmax(score), score.shape)
        if not np.isfinite(score[i, j]):
            break
        angle = float(angles[j])
        # rho is in pooled-pixel units about the pooled origin; shift to full-frame pixels.
        rho = ds * float(dists[i]) + centre * (np.cos(angle) + np.sin(angle))
        # A line running this close to the border along its whole length never leaves the
        # frame's edge band: at this angle, no in-frame line can sit further out than the
        # corner projections, so within TRAIL_EDGE_MARGIN_PX of either bound is a hugging line.
        ct, st = math.cos(angle), math.sin(angle)
        projections = [cx * ct + cy * st for cx, cy in corners]
        hugs_edge = (rho <= min(projections) + TRAIL_EDGE_MARGIN_PX or
                     rho >= max(projections) - TRAIL_EDGE_MARGIN_PX)
        if hugs_edge:
            score[max(0, i - TRAIL_PEAK_RHO_PAD):i + TRAIL_PEAK_RHO_PAD + 1,
                  max(0, j - TRAIL_PEAK_THETA_PAD):j + TRAIL_PEAK_THETA_PAD + 1] = -np.inf
            continue
        section = _line_cross_section(flat, noise, valid, angle, rho)
        if section is not None and \
                min_width <= section["width"] <= TRAIL_MAX_WIDTH_PX and \
                section["ridge"] >= TRAIL_MIN_RIDGE_SIGMA:
            ct, st = math.cos(angle), math.sin(angle)
            # float() throughout: rho carries numpy's dtype out of the accumulator, and these
            # go straight to json.dumps in the review server.
            return {
                "theta": float(angle), "rho": float(rho),
                "width": float(section["width"]), "ridge": float(section["ridge"]),
                "x0": float(rho * ct - section["t_lo"] * st),
                "y0": float(rho * st + section["t_lo"] * ct),
                "x1": float(rho * ct - section["t_hi"] * st),
                "y1": float(rho * st + section["t_hi"] * ct),
            }
        score[max(0, i - TRAIL_PEAK_RHO_PAD):i + TRAIL_PEAK_RHO_PAD + 1,
              max(0, j - TRAIL_PEAK_THETA_PAD):j + TRAIL_PEAK_THETA_PAD + 1] = -np.inf
    return None


def _hot_pixel_flags(data_sub, segm, labels):
    """True for sources whose raw flux is a single-pixel spike rather than a real star.

    These are uncalibrated frames (no dark subtraction), so hot/warm pixels are still present.
    Convolving a single-pixel spike with the same Gaussian kernel used for detection spreads it
    into a blob shaped like the kernel's own PSF, so on the *convolved* data a hot pixel looks
    just like a real star of that FWHM - shape metrics alone can't separate them. The raw pixel
    values still can: a real star's flux is spread over several raw pixels, while a hot pixel's
    neighbours sit at background.
    """
    slices = segm.slices
    flags = np.zeros(len(labels), dtype=bool)
    for i, lab in enumerate(labels):
        sl = slices[lab - 1]
        cutout = data_sub[sl]
        seg_cutout = segm.data[sl] == lab
        vals = cutout[seg_cutout]
        if vals.size < 2:
            flags[i] = True
            continue
        top2 = np.partition(vals, -2)[-2:]
        peak, second = float(top2.max()), float(top2.min())
        flags[i] = peak <= 0 or (second / peak) < HOT_PIXEL_SECOND_PEAK_FRAC
    return flags


_GAUSS_1D_CACHE = {}


def _gaussian_1d(fwhm, size):
    """The 1D factor of the matched-filter kernel, built once per process.

    A 2D Gaussian is exactly the outer product of two 1D Gaussians, and normalising the
    2D kernel is the same as normalising each factor, so the detection map can be built
    with two 1D passes (2*size multiply-adds per pixel) instead of one 2D pass (size^2).
    The row is taken from the same make_2dgaussian_kernel() the 2D form used, so the
    filter itself is unchanged - only the order the sums are accumulated in.
    """
    key = (fwhm, size)
    kernel = _GAUSS_1D_CACHE.get(key)
    if kernel is None:
        row = np.asarray(make_2dgaussian_kernel(fwhm, size=size))[size // 2]
        kernel = _GAUSS_1D_CACHE[key] = (row / row.sum()).astype(np.float64)
    return kernel


def _segment_sources(data_sub, std):
    """(segmentation, convolved data) for the frame's stars. The convolved map is reused by the
    source catalog, so the two are produced together."""
    # Separable form of astropy's convolve(), which zero-pads the border and accumulates
    # in float64 before handing back the input's float32 - matched here so the detection
    # map is the same one, for a third of the cost on a 9 Mpx frame.
    kernel = _gaussian_1d(FWHM_GUESS_PX, 9)
    conv = ndimage.correlate1d(data_sub, kernel, axis=0, mode="constant", cval=0.0,
                               output=np.float64)
    ndimage.correlate1d(conv, kernel, axis=1, mode="constant", cval=0.0, output=conv)
    conv = conv.astype(np.float32)
    return detect_sources(conv, DETECT_SIGMA * std, n_pixels=MIN_NPIXELS), conv


def locate_trail(path, psf_fwhm=None):
    """Re-measure one frame far enough to say *where* its trail lies, or None.

    The scan stores only the has_trail flag, so the review server recovers the geometry on
    demand for the one frame being looked at rather than forcing a rescan of the whole set.
    Everything after the segmentation (the source catalog, flux radii) is skipped; pass the
    frame's already-measured FWHM so the width gate matches what the scan applied.
    """
    with fits.open(str(path)) as hdul:
        data = np.asarray(first_image_hdu(hdul).data, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"expected a 2D image, got shape {data.shape}")
    median, _std = _background_stats(data)
    data -= median
    flat = _flatten_background(data)
    del data
    noise = _flat_noise(flat)
    segm, _conv = _segment_sources(flat, noise)
    found = find_trail(flat, noise, segm, psf_fwhm)
    if found is not None:
        found["frame_h"], found["frame_w"] = flat.shape
    return found


def _fit_psfs(data_sub, segm, labels, xs, ys):
    """Fit an elliptical 2D Gaussian (+ constant) to every star at once.

    Returns (fwhm, roundness, orientation_deg, ok), one entry per star. FWHM follows the
    photutils definition (2.3548 * rms of the two axis sigmas) so the thresholds keep their
    meaning; roundness is minor/major sigma; orientation is the major axis' angle from +x,
    in photutils' convention.

    Why a fit rather than the catalog's moments: SourceCatalog takes its shape properties
    from `convolved_data` when given one, i.e. from stars already smeared by the 4 px
    detection kernel. Against per-star fits on real subs that read FWHM 4.6 for 3.6 px
    stars and roundness 0.90-0.97 for 0.74-0.81 ones, with almost no correlation left
    between the two. Moments on the unconvolved pixels are better but still truncated by
    the footprint and pulled by its noise. A fit uses every pixel of the stamp at its
    proper weight, and neighbouring sources' pixels are masked out of it.

    All stars are solved together as one batched Levenberg-Marquardt problem: a 7x7 normal
    system per star, built with batched matmuls, so a few hundred fits cost well under 0.1 s.
    """
    n = len(xs)
    empty = np.full(n, np.nan)
    if n == 0:
        return empty, empty, empty, np.zeros(0, dtype=bool)
    half = PSF_FIT_HALF_PX
    off = np.arange(-half, half + 1)
    cx, cy = np.rint(xs).astype(np.intp), np.rint(ys).astype(np.intp)
    rows = (cy[:, None] + off[None, :])[:, :, None]
    cols = (cx[:, None] + off[None, :])[:, None, :]
    stamps = data_sub[rows, cols].astype(np.float64)                  # (n, S, S)
    seg = segm.data[rows, cols]
    weight = ((seg == 0) | (seg == np.asarray(labels)[:, None, None])).astype(np.float64)
    gx = off[None, None, :].astype(np.float64)
    gy = off[:, None].astype(np.float64)[None]

    border = np.concatenate([stamps[:, 0, :], stamps[:, -1, :],
                             stamps[:, 1:-1, 0], stamps[:, 1:-1, -1]], axis=1)
    bg0 = np.median(border, axis=1)
    inv_var0 = 1.0 / (FWHM_GUESS_PX / 2.3548) ** 2
    # params: amplitude, x0, y0, a, b, c, background; the exponent is
    # -(a dx^2 + 2 b dx dy + c dy^2) / 2, i.e. [[a, b], [b, c]] is the inverse covariance.
    p = np.stack([stamps.max(axis=(1, 2)) - bg0, xs - cx, ys - cy,
                  np.full(n, inv_var0), np.zeros(n), np.full(n, inv_var0), bg0], axis=1)

    def model(p):
        dx = gx - p[:, 1, None, None]
        dy = gy - p[:, 2, None, None]
        q = p[:, 3, None, None] * dx * dx + 2 * p[:, 4, None, None] * dx * dy + \
            p[:, 5, None, None] * dy * dy
        e = np.exp(-0.5 * np.maximum(q, -50.0))   # a non-positive-definite step can't overflow
        return p[:, 6, None, None] + p[:, 0, None, None] * e, e, dx, dy

    def chi2_of(p):
        m, _e, _dx, _dy = model(p)
        c2 = np.sum(weight * (stamps - m) ** 2, axis=(1, 2))
        return np.where(np.isfinite(c2), c2, np.inf)

    chi2 = chi2_of(p)
    lam = np.full(n, 1e-2)
    eye = np.eye(7)
    for _ in range(PSF_FIT_ITERS):
        m, e, dx, dy = model(p)
        amp_e = p[:, 0, None, None] * e
        a, b, c = p[:, 3, None, None], p[:, 4, None, None], p[:, 5, None, None]
        jac = np.stack([e, amp_e * (a * dx + b * dy), amp_e * (b * dx + c * dy),
                        -0.5 * amp_e * dx * dx, -amp_e * dx * dy, -0.5 * amp_e * dy * dy,
                        np.ones_like(e)], axis=-1).reshape(n, -1, 7)
        wf = weight.reshape(n, -1)
        resid = (stamps - m).reshape(n, -1)
        # Batched BLAS matmul: a three-operand einsum here was the fit's single biggest cost.
        jac_w = jac * wf[:, :, None]
        jtj = np.matmul(jac_w.transpose(0, 2, 1), jac)
        grad = np.matmul(jac_w.transpose(0, 2, 1), resid[:, :, None])[..., 0]
        damped = jtj + lam[:, None, None] * (jtj * eye + 1e-12 * eye)
        try:
            step = np.linalg.solve(damped, grad[..., None])[..., 0]
        except np.linalg.LinAlgError:
            break
        trial = p + np.where(np.isfinite(step), step, 0.0)
        trial_chi2 = chi2_of(trial)
        better = trial_chi2 < chi2
        p[better] = trial[better]
        chi2 = np.where(better, trial_chi2, chi2)
        lam = np.clip(np.where(better, lam * 0.3, lam * 10.0), 1e-7, 1e7)

    amp, x0, y0, a, b, c = (p[:, k] for k in range(6))
    det = a * c - b * b
    with np.errstate(invalid="ignore", divide="ignore"):
        sxx, syy, sxy = c / det, a / det, -b / det          # covariance = inverse of [[a,b],[b,c]]
        mid = 0.5 * (sxx + syy)
        rad = np.sqrt((0.5 * (sxx - syy)) ** 2 + sxy * sxy)
        major, minor = mid + rad, mid - rad
        fwhm = 2.3548 * np.sqrt(mid)
        roundness = np.sqrt(minor / major)
    orientation = np.degrees(0.5 * np.arctan2(2.0 * sxy, sxx - syy))
    ok = (np.isfinite(chi2) & (amp > 0) & (a > 0) & (c > 0) & (det > 0) &
          (np.abs(x0) <= 2.0) & (np.abs(y0) <= 2.0) &
          (minor > 0.25 ** 2) & (major < (half / 1.5) ** 2) &
          np.isfinite(fwhm) & np.isfinite(roundness))
    return fwhm, roundness, orientation, ok


def detect_star_metrics(flat, noise, h, w, saturated_idx=None, saturation=SATURATION_ADU):
    """Detect stars in a background-flattened frame and return (hfr, fwhm, roundness, trail,
    elongation). Roundness is minor/major axis ratio: 1.0 = perfectly round, lower = more
    elongated. trail is find_trail's geometry for a satellite trail, or None.

    elongation summarises the *direction* of star elongation across the frame (None if too few
    elongated stars to say anything): a high "uniform_r" (stars all stretched the same way,
    regardless of where they sit in the frame) points at tracking/periodic-error rather than
    optics, since PE smears every star along the same sky-motion axis. A high "radial_r" instead
    means elongation direction tracks each star's own bearing from the frame center - the
    signature of coma/tilt/field curvature, which stretch stars radially outward.

    Every usable star is counted and gives an HFR (flux_radius, measured on the source's own
    pixels rather than a fixed box). Shapes - FWHM, roundness, orientation - come from PSF fits
    to the brightest unsaturated ones only (see _fit_psfs): a faint star's shape is mostly noise.

    saturated_idx: flat indices of pixels that clipped in the *raw* frame. Saturation has to be
    judged there - after dark subtraction, flat division and background removal a clipped star
    peaks anywhere from ~59k to ~63k depending on where it sits under the flat, so any fixed
    threshold on the calibrated data lets some of them through.
    """
    segm, conv = _segment_sources(flat, noise)
    if segm is None:
        return [], [], [], find_trail(flat, noise, segm), None

    # convolved_data steadies the centroids and selection; the shapes are fitted separately.
    cat = SourceCatalog(flat, segm, convolved_data=conv)
    labels = np.asarray(cat.labels)
    # np.asarray unwraps astropy Quantity columns in one vectorised step.
    x = np.asarray(cat.x_centroid, dtype=float)
    y = np.asarray(cat.y_centroid, dtype=float)
    fwhm_conv = np.asarray(cat.fwhm, dtype=float)
    maxv = np.asarray(cat.max_value, dtype=float)
    hot_pixel = _hot_pixel_flags(flat, segm, labels)
    saturated = np.zeros(labels.size, dtype=bool)
    if saturated_idx is not None and len(saturated_idx):
        sat_labels = np.unique(segm.data.ravel()[saturated_idx])
        saturated = np.isin(labels, sat_labels[sat_labels > 0])

    edge_ok = (x >= MIN_SEP_FROM_EDGE) & (x <= w - MIN_SEP_FROM_EDGE) & \
              (y >= MIN_SEP_FROM_EDGE) & (y <= h - MIN_SEP_FROM_EDGE)
    keep = edge_ok & ~saturated & ~hot_pixel & np.isfinite(fwhm_conv) & \
        (fwhm_conv > 0.3) & (fwhm_conv < 30.0)
    if saturated_idx is None:
        keep &= maxv < saturation      # no raw mask given: best effort on calibrated data
    if not keep.any():
        return [], [], [], find_trail(flat, noise, segm), None

    # flux_radius is the most expensive column, so only measure the sources that survived.
    kept = np.flatnonzero(keep)
    hfr = np.asarray(cat[keep].flux_radius(0.5), dtype=float)
    finite = np.isfinite(hfr)
    kept, hfr = kept[finite], hfr[finite]

    bright = kept[maxv[kept] >= PSF_FIT_MIN_PEAK_SIGMA * noise]
    if bright.size > PSF_FIT_MAX_STARS:
        bright = bright[np.argsort(maxv[bright])[::-1][:PSF_FIT_MAX_STARS]]
    fwhm_fit, round_fit, orient_fit, ok = _fit_psfs(flat, segm, labels[bright], x[bright], y[bright])
    if int(ok.sum()) >= PSF_FIT_MIN_STARS:
        shaped = bright[ok]
        fwhm_final, roundness_final, orientation_final = fwhm_fit[ok], round_fit[ok], orient_fit[ok]
    else:
        # Too few bright stars to fit: moments of the unconvolved pixels, never the convolved.
        raw_cat = SourceCatalog(flat, segm)[kept]
        semimajor = np.asarray(raw_cat.semimajor_axis, dtype=float)
        semiminor = np.asarray(raw_cat.semiminor_axis, dtype=float)
        fwhm_final = np.asarray(raw_cat.fwhm, dtype=float)
        roundness_final = np.divide(semiminor, semimajor, out=np.full_like(semimajor, np.nan),
                                    where=semimajor > 0)
        orientation_final = np.asarray(raw_cat.orientation, dtype=float)
        good = np.isfinite(fwhm_final) & np.isfinite(roundness_final)
        shaped = kept[good]
        fwhm_final, roundness_final, orientation_final = \
            fwhm_final[good], roundness_final[good], orientation_final[good]

    # The frame's own PSF sets how narrow a real trail can be, so measure it before looking.
    trail = find_trail(flat, noise, segm, float(np.median(fwhm_final)) if fwhm_final.size else None)
    x_final, y_final = x[shaped], y[shaped]

    elong_mask = roundness_final < ELONGATION_ROUND_THRESH
    elong = None
    if int(np.count_nonzero(elong_mask)) >= MIN_ELONGATED_FOR_DIRECTION:
        orient_elong = orientation_final[elong_mask]
        uniform_r, uniform_angle = _circular_r_axial(orient_elong)

        # Bearing of each elongated star from the frame center, as an axial (0-180) angle so it
        # is directly comparable to orientation; then how far each star's own elongation is from
        # that radial bearing (mod 180) - clustered near 0 means "stretched away from center".
        dx = x_final[elong_mask] - w / 2.0
        dy = y_final[elong_mask] - h / 2.0
        radial_angle = np.degrees(np.arctan2(dy, dx)) % 180.0
        radial_diff = (orient_elong - radial_angle) % 180.0
        radial_r, radial_mean_diff = _circular_r_axial(radial_diff)

        elong = {
            "n": int(np.count_nonzero(elong_mask)),
            "uniform_r": uniform_r,
            "uniform_angle_deg": uniform_angle,
            "radial_r": radial_r,
            "radial_diff_deg": radial_mean_diff,
        }

    return hfr.tolist(), fwhm_final.tolist(), roundness_final.tolist(), trail, elong


def frame_full_scale(header, override=None):
    """The ADU a pixel reads when it is full: --full-scale if given, else the SATURATE card,
    else the sensor's integer range (255 for 8-bit data, 65535 otherwise).

    DATAMAX is deliberately not used: plenty of software writes it as the brightest pixel
    actually present, which would mark every frame's brightest star as clipped.
    """
    if override:
        return float(override)
    saturate = _number(header.get("SATURATE"))
    if saturate is not None and saturate >= 255:
        return saturate
    return 255.0 if header.get("BITPIX") == 8 else 65535.0


def frame_pixel_scale(header):
    """arcsec per pixel from the optics cards, or None."""
    pixel_um = _number(header.get("XPIXSZ", header.get("PIXSIZE1")))
    focal_mm = _number(header.get("FOCALLEN"))
    binning = _number(header.get("XBINNING")) or 1.0
    if not pixel_um or not focal_mm or focal_mm <= 0:
        return None
    return 206.265 * pixel_um * binning / focal_mm


def frame_signature(plan, opts):
    """What a row was measured against: the masters that calibrated it and every option
    that changes a measured number. Together with an unchanged file (source_stat) a cached
    row whose signature still matches would come out identical if measured again."""
    opts = opts or {}
    parts = [MEASURE_VERSION, plan.get("dark"), plan.get("flat"),
             plan.get("pedestal"), opts.get("pedestal"), opts.get("full_scale")]
    return hashlib.sha1(json.dumps(parts, default=str).encode()).hexdigest()[:16]


def _stat_of(path: Path):
    st = path.stat()
    return [st.st_mtime_ns, st.st_size]


def process_file(path: Path, calib=None, opts=None):
    """Measure one frame.

    opts: {"pedestal": bias level in ADU to take the sky signal against when no dark or
    bias ends up subtracted (BIAS_ADU if not given), "full_scale": the sensor's full-scale
    ADU, overriding what the header says}.
    """
    opts = opts or {}
    source_stat = _stat_of(path)
    # One open for both header and data: getdata()+getheader() parsed the file twice.
    with fits.open(str(path), memmap=False) as hdul:
        hdu = first_image_hdu(hdul)
        header = hdu.header
        raw = np.asarray(hdu.data)
    if raw.ndim != 2:
        raise ValueError(f"expected a 2D image, got shape {raw.shape}")
    h, w = raw.shape

    # Summarise ADU off the native (usually uint16) array with a float64 accumulator:
    # exact, and it does not need a float64 copy of the whole frame to exist first.
    # These stay the numbers the sensor actually recorded, calibrated or not, because
    # they are what saturation and the camera's own range are read against.
    adu_min = float(raw.min())
    adu_max = float(raw.max())
    adu_mean = float(raw.mean(dtype=np.float64))
    exposure = header.get("EXPOSURE", header.get("EXPTIME"))
    filt = str(header.get("FILTER", "")).strip()
    settings = frame_settings(header)
    full_scale = frame_full_scale(header, opts.get("full_scale"))
    saturation = SATURATION_FRAC * full_scale
    # Clipped pixels are only recognisable as such before calibration rescales them.
    saturated_idx = np.flatnonzero(raw >= saturation)

    # float32 for the measurement pipeline: every step below (sigma clipping, block
    # medians, convolution, segmentation) is memory-bandwidth bound on a ~9 Mpx frame,
    # and float32 still resolves ~1e-4 ADU at these levels - four orders of magnitude
    # finer than the read noise the metrics are measured against.
    data = raw.astype(np.float32)
    del raw

    plan = calibration_plan(calib, path, exposure, filt, settings)
    warnings = list(plan["warnings"])
    dark = master(plan["dark"]) if plan["dark"] else None
    flat_field = master(plan["flat"]) if plan["flat"] else None
    calibration = plan["label"]
    dark_ok = dark is not None and dark.shape == data.shape
    flat_ok = flat_field is not None and flat_field.shape == data.shape
    if dark_ok:
        data -= dark
    if flat_ok:
        data /= flat_field
    # A master that does not match the frame's size is skipped above; the label has to say
    # so, or the report claims a calibration that never happened.
    if (dark is not None and not dark_ok) or (flat_field is not None and not flat_ok):
        warnings.append("a master's size does not match the frame; skipped")
        calibration = " + ".join(
            part for part in calibration.split(" + ")
            if (part == "flat" and flat_ok) or (part != "flat" and dark_ok))
    signature = frame_signature(plan, opts)
    # Saved before any measurement touches the array: what lands on disk is the calibrated
    # frame, not the background-subtracted one the metrics are taken from.
    calibrated_path = write_calibrated(calib, path, data, header, calibration,
                                       f"{signature}-{source_stat[0]}-{source_stat[1]}")
    pedestal = plan["pedestal"] if plan["pedestal"] is not None else opts.get("pedestal")
    if dark_ok:
        # The pedestal is gone, so what is left is the sky signal itself rather than an
        # estimate made by subtracting a nominal bias.
        signal_adu = float(data.mean(dtype=np.float64))
    else:
        signal_adu = adu_mean - (pedestal if pedestal is not None else BIAS_ADU)
    signal_adu_per_sec = (
        signal_adu / exposure
        if isinstance(exposure, (int, float)) and exposure > 0
        else None
    )

    median, _std = _background_stats(data)

    # Subtract in place: the raw frame is not needed again and a 2nd full-size copy is not free.
    data -= median
    # One background model for everything below. Stars used to be detected against the global
    # level and sigma, so on this nebula-filled field faint stars over bright nebulosity fell
    # under a threshold inflated by the nebula itself, while nebula knots read as stars.
    flat = _flatten_background(data)
    del data
    noise = _flat_noise(flat)
    hfrs, fwhms, roundnesses, trail, elong = detect_star_metrics(
        flat, noise, h, w, saturated_idx, saturation=saturation)
    if trail is not None:
        trail["frame_h"], trail["frame_w"] = h, w

    row = {
        "path": str(path),
        "name": path.name,
        "filter": filt,
        "exposure": exposure,
        "calibration": calibration,
        "calibration_warnings": warnings,
        "calibrated_path": calibrated_path,
        **solved_fields(None),
        "date_obs": str(header.get("DATE-LOC", header.get("DATE-OBS", ""))).strip(),
        "settings": settings,
        "full_scale_adu": full_scale,
        "pixel_scale_arcsec": frame_pixel_scale(header),
        "adu_min": adu_min,
        "adu_max": adu_max,
        "adu_mean": adu_mean,
        "signal_adu": signal_adu,
        "signal_adu_per_sec": signal_adu_per_sec,
        # What signal_adu was measured against when nothing was subtracted, so a backfill
        # of an older row never mixes pedestals within one set.
        "pedestal_adu": None if dark_ok else (pedestal if pedestal is not None else BIAS_ADU),
        "num_stars": len(hfrs),
        "hfr_avg": float(np.median(hfrs)) if hfrs else None,
        "fwhm_avg": float(np.median(fwhms)) if fwhms else None,
        "roundness_avg": float(np.median(roundnesses)) if roundnesses else None,
        "noise_adu": noise,
        "has_trail": trail is not None,
        # Stored so the review server can draw it without re-measuring the frame.
        "trail": trail,
        "elong_n": elong["n"] if elong else 0,
        "elong_uniform_r": elong["uniform_r"] if elong else None,
        "elong_uniform_angle_deg": elong["uniform_angle_deg"] if elong else None,
        "elong_radial_r": elong["radial_r"] if elong else None,
        "elong_radial_diff_deg": elong["radial_diff_deg"] if elong else None,
        # What an incremental rescan compares against (see build_dataset).
        "source_stat": source_stat,
        "scan_sig": signature,
    }
    add_derived_adu_fields(row)
    row["quality_score"], row["quality_reasons"], row["quality_verdict"] = compute_quality(row)
    return row


def solved_fields(solved):
    """The row fields a plate solve fills in, all None when it did not solve."""
    solved = solved or {}
    return {"solved_ra": solved.get("ra"), "solved_dec": solved.get("dec"),
            "solved_rotation": solved.get("rotation"),
            "solved_scale_arcsec": solved.get("scale_arcsec"),
            "pointing_error_arcsec": solved.get("error_arcsec")}


def first_image_header(path: Path):
    """Header of the first image HDU, without reading any pixel data."""
    with fits.open(str(path)) as hdul:
        hdus: Any = hdul   # see first_image_hdu for why this is typed loosely
        for hdu in hdus:
            if int(hdu.header.get("NAXIS", 0)) >= 2:
                return hdu.header.copy()
    raise ValueError("file contains no image data")


def solve_dataset(dataset, solver, workers=None, progress=True, redo=False):
    """Plate solve the frames of a measured dataset, in place.

    Solving only needs the file and its header, so it runs here, after measuring, rather
    than inside every measurement worker: there one ASTAP per core - each of them
    multithreaded itself - oversubscribed the machine several times over. Frames that
    already carry a solution are skipped unless `redo`; frames no longer at their path
    (moved to _deleted by the review server, say) are skipped and counted.
    """
    rows = [r for entry in dataset for r in entry.get("files", [])
            if "adu_mean" in r and (redo or r.get("solved_ra") is None)]
    if not rows:
        return {"solved": 0, "failed": 0, "missing": 0}

    def solve_one(row):
        path = Path(row["path"])
        if not path.exists():
            return "missing", None
        try:
            header = first_image_header(path)
        except (OSError, ValueError):
            return "failed", None
        solved = solve_frame(solver, path, header, int(header["NAXIS2"]))
        return ("solved" if solved else "failed"), solved

    counts = {"solved": 0, "failed": 0, "missing": 0}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, workers or ASTAP_CONCURRENCY)) as pool:
        futures = {pool.submit(solve_one, r): r for r in rows}
        for done, future in enumerate(as_completed(futures), 1):
            row = futures[future]
            outcome, solved = future.result()
            counts[outcome] += 1
            if solved:
                row.update(solved_fields(solved))
                if solver.get("update_header"):
                    # --write-wcs rewrote the file; record its new identity, or the next
                    # incremental scan would re-measure every frame it touched.
                    try:
                        row["source_stat"] = _stat_of(Path(row["path"]))
                    except OSError:
                        pass
            if progress:
                _report_progress(done, len(rows), t0, row.get("name", ""))
    if progress:
        print(f"\nplate solving: {counts['solved']} solved, {counts['failed']} failed, "
              f"{counts['missing']} no longer on disk "
              f"({sum(len(e.get('files', [])) for e in dataset) - len(rows)} skipped)",
              file=sys.stderr)
    return counts


def process_file_safe(path: Path, calib=None, opts=None):
    """process_file, but a failure becomes an error row instead of aborting the run."""
    try:
        return process_file(path, calib, opts)
    except Exception as e:
        return {"path": str(path), "name": path.name, "error": str(e) or repr(e)}


def add_derived_adu_fields(row):
    """Backfill derived fields for rows loaded from older JSON caches, and the arcsec
    figures, which depend on a plate solve that may have been added since the scan."""
    if "adu_mean" not in row:
        return row
    row.setdefault("signal_adu", row["adu_mean"] - (row.get("pedestal_adu") or BIAS_ADU))
    row.setdefault("has_trail", False)
    row.setdefault("elong_n", 0)
    row.setdefault("elong_uniform_r", None)
    row.setdefault("elong_uniform_angle_deg", None)
    row.setdefault("elong_radial_r", None)
    row.setdefault("elong_radial_diff_deg", None)
    row.setdefault("calibration_warnings", [])
    exposure = row.get("exposure")
    if "signal_adu_per_sec" not in row:
        row["signal_adu_per_sec"] = (
            row["signal_adu"] / exposure
            if isinstance(exposure, (int, float)) and exposure > 0
            else None
        )
    # A solved scale is measured; the optics cards are only what the header claims.
    scale = row.get("solved_scale_arcsec") or row.get("pixel_scale_arcsec")
    row["scale_arcsec"] = scale
    for field in ("hfr", "fwhm"):
        px = row.get(f"{field}_avg")
        row[f"{field}_arcsec"] = px * scale if scale and isinstance(px, (int, float)) else None
    return row


def add_relative_fields(dataset):
    """rel_stars / rel_fwhm / rel_sky: each frame against the median of its own set - same
    panel, filter and exposure, across every session. A set too small to have a meaningful
    median gets None, which compute_quality skips."""
    sets = {}
    for entry in dataset:
        parsed = parse_folder(entry.get("folder", ""))
        panel, filt = (parsed[1], parsed[2]) if parsed else (entry.get("folder", ""), "")
        for row in entry.get("files", []):
            if "adu_mean" in row:
                sets.setdefault((panel, filt, exposure_key(row.get("exposure"))), []).append(row)
    for rows in sets.values():
        for rel, field in (("rel_stars", "num_stars"), ("rel_fwhm", "fwhm_avg"),
                           ("rel_sky", "signal_adu_per_sec")):
            values = [r[field] for r in rows if isinstance(r.get(field), (int, float))]
            median = float(np.median(values)) if len(values) >= QUALITY_RELATIVE_MIN_SET else 0.0
            for r in rows:
                v = r.get(field)
                r[rel] = float(v) / median if median > 0 and isinstance(v, (int, float)) else None
    return dataset


def add_dataset_derived_fields(dataset):
    """Backfill, relative metrics, then the quality score - recomputed on every load, so a
    change to the thresholds above takes effect without re-measuring a single frame."""
    for entry in dataset:
        for row in entry.get("files", []):
            add_derived_adu_fields(row)
    add_relative_fields(dataset)
    for entry in dataset:
        for row in entry.get("files", []):
            if "adu_mean" in row:
                row["quality_score"], row["quality_reasons"], row["quality_verdict"] = \
                    compute_quality(row)
    return dataset


def _json_clean(obj):
    """obj with every NaN/inf float replaced by None.

    json.dumps writes them as the bare words NaN/Infinity, which Python and browsers
    accept but strict readers (jq, most other languages) reject outright - a folder whose
    frames all failed to read used to make the whole cache unreadable to them.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_clean(v) for v in obj]
    if isinstance(obj, np.generic):
        return _json_clean(obj.item())
    return obj


def write_json_atomic(path: Path, obj, indent=2):
    """Write strict JSON via a temp file + rename, so an interrupted write - Ctrl+C, a
    full disk - leaves the previous file intact instead of a truncated one."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(_json_clean(obj), indent=indent, allow_nan=False),
                   encoding="utf-8")
    os.replace(tmp, path)


_PROGRESS_LAST = [0.0]


def _report_progress(done, total, t0, name):
    elapsed = time.time() - t0
    if sys.stderr.isatty():
        print(f"\r[{done}/{total}] {elapsed:6.1f}s  {name[:60]:<60}", end="", file=sys.stderr)
        return
    # Redirected to a log: a carriage-return line per frame is thousands of lines of
    # noise there, so report every ~10 s and at the end instead.
    if done == total or time.time() - _PROGRESS_LAST[0] >= 10.0:
        _PROGRESS_LAST[0] = time.time()
        print(f"[{done}/{total}] {elapsed:6.1f}s", file=sys.stderr)


def _available_memory_bytes():
    """Memory free for new work, or None where the platform will not say."""
    try:
        import psutil  # type: ignore[import-not-found]
        return int(psutil.virtual_memory().available)
    except ImportError:
        pass
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError):
        pass
    try:   # macOS has no SC_AVPHYS_PAGES; fall back to a share of physical memory
        return int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") * 0.5)
    except (AttributeError, ValueError, OSError):
        return None


def memory_capped_workers(requested, paths):
    """`requested` workers (CPU count when None), but no more than memory can hold.

    Each worker holds several full-frame float copies at its peak - the frame, the
    flattened frame, the float64 convolution, the segmentation - about
    WORKER_BYTES_PER_PIXEL per pixel. On a 60 Mpx sensor that is gigabytes a worker, and a
    worker per core ran a 16-core machine out of memory. An explicit -w is honoured.
    """
    if requested is not None:
        return requested
    cpus = os.cpu_count() or 1
    if not paths:
        return cpus
    try:
        header = first_image_header(paths[0])
        pixels = int(header["NAXIS1"]) * int(header["NAXIS2"])
    except (OSError, ValueError, KeyError):
        return cpus
    available = _available_memory_bytes()
    if not available:
        return cpus
    fit = max(1, int(available * 0.8 // (pixels * WORKER_BYTES_PER_PIXEL)))
    if fit < cpus:
        print(f"scan: {fit} worker(s) instead of {cpus} - each needs "
              f"~{pixels * WORKER_BYTES_PER_PIXEL / 1e9:.1f} GB for these "
              f"{pixels / 1e6:.0f} Mpx frames", file=sys.stderr)
    return min(cpus, fit)


def _ignore_sigint():
    """Pool initializer: Ctrl+C is the parent's to handle. A worker that took it too died
    mid-frame, broke the pool, and the parent fell back to finishing the scan serially."""
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _make_pool(workers):
    """Processes by default. A spawned worker re-imports __main__, which does not exist
    when build_dataset is driven from a REPL or notebook - fall back to threads there;
    numpy and photutils drop the GIL for the heavy work, so it still parallelises."""
    if getattr(sys.modules.get("__main__"), "__file__", None) is None:
        return ThreadPoolExecutor(max_workers=workers)
    # The workers already fill every core, one frame each; a BLAS that also starts a thread
    # per core in every worker (the PSF fit's batched solves) only thrashes. Spawned workers
    # import numpy afresh, so the environment they inherit is what their BLAS sees.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, "1")
    return ProcessPoolExecutor(max_workers=workers, initializer=_ignore_sigint)


def process_paths(paths, progress=True, workers=None, calib=None, opts=None, on_result=None):
    """Measure every path, in input order. Frames are independent, so fan them out
    across workers: the per-frame cost is numpy/photutils work on a ~9 Mpx array.

    on_result(index, row) is called in the parent as each row arrives - build_dataset uses
    it to checkpoint progress, so an interrupted scan does not start over.
    """
    total = len(paths)
    t0 = time.time()
    workers = memory_capped_workers(workers, paths)
    workers = max(1, min(workers, total)) if total else 1

    results: list[Any] = [None] * total   # rows land here as they come back, in input order

    def land(i, row):
        results[i] = row
        if on_result:
            on_result(i, row)

    if workers > 1:
        pool = _make_pool(workers)
        try:
            pending = {pool.submit(process_file_safe, p, calib, opts): i
                       for i, p in enumerate(paths)}
            for done, future in enumerate(as_completed(pending), 1):
                i = pending[future]
                land(i, future.result())
                if progress:
                    _report_progress(done, total, t0, paths[i].name)
            pool.shutdown()
            return results
        except KeyboardInterrupt:
            # Not a `with` block: leaving one waits for every queued frame to finish.
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        except Exception as e:  # pool unavailable / broken -> still produce a report
            pool.shutdown(wait=False, cancel_futures=True)
            # Keep whatever already came back; re-measuring those frames serially can
            # cost minutes on a large run.
            print(f"\nparallel scan failed ({e}); finishing the rest serially", file=sys.stderr)

    done = sum(r is not None for r in results)
    for i, path in enumerate(paths):
        if results[i] is not None:
            continue
        land(i, process_file_safe(path, calib, opts))
        done += 1
        if progress:
            _report_progress(done, total, t0, path.name)
    return results


def _reusable(prev, path: Path, calib, opts):
    """True when a cached row would come out the same if `path` were measured again."""
    if not prev or "adu_mean" not in prev or not prev.get("source_stat"):
        return False
    try:
        if _stat_of(path) != list(prev["source_stat"]):
            return False
    except OSError:
        return False
    plan = calibration_plan(calib, path, prev.get("exposure"), prev.get("filter"),
                            prev.get("settings") or {})
    return frame_signature(plan, opts) == prev.get("scan_sig")


def load_previous_rows(*sources):
    """{path: row} from earlier caches/checkpoints; later sources win."""
    rows = {}
    for source in sources:
        if not source or not Path(source).exists():
            continue
        try:
            data = json.loads(Path(source).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        entries = data.get("rows", []) if isinstance(data, dict) else \
            [r for entry in data for r in entry.get("files", [])]
        for row in entries:
            if isinstance(row, dict) and row.get("path"):
                rows[row["path"]] = row
    return rows


def build_dataset(root: Path, progress=True, workers=None, darks_dir=None, calibrate=True,
                  calibrated_dir=None, solver=None, full_scale=None, previous=None,
                  checkpoint=None):
    """Measure every LIGHT frame under root into the folder-grouped dataset.

    previous: {path: row} from an earlier run. A frame whose file, masters and options are
    unchanged keeps its row instead of being measured again (see _reusable), which turns a
    rescan after adding one night into seconds. checkpoint: a path that measured rows are
    flushed to every CHECKPOINT_INTERVAL_S, so Ctrl+C - or a crash - loses at most that
    much; the next run picks them up through `previous`.
    """
    scanned = scan_light_dirs(root, exclude=[calibrated_dir] if calibrated_dir else ())
    paths = [p for files in scanned.values() for p in files]
    # Masters are built here, in the parent, so the workers only memory-map them.
    calib = (prepare_calibration(root, darks_dir, progress=progress,
                                 calibrated_dir=calibrated_dir) if calibrate else None)
    # The pedestal behind any frame that ends up without a dark or bias subtracted: the
    # master bias' own level where there is one (even under --no-calibrate), else BIAS_ADU.
    opts = {"pedestal": calib.get("bias_level") if calib else master_bias_level(darks_dir),
            "full_scale": full_scale}

    previous = previous or {}
    rows_flat: list[Any] = [None] * len(paths)
    todo = []
    for i, path in enumerate(paths):
        prev = previous.get(str(path))
        if _reusable(prev, path, calib, opts):
            rows_flat[i] = prev
        else:
            todo.append(i)
    if progress and previous:
        print(f"scan: {len(paths) - len(todo)} frame(s) unchanged since the last scan, "
              f"{len(todo)} to measure", file=sys.stderr)

    last_flush = [time.time()]

    def flush():
        if checkpoint:
            write_json_atomic(checkpoint, {"rows": [r for r in rows_flat if r is not None]},
                              indent=None)
        last_flush[0] = time.time()

    def on_result(j, row):
        rows_flat[todo[j]] = row
        if checkpoint and time.time() - last_flush[0] >= CHECKPOINT_INTERVAL_S:
            flush()

    try:
        process_paths([paths[i] for i in todo], progress=progress, workers=workers,
                      calib=calib, opts=opts, on_result=on_result)
    except KeyboardInterrupt:
        flush()
        if progress:
            done = sum(r is not None for r in rows_flat)
            print(f"\ninterrupted: {done}/{len(paths)} frames kept"
                  f"{' in ' + str(checkpoint) if checkpoint else ''}; "
                  f"run the same command again to continue", file=sys.stderr)
        raise
    if progress and todo:
        print(file=sys.stderr)

    dataset = []
    cursor = 0
    for light_dir, files in scanned.items():
        rows = rows_flat[cursor:cursor + len(files)]
        cursor += len(files)

        measured = [r for r in rows if "adu_mean" in r]
        means = [r["adu_mean"] for r in measured]
        folder_mean = float(np.mean(means)) if means else None
        for r in measured:
            add_derived_adu_fields(r)
        # Deviation is measured against frames of the same exposure length, not against
        # the whole folder: a folder holding 30 s and 45 s subs has two different sky
        # levels in it, and differencing against their blend reads every frame as a large
        # outlier in whichever direction its own length sits.
        for exp_rows in split_by_exposure(measured).values():
            exp_means = [r["adu_mean"] for r in exp_rows]
            exp_mean = float(np.mean(exp_means))
            for r in exp_rows:
                r["adu_dev"] = r["adu_mean"] - exp_mean

        dataset.append({
            "folder": light_dir.relative_to(root).as_posix(),
            "folder_mean_adu": folder_mean,
            "files": rows,
        })
    if solver:
        solve_dataset(dataset, solver, progress=progress)
    warned = sum(1 for r in rows_flat if r and r.get("calibration_warnings"))
    if progress and warned:
        print(f"calibration: {warned} frame(s) carry calibration warnings "
              f"(shown in the Cal column's tooltip)", file=sys.stderr)
    return add_dataset_derived_fields(dataset)


def fmt(v, digits=1):
    if v is None:
        return EM_DASH
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return EM_DASH
    return f"{v:,.{digits}f}"


def fmt_pct(v, digits=1):
    if v is None:
        return EM_DASH
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return EM_DASH
    return f"{v * 100:,.{digits}f}%"


# --- Session / Panel / Filter grouping ---------------------------------------------------
# Folder strings look like "session 01/Pelican Panels Panel 1/B/LIGHT" -> (session, panel, filter)

FILTER_ORDER = ["L", "R", "G", "B", "H", "O", "S"]
# Grounded in what each filter actually passes: R/G/B are literal broadband color
# channels (rendered as the color they reconstruct), L is panchromatic/colorless
# (neutral white-gray), and H/O/S are narrowband filters isolating a specific
# emission line, colored by that line's true wavelength (Halpha 656.3nm, OIII
# 500.7nm, SII 672.4nm - both red, S shifted darker/deeper to reflect its longer
# wavelength since human red-cone response falls off past ~660nm).
FILTER_COLORS = {
    "L": "#e6e6e6",
    "R": "#ff2b2b", "G": "#2ecc71", "B": "#3b82f6",
    "H": "#ff4d4d", "O": "#2dd4bf", "S": "#a1201c",
}
DEFAULT_FILTER_COLOR = "#999999"


def filter_tint_css(filt):
    color = FILTER_COLORS.get(filt, DEFAULT_FILTER_COLOR).lstrip("#")
    if len(color) != 6:
        return "rgba(153,153,153,0.10)"
    try:
        r, g, b = (int(color[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return "rgba(153,153,153,0.10)"
    return f"rgba({r},{g},{b},0.10)"


def natural_num(s):
    """Natural sort key: "Panel 2" before "Panel 10", and names that share a leading number
    ("2024-10-03", "2024-11-12") or carry none at all still order by the rest of the name
    rather than tying and falling back to whatever order they were collected in."""
    return [(0, int(chunk), "") if chunk.isdigit() else (1, 0, chunk.lower())
            for chunk in re.findall(r"\d+|\D+", s)]


def filter_sort_key(f):
    return FILTER_ORDER.index(f) if f in FILTER_ORDER else 99


def exposure_key(value):
    """Canonical identity of a frame's exposure time, as a short string.

    Frames of the same filter shot at different lengths are not comparable by eye: a 45 s
    sub sits at a different sky level and star depth than a 30 s one, so pooling them into
    one thumbnail group fits a shared stretch to a population that has two different
    backgrounds in it and misrepresents both halves. Splitting on this key keeps each
    length its own comparable set. "" when the header carried no usable exposure.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return ""
    return f"{float(value):g}" if value > 0 else ""


def exposure_sort_key(key):
    """Shortest exposure first; frames with no usable exposure last."""
    return (1, 0.0) if not key else (0, float(key))


def exposure_label(key):
    return f"{key}s" if key else "no exposure"


ROOT_LABEL = "(root)"   # the session/panel of a tree scanned from below that level


def parse_folder(folder_str):
    """Split a LIGHT folder path, relative to the scan root, into (session, panel, filter).

    The canonical layout is session/panel/filter/LIGHT, but a root one level deeper (a
    single session) or shallower (a year folder above the sessions) is just as common, and
    used to drop every frame into an ungrouped leftovers list with no matrix entry and no
    relative metrics. The two folders above LIGHT are always panel and filter; whatever
    sits above those is the session (joined, if several levels), or ROOT_LABEL when the
    root itself is the session or panel. None only for a LIGHT folder at the very root.
    """
    parts = [p for p in folder_str.replace("\\", "/").split("/") if p and p != "."]
    if not parts or parts[-1].upper() != "LIGHT":
        return None
    n = len(parts) - 1
    if n >= 3:
        return "/".join(parts[:n - 2]), parts[n - 2], parts[n - 1]
    if n == 2:
        return ROOT_LABEL, parts[0], parts[1]
    if n == 1:
        return ROOT_LABEL, ROOT_LABEL, parts[0]
    return None


def group_by_panel(entries, folder_of):
    """[(panel, [(session, [(filter, entry), ...]), ...]), ...] in natural sort order, plus
    the entries whose folder name did not parse. Shared by the report and the review server
    so both lay panels/sessions/filters out identically."""
    groups = {}
    leftovers = []
    for entry in entries:
        parsed = parse_folder(folder_of(entry))
        if not parsed:
            leftovers.append(entry)
            continue
        session, panel, filt = parsed
        groups.setdefault(panel, {}).setdefault(session, {})[filt] = entry

    result = []
    for panel in sorted(groups, key=natural_num):
        sess_list = []
        for session in sorted(groups[panel], key=natural_num):
            by_filter = groups[panel][session]
            filters = sorted(by_filter, key=filter_sort_key)
            sess_list.append((session, [(f, by_filter[f]) for f in filters]))
        result.append((panel, sess_list))
    return result, leftovers


def group_by_panel_filter(entries, folder_of):
    """[(panel, [(filter, [(session, entry), ...]), ...]), ...] in natural sort order, plus
    the entries whose folder name did not parse. Same grouping as group_by_panel but with
    filter/session swapped, so all sessions for one panel+filter land in a single table."""
    groups = {}
    leftovers = []
    for entry in entries:
        parsed = parse_folder(folder_of(entry))
        if not parsed:
            leftovers.append(entry)
            continue
        session, panel, filt = parsed
        groups.setdefault(panel, {}).setdefault(filt, {})[session] = entry

    result = []
    for panel in sorted(groups, key=natural_num):
        filt_list = []
        for filt in sorted(groups[panel], key=filter_sort_key):
            by_session = groups[panel][filt]
            sessions = sorted(by_session, key=natural_num)
            filt_list.append((filt, [(s, by_session[s]) for s in sessions]))
        result.append((panel, filt_list))
    return result, leftovers


def split_by_exposure(files):
    """{exposure_key: [rows]} for one folder's frames, shortest exposure first.

    A folder is one panel/session/filter, but not necessarily one exposure length - the
    scanner attributes LIGHT/30.00 and LIGHT/45.00 to the same LIGHT folder. Everything
    the review UI compares (a matrix cell, a summary line, a thumbnail grid, a sigma clip)
    is only meaningful within one length, so this is the grain all of them are built on.
    """
    by_exposure = {}
    for f in files:
        by_exposure.setdefault(exposure_key(f.get("exposure")), []).append(f)
    return {k: by_exposure[k] for k in sorted(by_exposure, key=exposure_sort_key)}


def split_sessions_by_exposure(sess_list):
    """{exposure: [(session, [rows])]} from a panel+filter's [(session, entry)] list.

    Sessions keep their order within each exposure, and an exposure only lists the
    sessions that actually carry frames of that length.
    """
    by_exposure = {}
    for session, entry in sess_list:
        for exposure, rows in split_by_exposure(entry["files"]).items():
            by_exposure.setdefault(exposure, []).append((session, rows))
    return {k: by_exposure[k] for k in sorted(by_exposure, key=exposure_sort_key)}


def aggregate_entry(entry):
    """Roll up one folder's per-file rows into per-(panel,session,filter) summary stats."""
    return aggregate_files([f for f in entry["files"] if "error" not in f])


def aggregate_files(files):
    """Summary stats over an arbitrary set of frame rows, or None if there are none."""
    if not files:
        return None
    hfrs = [f["hfr_avg"] for f in files if f.get("hfr_avg") is not None]
    fwhms = [f["fwhm_avg"] for f in files if f.get("fwhm_avg") is not None]
    roundnesses = [f["roundness_avg"] for f in files if f.get("roundness_avg") is not None]
    stars = [f["num_stars"] for f in files if f.get("num_stars") is not None]
    adu_means = [f["adu_mean"] for f in files if f.get("adu_mean") is not None]
    signal_adus = [f["signal_adu"] for f in files if f.get("signal_adu") is not None]
    signal_rates = [f["signal_adu_per_sec"] for f in files if f.get("signal_adu_per_sec") is not None]
    exposures = [f["exposure"] for f in files if isinstance(f.get("exposure"), (int, float))]
    uniform_rs = [f["elong_uniform_r"] for f in files if f.get("elong_uniform_r") is not None]
    radial_rs = [f["elong_radial_r"] for f in files if f.get("elong_radial_r") is not None]
    quality_scores = [f["quality_score"] for f in files if f.get("quality_score") is not None]
    return {
        "frames": len(files),
        "quality": float(np.mean(quality_scores)) if quality_scores else None,
        "total_exp_min": (sum(exposures) / 60.0) if exposures else None,
        "hfr": float(np.median(hfrs)) if hfrs else None,
        "fwhm": float(np.median(fwhms)) if fwhms else None,
        "roundness": float(np.median(roundnesses)) if roundnesses else None,
        "stars": float(np.mean(stars)) if stars else None,
        "adu_mean": float(np.mean(adu_means)) if adu_means else None,
        "signal_adu": float(np.mean(signal_adus)) if signal_adus else None,
        "signal_adu_per_sec": float(np.mean(signal_rates)) if signal_rates else None,
        "adu_spread": float(np.std(adu_means)) if len(adu_means) > 1 else 0.0,
        "elong_uniform_r": float(np.mean(uniform_rs)) if uniform_rs else None,
        "elong_radial_r": float(np.mean(radial_rs)) if radial_rs else None,
    }


def build_comparison(dataset):
    """comp[panel][session][filter][exposure] = aggregated stats, plus the sorted axis
    labels and the (filter, exposure) pairs that make up the matrix rows."""
    comp = {}
    panels_seen, sessions_seen, rows_seen = set(), set(), set()
    for entry in dataset:
        parsed = parse_folder(entry["folder"])
        if not parsed:
            continue
        session, panel, filt = parsed
        files = [f for f in entry["files"] if "error" not in f]
        for exposure, exp_files in split_by_exposure(files).items():
            agg = aggregate_files(exp_files)
            if agg is None:
                continue
            comp.setdefault(panel, {}).setdefault(session, {}).setdefault(filt, {})[exposure] = agg
            rows_seen.add((filt, exposure))
        panels_seen.add(panel)
        sessions_seen.add(session)
    panels = sorted(panels_seen, key=natural_num)
    sessions = sorted(sessions_seen, key=natural_num)
    rows = sorted(rows_seen, key=lambda fe: (filter_sort_key(fe[0]), exposure_sort_key(fe[1])))
    return comp, panels, sessions, rows


def group_entries_by_panel(dataset):
    return group_by_panel(dataset, lambda entry: entry["folder"])


def group_entries_by_panel_filter(dataset):
    return group_by_panel_filter(dataset, lambda entry: entry["folder"])


COMPARISON_METRICS = [
    # (key, label, digits, is_percentage)
    ("quality", "Quality score (avg, 0-100)", 0, False),
    ("hfr", "HFR (median)", 2, False),
    ("fwhm", "FWHM (median, px)", 2, False),
    ("roundness", "Roundness (median, 100%=perfectly round)", 1, True),
    ("stars", "Stars (avg)", 0, False),
    ("adu_mean", "ADU mean (avg)", 0, False),
    ("signal_adu", "Signal ADU (avg, bias subtracted)", 0, False),
    ("signal_adu_per_sec", "Signal ADU/sec (avg)", 2, False),
    ("adu_spread", "ADU spread (σ)", 1, False),
    ("elong_uniform_r", "Elongation: same direction (0-1)", 2, False),
    ("elong_radial_r", "Elongation: radial (coma-like, 0-1)", 2, False),
    ("frames", "Frames", 0, False),
    ("total_exp_min", "Total exposure (min)", 1, False),
]

# Row totals shown after the per-session columns: one row is a single panel+filter, so
# these are that filter's whole take on that panel, summed across every session it was
# shot in. Unlike the metric columns they do not change with the selected metric.
MATRIX_TOTAL_COLUMNS = [
    ("frames", "Total frames"),
    ("exposure", "Total exposure"),
]

# The pages' JavaScript and CSS live in static/ as ordinary files, so they can be linted and
# edited as what they are; both the report and the review page inline them, which keeps
# the report a single self-contained HTML file.
STATIC_DIR = Path(__file__).resolve().parent / "static"


def static_text(name):
    return (STATIC_DIR / name).read_text(encoding="utf-8")


# Shared by the static report and the review server so the two can never disagree about
# what a row total says.
MATRIX_TOTALS_JS = static_text("matrix_totals.js")

MATRIX_CSS = static_text("matrix.css")

COMPARISON_CSS = MATRIX_CSS + static_text("report.css")


def render_metric_buttons(parts):
    parts.append('<div class="metric-row" id="metric-row">')
    for i, (key, label, _digits, _pct) in enumerate(COMPARISON_METRICS):
        active = " active" if i == 0 else ""
        parts.append(
            f'<button class="metric-btn{active}" data-metric="{html.escape(key, quote=True)}" '
            f'onclick="setMatrixMetric(\'{html.escape(key, quote=True)}\')">{html.escape(label)}</button>'
        )
    parts.append('</div>')


def render_matrix_skeleton(parts, panels, sessions, rows, row_extra=None):
    """Empty panel x session table; the cells are filled in by renderMatrix() on the client.

    `rows` is the list of (filter, exposure) pairs one row each - a filter shot at two
    lengths gets a row per length, since frames of different lengths are not comparable
    on any of the metrics the cells show. Every panel carries the same row list so the
    filters stay aligned down the table; a pair a panel was never shot at reads as a dash.
    row_extra(panel, filt, exposure), if given, returns extra HTML appended into the
    filter cell - used by the review server to add per-row thumbnail/statistics actions.
    """
    parts.append('<div class="matrix-wrap"><table class="matrix"><thead><tr><th>Panel</th><th>Filter</th>')
    for session in sessions:
        parts.append(f"<th>{html.escape(session)}</th>")
    for key, label in MATRIX_TOTAL_COLUMNS:
        first = " total-first" if key == MATRIX_TOTAL_COLUMNS[0][0] else ""
        parts.append(f'<th class="matrix-total{first}">{html.escape(label)}</th>')
    parts.append('</tr></thead><tbody>')
    for panel in panels:
        for i, (filt, exposure) in enumerate(rows):
            tint = filter_tint_css(filt)
            parts.append(f'<tr style="--filter-tint:{html.escape(tint, quote=True)}">')
            if i == 0:
                parts.append(f'<td class="matrix-panel" rowspan="{len(rows)}">{html.escape(panel)}</td>')
            color = FILTER_COLORS.get(filt, DEFAULT_FILTER_COLOR)
            parts.append(
                '<td class="matrix-filter"><div class="cell-line"><span class="swatch" style="background:'
                + html.escape(color, quote=True)
                + '"></span><span class="row-filter">'
                + html.escape(filt)
                + '</span><span class="row-exp row-exp-fixed">'
                + html.escape(exposure_label(exposure))
                + '</span>'
                + (row_extra(panel, filt, exposure) if row_extra else '')
                + '</div></td>'
            )
            cell_keys = (f'data-panel="{html.escape(panel, quote=True)}" '
                         f'data-filter="{html.escape(filt, quote=True)}" '
                         f'data-exposure="{html.escape(exposure, quote=True)}"')
            for session in sessions:
                parts.append(
                    f'<td class="matrix-cell" {cell_keys} '
                    f'data-session="{html.escape(session, quote=True)}"></td>'
                )
            for key, _label in MATRIX_TOTAL_COLUMNS:
                first = " total-first" if key == MATRIX_TOTAL_COLUMNS[0][0] else ""
                parts.append(f'<td class="matrix-total{first}" {cell_keys} data-total="{key}"></td>')
            parts.append('</tr>')
    parts.append('</tbody></table></div>')


def json_for_script(obj):
    """json.dumps safe to embed in an inline <script>: a "</script>" inside a string
    would otherwise terminate the block early."""
    return json.dumps(_json_clean(obj)).replace("</", "<\\/")


def build_comparison_section(dataset):
    comp, panels, sessions, rows = build_comparison(dataset)
    if not panels:
        return ""

    parts = [f"<style>{COMPARISON_CSS}</style>"]
    parts.append('<h2 style="margin-top:0">Panel &times; Session comparison</h2>')
    parts.append('<div class="meta">Click a metric to switch what the matrix shows. Each row is one filter at one exposure length, aligned across sessions.</div>')
    render_metric_buttons(parts)
    render_matrix_skeleton(parts, panels, sessions, rows)

    parts.append("<script>" + MATRIX_TOTALS_JS
                 + "\nconst COMP = " + json_for_script(comp) + ";\n"
                 + "const COMP_METRICS = " + json_for_script(COMPARISON_METRICS) + ";\n"
                 + static_text("comparison.js") + "</script>")

    return "\n".join(parts)


TABLE_HEADERS = ["File", "Filter", "Quality", "Exp (s)", "Cal", "ADU min", "ADU max", "ADU mean",
                 "Signal ADU", "Signal ADU/s",
                 "Δ mean vs set", "HFR avg", "FWHM avg (px)", "# Stars", "Roundness", "Elongation", "Trail"]


def fmt_elongation(r):
    """One-line verdict for the elongation-direction stats: whichever of "same direction
    everywhere" (tracking/PE) or "radially outward" (coma/tilt) the frame's elongated stars
    agree on more strongly, or an em-dash if there were too few elongated stars to say."""
    uniform_r, radial_r = r.get("elong_uniform_r"), r.get("elong_radial_r")
    if uniform_r is None or radial_r is None:
        return EM_DASH
    if uniform_r >= radial_r:
        return f"tracking r={uniform_r:.2f}"
    return f"radial r={radial_r:.2f}"


CALIBRATION_HINT = {
    "Cal": "Calibration applied: D = master dark, B = master bias, F = master flat",
    "ADU min": "As recorded by the sensor, before calibration",
    "ADU max": "As recorded by the sensor, before calibration - 65535 means a star hit the ceiling",
    "ADU mean": "As recorded by the sensor, before calibration",
    "Signal ADU": "Mean of the calibrated frame: the sky signal with the pedestal removed",
    "Signal ADU/s": "Calibrated sky signal per second of exposure",
}


def fmt_calibration(r):
    """Compact tag for what was subtracted and divided out, e.g. dark + flat -> "D+F"."""
    label = (r.get("calibration") or "").lower()
    parts = [c for c, token in (("D", "dark"), ("B", "bias"), ("F", "flat")) if token in label]
    return "+".join(parts)


def fmt_quality(r):
    """<td> for the composite quality score: a sortable number, colored by verdict,
    with the deduction reasons as a hover tooltip."""
    score, verdict = r.get("quality_score"), r.get("quality_verdict")
    if score is None:
        return f'<td>{EM_DASH}</td>'
    reasons = r.get("quality_reasons") or []
    title = html.escape("; ".join(reasons), quote=True) if reasons else "no deductions"
    return f'<td class="q-{html.escape(verdict or "", quote=True)}" title="{title}">{score:.0f}</td>'


def _calibration_cell(r):
    """The Cal cell: the D/B/F tag, with what was applied - and any master that was
    skipped for not matching the frame - in its tooltip, flagged with a "!"."""
    warnings = r.get("calibration_warnings") or []
    title = str(r.get("calibration") or "not calibrated")
    if warnings:
        title += "\n" + "\n".join(f"! {w}" for w in warnings)
    tag = fmt_calibration(r) or EM_DASH
    flag = ' <span class="cal-warn">!</span>' if warnings else ""
    return f'<td class="cal-tag" title="{html.escape(title, quote=True)}">{tag}{flag}</td>'


def _arcsec_cell(r, field):
    """An HFR/FWHM cell in pixels, with the angular size in the tooltip when known."""
    px, arcsec = r.get(f"{field}_avg"), r.get(f"{field}_arcsec")
    title = f' title="{arcsec:.2f}&quot; at {r["scale_arcsec"]:.2f}&quot;/px"' \
        if isinstance(arcsec, (int, float)) and r.get("scale_arcsec") else ""
    return f"<td{title}>{fmt(px, 2)}</td>"


def render_row_cells(r):
    """The <td> cells for one frame, in TABLE_HEADERS order. Shared with the review server
    so the static report and the live table can never drift apart."""
    dev = r.get("adu_dev")
    dev_cls = ""
    dev_str = fmt(dev, 1)
    if isinstance(dev, (int, float)) and not math.isnan(dev):
        dev_cls = "dev-pos" if dev > 0 else ("dev-neg" if dev < 0 else "")
        dev_str = f"{'+' if dev > 0 else ''}{dev:,.1f}"

    exp = r.get("exposure")
    if isinstance(exp, (int, float)) and not isinstance(exp, bool):
        exp_str = f"{exp:g}"
    else:
        exp_str = html.escape(str(exp)) if exp else EM_DASH

    return [
        f'<td title="{html.escape(str(r.get("path", "")), quote=True)}">{html.escape(str(r.get("name", "")))}</td>',
        f'<td>{html.escape(r.get("filter") or "") or EM_DASH}</td>',
        fmt_quality(r),
        f'<td>{exp_str}</td>',
        _calibration_cell(r),
        f'<td>{fmt(r.get("adu_min"), 0)}</td>',
        f'<td>{fmt(r.get("adu_max"), 0)}</td>',
        f'<td>{fmt(r.get("adu_mean"), 1)}</td>',
        f'<td>{fmt(r.get("signal_adu"), 1)}</td>',
        f'<td>{fmt(r.get("signal_adu_per_sec"), 2)}</td>',
        f'<td class="dev-cell {dev_cls}">{dev_str}</td>',
        _arcsec_cell(r, "hfr"),
        _arcsec_cell(r, "fwhm"),
        f'<td>{int(r.get("num_stars") or 0)}</td>',
        f'<td>{fmt_pct(r.get("roundness_avg"), 1)}</td>',
        f'<td>{html.escape(fmt_elongation(r))}</td>',
        f'<td class="trail-flag">{"⚠ trail" if r.get("has_trail") else EM_DASH}</td>',
    ]


def render_table_head(parts, with_session=False):
    parts.append('<div class="wrap"><table class="sortable-table"><thead><tr>')
    if with_session:
        parts.append('<th class="sortable">Session</th>')
    parts.extend(_table_header_cell(h) for h in TABLE_HEADERS)
    parts.append("</tr></thead><tbody>")


def _table_header_cell(header):
    hint = CALIBRATION_HINT.get(header)
    title = f' title="{html.escape(hint, quote=True)}"' if hint else ""
    return f'<th class="sortable"{title}>{html.escape(header)}</th>'


def render_folder_table(entry):
    files = entry["files"]
    parts = [f'<div class="meta">{len(files)} light frame(s) &nbsp;&middot;&nbsp; '
             f'folder mean ADU: {fmt(entry["folder_mean_adu"])}</div>']
    render_table_head(parts)

    for r in files:
        if "error" in r:
            parts.append(
                f'<tr><td>{html.escape(str(r.get("name", "")))}</td>'
                f'<td class="err" colspan="{len(TABLE_HEADERS) - 1}">read error: {html.escape(str(r["error"]))}</td></tr>'
            )
            continue
        parts.append("<tr>")
        parts.extend(render_row_cells(r))
        parts.append("</tr>")

    parts.append("</tbody></table></div>")
    return parts


def render_sessions_table(sess_list):
    """One table for a panel+filter+exposure combining every session's frames, with a
    leading Session column so rows stay comparable across sessions.

    sess_list is [(session, [rows])]: already narrowed to one exposure length by the
    caller, since frames of different lengths do not compare on ADU or signal.
    """
    total = sum(len(rows) for _, rows in sess_list)
    parts = [f'<div class="meta">{total} light frame(s) across {len(sess_list)} session(s)</div>']
    render_table_head(parts, with_session=True)

    for session, rows in sess_list:
        for r in rows:
            if "error" in r:
                parts.append(
                    f'<tr><td>{html.escape(session)}</td><td>{html.escape(str(r.get("name", "")))}</td>'
                    f'<td class="err" colspan="{len(TABLE_HEADERS) - 1}">read error: {html.escape(str(r["error"]))}</td></tr>'
                )
                continue
            parts.append("<tr>")
            parts.append(f"<td>{html.escape(session)}</td>")
            parts.extend(render_row_cells(r))
            parts.append("</tr>")

    parts.append("</tbody></table></div>")
    return parts


BASE_CSS = static_text("base.css")

SORT_TABLE_JS = static_text("sort_table.js")


def build_html(dataset, root_name):
    parts = [f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>FITS LIGHT frame report — {html.escape(root_name)}</title>
<style>{BASE_CSS}
body {{ padding:24px 32px 64px; }}
</style></head><body>
<h1>FITS LIGHT frame report</h1>
<div class="sub">Root: {html.escape(root_name)} &nbsp;&middot;&nbsp; Generated {html.escape(time.strftime('%Y-%m-%d %H:%M'))}</div>
"""]

    parts.append(build_comparison_section(dataset))

    grouped, leftovers = group_entries_by_panel_filter(dataset)
    for panel, filt_list in grouped:
        parts.append(f'<h2>{html.escape(panel)}</h2>')
        for filt, sess_list in filt_list:
            # One section per exposure length: a 30 s and a 45 s sub of the same filter
            # sit at different ADU and signal levels, so a shared table invites comparing
            # numbers that were never comparable.
            for exposure, per_session in split_sessions_by_exposure(sess_list).items():
                parts.append(f'<h3>Filter {html.escape(filt)} '
                             f'<span class="row-exp">{html.escape(exposure_label(exposure))}</span></h3>')
                parts.extend(render_sessions_table(per_session))

    for entry in leftovers:
        parts.append(f'<h2>{html.escape(entry["folder"])}</h2>')
        parts.extend(render_folder_table(entry))

    parts.append(f"<script>{SORT_TABLE_JS}</script>")
    parts.append("</body></html>")
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", nargs="?", default=".", help="Root folder to scan for LIGHT subfolders")
    ap.add_argument("-o", "--output", default="fits_light_report.html", help="Output HTML file")
    ap.add_argument("-j", "--json-cache", default=None, help="Optional path to dump/read raw results as JSON")
    ap.add_argument("-w", "--workers", type=int, default=None,
                    help="Parallel worker processes (default: CPU count, capped by free memory; "
                         "1 = serial)")
    ap.add_argument("--rescan", action="store_true",
                    help="Scan again even though the JSON cache exists. Frames whose file, "
                         "masters and options are unchanged keep their cached measurements")
    ap.add_argument("--full-rescan", action="store_true",
                    help="Like --rescan, but re-measure every frame")
    ap.add_argument("--darks", default=None, metavar="DIR",
                    help="Folder of PixInsight master bias/dark XISF files. Darks are matched "
                         "on exact exposure (and gain/offset/binning where known); the "
                         "matching bias is used when no dark matches.")
    ap.add_argument("--no-calibrate", action="store_true",
                    help="Measure the raw frames, ignoring master darks and any FLAT folders")
    ap.add_argument("--full-scale", type=float, default=None, metavar="ADU",
                    help="The sensor's full-scale ADU (default: the SATURATE card, else 65535). "
                         "Set e.g. 4095 or 16383 for 12/14-bit data that is not scaled to 16 bit")
    ap.add_argument("--solve", action="store_true",
                    help="Plate solve every frame with ASTAP and record RA/Dec, rotation, "
                         "pixel scale and how far the mount's own pointing was off")
    ap.add_argument("--write-wcs", action="store_true",
                    help="With --solve, also write the WCS into each frame's own FITS header "
                         "(pixel data is untouched; the original header is kept in "
                         f"<root>/{WCS_BACKUP_DIRNAME} for --undo-wcs)")
    ap.add_argument("--undo-wcs", action="store_true",
                    help="Put back every original header saved by --write-wcs, then exit")
    ap.add_argument("--solve-only", action="store_true",
                    help="Plate solve the frames already in the JSON cache (-j) and add the "
                         "results to it, without re-measuring anything. Frames that already "
                         "have a solution are skipped unless --resolve is given")
    ap.add_argument("--resolve", action="store_true",
                    help="With --solve-only, solve frames that already have a solution again")
    ap.add_argument("--astap", default=None, metavar="PATH",
                    help="Path to the ASTAP binary, if it is not in a standard location")
    ap.add_argument("--astap-db", default=None, metavar="DIR",
                    help="Path to the ASTAP star database folder")
    ap.add_argument("--save-calibrated", nargs="?", const="calibrated", default=None,
                    metavar="DIR",
                    help="Also write each calibrated frame as float32 FITS, mirroring the "
                         "source tree (default folder: <root>/calibrated). ~36 MB a frame.")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    cache = Path(args.json_cache) if args.json_cache else None
    if args.undo_wcs:
        undo_wcs(root)
        return
    rescan = args.rescan or args.full_rescan

    if args.solve_only:
        if not cache or not cache.exists():
            sys.exit("--solve-only needs an existing JSON cache (-j); run a scan first")
        dataset = json.loads(cache.read_text(encoding="utf-8"))
        solver = prepare_solver(args.astap, args.astap_db, update_header=args.write_wcs,
                                root=root)
        if solver is None:
            sys.exit("--solve-only: ASTAP not found (pass --astap PATH)")
        solve_dataset(dataset, solver, workers=args.workers, redo=args.resolve)
        dataset = add_dataset_derived_fields(dataset)
        write_json_atomic(cache, dataset)
    elif cache and cache.exists() and not rescan:
        print(f"Using cached measurements from {cache} (pass --rescan to pick up new or "
              f"changed frames)", file=sys.stderr)
        if args.solve or args.write_wcs or args.save_calibrated:
            # These only act while frames are being measured; quietly dropping them left
            # e.g. --solve looking like it ran and simply found nothing. (--darks is left
            # out: run_light_stats.sh always passes it.)
            print("  note: --solve/--write-wcs/--save-calibrated have no effect without "
                  "--rescan (or use --solve-only)", file=sys.stderr)
        dataset = add_dataset_derived_fields(json.loads(cache.read_text(encoding="utf-8")))
    else:
        calibrated_dir = None
        if args.save_calibrated:
            calibrated_dir = Path(args.save_calibrated)
            if not calibrated_dir.is_absolute():
                calibrated_dir = root / calibrated_dir
        solver = None
        if args.solve or args.write_wcs:
            solver = prepare_solver(args.astap, args.astap_db, update_header=args.write_wcs,
                                    root=root)
        # Rows measured so far by an interrupted run, kept next to the cache.
        checkpoint = (cache.with_name(cache.name + ".partial") if cache
                      else root / ".fits_light_stats.partial.json")
        previous = {} if args.full_rescan else load_previous_rows(cache, checkpoint)
        try:
            dataset = build_dataset(root, workers=args.workers, darks_dir=args.darks,
                                    calibrate=not args.no_calibrate,
                                    calibrated_dir=calibrated_dir, solver=solver,
                                    full_scale=args.full_scale, previous=previous,
                                    checkpoint=checkpoint)
        except KeyboardInterrupt:
            sys.exit(130)
        if cache:
            write_json_atomic(cache, dataset)
        checkpoint.unlink(missing_ok=True)

    # The report carries em dashes, sigmas and filter glyphs, so never leave the
    # encoding up to the platform default.
    Path(args.output).write_text(build_html(dataset, root.name), encoding="utf-8")
    print(f"Wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
