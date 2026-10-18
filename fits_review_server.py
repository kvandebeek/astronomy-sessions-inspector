#!/usr/bin/env python3
"""
Local interactive review server for the FITS LIGHT-frame report.

Serves the same per-folder tables as fits_light_stats.py, but each row is
clickable: it opens a live-rendered preview of the frame (linear or
auto-stretch) and lets you delete it. "Delete" moves the file into a
"_deleted" folder next to the report (mirroring its original relative path)
rather than erasing it, so it can be recovered.

Usage:
    source .venv/bin/activate
    python3 fits_review_server.py [-j fits_light_data.json] [-p 8765]

Then open http://127.0.0.1:8765 in a browser. The server only binds to
localhost and only ever reads/moves files that were part of the scanned
dataset (no arbitrary path access).
"""
import argparse
import io
import json
import math
import os
import re
import shutil
import sys
import threading
import time
import html
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, parse_qs

import numpy as np
from astropy.io import fits
from astropy.visualization import AsymmetricPercentileInterval
from PIL import Image

import fits_light_stats as stats

PREVIEW_MAX_SIZE = 900
THUMB_MAX_SIZE = 320         # grid thumbnails render smaller than the single-frame preview
PREVIEW_CACHE_SIZE = 32      # rendered PNGs kept in memory; a frame costs ~0.3 s to re-render
THUMB_CACHE_SIZE = 256       # a full panel/filter grid can hold well over 32 frames at once
HISTOGRAM_CACHE_SIZE = 64
GROUP_STRETCH_CACHE_SIZE = 64
TRAIL_CACHE_SIZE = 64        # located trails; ~0.7 s of re-measurement each
DECIMATED_PREVIEW_CACHE_SIZE = 8     # decimated frames held for reuse, ~3 MB each at 900 px
DECIMATED_THUMB_CACHE_SIZE = 128     # ~0.4 MB each at 320 px
FULL_FRAME_CACHE_SIZE = 3            # undecimated frames for 1:1 inspection, ~18 MB each
CROP_MAX_SIZE = 2200                 # largest 1:1 crop a client may ask for, per axis
CROP_CACHE_SIZE = 24                 # rendered 1:1 crops; panning revisits them constantly
GROUP_STRETCH_SAMPLE_FRAMES = 12     # frames pooled when fitting one group's shared stretch
SOURCE_RAW, SOURCE_CALIBRATED = "raw", "calibrated"
SOURCES = (SOURCE_RAW, SOURCE_CALIBRATED)
DELETED_SUBDIR = "_deleted"
STATE_FILENAME = "fits_review_state.json"
MAX_POST_BYTES = 64 * 1024


class Dataset:
    """The scanned frames plus their deleted/restored state. Mutating methods are
    serialised: ThreadingHTTPServer runs each request on its own thread."""

    def __init__(self, json_path: Path, root: Path):
        self.root = root
        self.deleted_dir = root / DELETED_SUBDIR
        self.lock = threading.Lock()
        self.raw = stats.add_dataset_derived_fields(
            json.loads(json_path.read_text(encoding="utf-8")))

        self.folders = []
        self.files = []  # flat list, index == id
        for folder_entry in self.raw:
            folder_name = folder_entry["folder"]
            parsed = stats.parse_folder(folder_name)
            folder_id = len(self.folders)
            file_ids = []
            for f in folder_entry["files"]:
                if "error" in f:
                    continue
                f = dict(f)
                f["id"] = len(self.files)
                f["folder_id"] = folder_id
                f["deleted"] = False
                # Precomputed here rather than derived in the browser: the key has to be
                # the same string the thumbnail URLs and group lists are built from, and
                # Python's %g and JS's number formatting are not the same function.
                f["exp_key"] = stats.exposure_key(f.get("exposure"))
                # Written by a scan run with --save-calibrated. Checked once here rather
                # than per request; the client only needs to know it is there.
                calibrated = f.get("calibrated_path")
                f["has_calibrated"] = bool(calibrated and Path(calibrated).exists())
                self.files.append(f)
                file_ids.append(f["id"])
            self.folders.append({
                "id": folder_id,
                "folder": folder_name,
                "folder_mean_adu": folder_entry["folder_mean_adu"],
                "file_ids": file_ids,
                "session": parsed[0] if parsed else "",
                "panel": parsed[1] if parsed else "",
                "filter": parsed[2] if parsed else "",
            })

        self.by_path = {f["path"]: f for f in self.files}
        self.units = self.build_units()
        self.state_path = root / STATE_FILENAME
        self._load_state()

    def _load_state(self):
        """State is keyed by source path. Older state files keyed by list index are
        migrated on load - indices silently point at the wrong frame as soon as the
        dataset JSON is rescanned."""
        if not self.state_path.exists():
            return
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            print(f"Ignoring unreadable {self.state_path}: {e}", file=sys.stderr)
            return

        for key, info in (state.get("deleted") or {}).items():
            if not isinstance(info, dict) or not info.get("deleted_path"):
                continue
            f = self.by_path.get(info.get("path") or key)
            if f is None and key.lstrip("-").isdigit():
                idx = int(key)
                f = self.files[idx] if 0 <= idx < len(self.files) else None
            if f is None:
                continue
            f["deleted"] = True
            f["deleted_path"] = info["deleted_path"]

        # The state file records where the server put a frame, not where it is now. If
        # someone moved one back by hand, believe the disk: otherwise every preview of
        # that frame 404s with no way back from the UI.
        #
        # Only if it really is back, though. Checking the _deleted copy alone meant that
        # emptying _deleted - the obvious way to reclaim the space after a review -
        # un-deleted every frame in it, and each came back as an ordinary row pointing at
        # a source file that was equally gone: a broken thumbnail with no way to act on it.
        # A frame at neither path is gone, which is what "deleted" already means, so it
        # stays marked; `missing` says it can no longer be restored.
        for f in self.files:
            if not f.get("deleted"):
                continue
            if Path(f["path"]).exists():
                f["deleted"] = False
                f.pop("deleted_path", None)
            elif not Path(f["deleted_path"]).exists():
                f["missing"] = True

    def save_state(self):
        """Written via a temp file + replace: this is the only record of where a
        deleted frame went, so a truncated write would strand the files."""
        payload = {
            "deleted": {
                f["path"]: {"path": f["path"], "deleted_path": f["deleted_path"]}
                for f in self.files if f.get("deleted")
            }
        }
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, self.state_path)

    def effective_source(self, fid: int, source: str) -> str:
        """The source a frame can actually be shown in.

        A frame with no calibrated copy falls back to raw. That fallback has to be visible
        to the caller, because the two sit at completely different ADU levels: pooling a
        fallback frame into a calibrated group's shared stretch fits one interval across a
        ~2900 ADU background and a ~55 ADU one, and every frame in the grid comes out
        crushed to black or blown to white.
        """
        if source == SOURCE_CALIBRATED and self.files[fid].get("has_calibrated"):
            return SOURCE_CALIBRATED
        return SOURCE_RAW

    def current_path(self, fid: int, source: str = SOURCE_RAW) -> Path:
        """The file a render should read for this frame.

        The calibrated copy is a derived artefact that delete/restore never moves, so it
        is addressed directly and only when it is actually on disk; everything else falls
        back to wherever the original currently lives.
        """
        f = self.files[fid]
        if source == SOURCE_CALIBRATED and f.get("has_calibrated"):
            return Path(f["calibrated_path"])
        return Path(f["deleted_path"] if f.get("deleted") else f["path"])

    @staticmethod
    def _free_destination(dst: Path) -> Path:
        """Never let a move clobber an existing file."""
        if not dst.exists():
            return dst
        for n in range(1, 1000):
            candidate = dst.with_name(f"{dst.stem}.{n}{dst.suffix}")
            if not candidate.exists():
                return candidate
        raise FileExistsError(f"cannot find a free name next to {dst}")

    def _move_to_deleted(self, f) -> bool:
        if f.get("deleted"):
            return False
        src = Path(f["path"])
        try:
            rel = src.relative_to(self.root)
        except ValueError:
            raise ValueError(
                f"{src} is outside the server root {self.root}; "
                "start the server with -r pointing at the folder the dataset was scanned from"
            )
        dst = self._free_destination(self.deleted_dir / rel)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        f["deleted"] = True
        f["deleted_path"] = str(dst)
        return True

    def _move_from_deleted(self, f) -> bool:
        if not f.get("deleted"):
            return False
        if f.get("missing"):
            raise FileNotFoundError(
                "the file is at neither its original path nor in _deleted; "
                "it was removed outside the reviewer and cannot be restored")
        src = Path(f["deleted_path"])
        dst = self._free_destination(Path(f["path"]))
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        if str(dst) != f["path"]:
            # Something had already taken the original name, so the frame came back under
            # a different one. Record where it actually is: leaving the old path behind
            # would point every later read - and the next delete - at the other file.
            self.by_path.pop(f["path"], None)
            f["path"] = str(dst)
            self.by_path[f["path"]] = f
        f["deleted"] = False
        f.pop("deleted_path", None)
        return True

    def _apply(self, fids, move):
        """Move every requested frame under one lock acquisition and one state write, and
        report per-frame failures instead of abandoning the rest of the batch. Deleting a
        40-frame selection used to be 40 round trips, each rewriting the whole state file.

        Returns {fid: message} for the frames that could not be moved.
        """
        failed, changed = {}, False
        with self.lock:
            for fid in fids:
                try:
                    changed |= move(self.files[fid])
                except Exception as e:
                    failed[fid] = str(e) or repr(e)
            if changed:
                self.save_state()
        return failed

    def delete_many(self, fids):
        return self._apply(fids, self._move_to_deleted)

    def restore_many(self, fids):
        return self._apply(fids, self._move_from_deleted)

    def build_units(self):
        """One unit per (folder, exposure length), in folder order.

        A folder is one panel/session/filter but can hold several exposure lengths, and
        every comparison the page makes - a matrix cell, a summary line, a sigma clip, a
        thumbnail grid - is only meaningful within one length. So this, not the folder,
        is the grain the client aggregates on.
        """
        units = []
        for folder in self.folders:
            by_exposure = {}
            for fid in folder["file_ids"]:
                by_exposure.setdefault(self.files[fid]["exp_key"], []).append(fid)
            for exposure in sorted(by_exposure, key=stats.exposure_sort_key):
                for fid in by_exposure[exposure]:
                    self.files[fid]["unit_id"] = len(units)
                units.append({
                    "id": len(units),
                    "folder_id": folder["id"],
                    "panel": folder["panel"],
                    "session": folder["session"],
                    "filter": folder["filter"],
                    "exposure": exposure,
                    "file_ids": by_exposure[exposure],
                })
        return units

    def matrix_rows(self):
        """The (filter, exposure) pairs that make up the matrix rows, filter order first."""
        pairs = {(u["filter"], u["exposure"]) for u in self.units if u["filter"]}
        return sorted(pairs, key=lambda fe: (stats.filter_sort_key(fe[0]),
                                             stats.exposure_sort_key(fe[1])))

    def grouped_folders(self):
        return stats.group_by_panel_filter(self.folders, lambda fe: fe["folder"])

    def group_file_ids(self, panel: str, filt: str, exposure=None):
        """All frame ids for one panel+filter, across every session's folder.

        `exposure` narrows it to one exposure length (a stats.exposure_key); None keeps
        every length, which is what a panel+filter shot at a single length amounts to.
        """
        return [fid for folder in self.folders
                if folder["panel"] == panel and folder["filter"] == filt
                for fid in folder["file_ids"]
                if exposure is None
                or stats.exposure_key(self.files[fid].get("exposure")) == exposure]

    def group_exposures(self, panel: str, filt: str):
        """The distinct exposure lengths one panel+filter was shot at, shortest first."""
        return sorted({stats.exposure_key(self.files[fid].get("exposure"))
                       for fid in self.group_file_ids(panel, filt)},
                      key=stats.exposure_sort_key)


class LRU:
    """Bounded LRU with single-flight: concurrent misses on one key compute once and the
    losing threads wait for that result.

    ThreadingHTTPServer runs every request on its own thread and a browser opens several
    connections at a time, so opening a thumbnail grid used to fire the same multi-second
    group-stretch fit from six threads at once - each of them reading the same 35 FITS
    files off disk.
    """

    def __init__(self, maxsize):
        self.maxsize = maxsize
        self.lock = threading.Lock()
        self.entries = OrderedDict()
        self.pending = {}

    _MISS = object()   # a cached value may legitimately be None (see below), so `is None`
    #                    cannot stand in for "not cached": locate_trail returns None for a
    #                    flagged frame it fails to confirm, and testing the value itself
    #                    re-ran that ~0.7 s measurement on every request for such a frame.

    def get(self, key, produce):
        while True:
            with self.lock:
                hit = self.entries.get(key, self._MISS)
                if hit is not self._MISS:
                    self.entries.move_to_end(key)
                    return hit
                waiter = self.pending.get(key)
                if waiter is None:
                    waiter = self.pending[key] = threading.Event()
                    break
            # Someone else is already computing this. Wait, then loop: either their value
            # is cached now, or they failed and this thread takes over as the leader.
            waiter.wait()

        try:
            value = produce()
        except BaseException:
            with self.lock:
                self.pending.pop(key, None)
            waiter.set()
            raise
        with self.lock:
            self.entries[key] = value
            self.entries.move_to_end(key)
            while len(self.entries) > self.maxsize:
                self.entries.popitem(last=False)
            self.pending.pop(key, None)
        waiter.set()
        return value


previews = LRU(PREVIEW_CACHE_SIZE)
histograms = LRU(HISTOGRAM_CACHE_SIZE)
thumbs = LRU(THUMB_CACHE_SIZE)
group_stretches = LRU(GROUP_STRETCH_CACHE_SIZE)
trails = LRU(TRAIL_CACHE_SIZE)
crops = LRU(CROP_CACHE_SIZE)
_decimated_preview = LRU(DECIMATED_PREVIEW_CACHE_SIZE)
_decimated_thumb = LRU(DECIMATED_THUMB_CACHE_SIZE)
_full_frames = LRU(FULL_FRAME_CACHE_SIZE)
_frame_shapes = LRU(64)


def file_key(path: Path):
    """Identity of a file's current contents, and the prefix of every cache key here.
    A delete/restore moves the frame and a rescan can replace it, so path+mtime+size is
    what actually has to invalidate a cached render."""
    st = path.stat()
    return (str(path), st.st_mtime_ns, st.st_size)


def _read_decimated(path: Path, max_size: int) -> np.ndarray:
    # Decimate before the float conversion: the discarded rows never become floats, so
    # the full-frame float64 copy the preview used to allocate is avoided entirely.
    # memmap is left to astropy: forcing it on raises for the BZERO/BSCALE-scaled
    # unsigned frames these cameras write.
    with fits.open(str(path)) as hdul:
        hdu = stats.first_image_hdu(hdul)
        shape = hdu.shape
        if len(shape) != 2:
            raise ValueError(f"expected a 2D image, got shape {shape}")
        h, w = shape
        scale = max(1, int(-(-max(h, w) // max_size)))
        return np.asarray(hdu.data[::scale, ::scale], dtype=np.float32)


def load_decimated(path: Path, max_size: int = PREVIEW_MAX_SIZE, key=None) -> np.ndarray:
    """Decimated read of one frame, cached.

    The preview PNG, its histogram, a grid thumbnail and the pooled stretch that
    thumbnail is scaled against all want the same array. Reading the FITS once per
    consumer was the largest single cost of opening a grid: every frame in the group was
    read once to fit the shared stretch and then a second time to render its thumbnail.

    The returned array is shared between callers - treat it as read-only.
    """
    cache = _decimated_thumb if max_size <= THUMB_MAX_SIZE else _decimated_preview
    if key is None:
        key = file_key(path)
    return cache.get(key + (max_size,), lambda: _read_decimated(path, max_size))


def _read_full(path: Path) -> np.ndarray:
    """The undecimated frame, in its stored dtype. Kept native (usually uint16) rather than
    converted: only the small crop actually being shown needs to become floats."""
    with fits.open(str(path)) as hdul:
        data = stats.first_image_hdu(hdul).data
        if data.ndim != 2:
            raise ValueError(f"expected a 2D image, got shape {data.shape}")
        return np.asarray(data)


def load_full(path: Path, key=None) -> np.ndarray:
    """Undecimated frame, cached. Panning around at 1:1 issues a request per drag, and
    re-reading ~18 MB off the array for each one is the whole cost of the interaction.

    The returned array is shared between callers - treat it as read-only.
    """
    if key is None:
        key = file_key(path)
    return _full_frames.get(key, lambda: _read_full(path))


def frame_shape(path: Path, key=None):
    """(height, width) of a frame, read from the header alone - no pixel data. The client
    needs it to clamp panning before it has fetched any crop."""
    if key is None:
        key = file_key(path)

    def read():
        with fits.open(str(path)) as hdul:
            # Typed loosely for the same reason as stats.first_image_hdu: iterating an
            # HDUList yields HDUs of several classes and a checker otherwise reads the
            # loop variable as the HDUList itself.
            hdus: Any = hdul
            for hdu in hdus:
                if int(hdu.header.get("NAXIS", 0)) >= 2:
                    return int(hdu.header["NAXIS2"]), int(hdu.header["NAXIS1"])
        raise ValueError("file contains no image data")

    return _frame_shapes.get(key, read)


def _stretch_to_png(small: np.ndarray, vmin: float, vmax: float, stretch) -> bytes:
    if vmax <= vmin:
        vmax = vmin + 1.0
    # Not in-place: `small` is the shared cached array, not a private copy.
    scaled = (small - vmin) / (vmax - vmin)
    np.clip(scaled, 0.0, 1.0, out=scaled)
    img8 = (np.clip(stretch(scaled), 0.0, 1.0) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img8, mode="L").save(buf, format="PNG")
    return buf.getvalue()


# The white point for "linear". A plain min/max put it on the brightest pixel in the
# frame - a saturated star or a hot pixel at ~65500 - which mapped a 2900 ADU sky to 0:
# every linear preview and every linear thumbnail rendered essentially black. Clipping the
# top fraction of a percent puts the white point just above the star field instead, while
# the black point stays at the true floor so the ramp is still linear from zero signal.
LINEAR_WHITE_PERCENTILE = 99.5


MIDTONE_MIN, MIDTONE_MAX = 0.001, 0.999


class MidtoneStretch:
    """The black / midtone / white transfer function used across astro imaging.

    `m` is the input level - already scaled to [0,1] between the black and white points -
    that comes out as mid grey, so it is exactly the quantity a midtone slider holds and
    the position of the midtone handle between the two other handles. m = 0.5 is the
    identity, which is why "linear" is this function rather than a separate path.
    """

    def __init__(self, m):
        self.m = min(max(float(m), MIDTONE_MIN), MIDTONE_MAX)

    def __call__(self, values, clip=True, out=None):
        m = self.m
        result = np.array(values, copy=True) if out is None else out
        if out is not None and out is not values:
            np.copyto(out, values)
        if m == 0.5:
            return result
        # (2m-1)x - m is negative for every x in [0,1] when m is in (0,1), so no guard
        # against a zero denominator is needed - only against m reaching 0 or 1, which
        # the constructor clamps.
        denom = (2.0 * m - 1.0) * result - m
        np.multiply(result, m - 1.0, out=result)
        np.divide(result, denom, out=result)
        return result


# Every stretch here is one black point, one white point and one midtone, fitted to the
# frame's own pixels - the standard screen-transfer autostretch.
#
# It replaced a narrow zscale window because that window is the wrong tool for inspecting
# frames: fitted to the sky it put the white point a couple of hundred ADU above the
# background, so on a real sub between a thousand and fifteen hundred pixels per frame -
# every star core - clipped flat to 255 with no shape left in them. Anchoring white at the
# sensor's full scale instead clips nothing, and the midtone does the lifting. Measured
# over one frame of each filter, cores at pure white went from 1022-1433 to zero, and the
# background landed on the same grey for every filter instead of drifting across 152-176.
FULL_SCALE_ADU = 65535.0        # 16-bit ceiling: white sits here so no star ever clips
STF_SHADOW_CLIP = 2.8           # black point, in normalised MADs below the median
STF_TARGET_BACKGROUND = {"auto": 0.25, "super": 0.40}
STRETCH_MODES = ("auto", "super", "linear")


def _midtone_for(x: float, y: float) -> float:
    """The midtone m whose transfer function maps input level x to output level y."""
    return x * (y - 1.0) / (2.0 * x * y - x - y)


def levels_from_sample(sample: np.ndarray, stretch_mode: str):
    """(black, white, midtone) for one mode, fitted to a sample of a frame's pixels.

    The single definition of every stretch in this server: the preview, the 1:1 crops and
    a thumbnail group's shared limits all come through here, so they cannot drift apart.
    """
    if stretch_mode == "linear":
        black, white = AsymmetricPercentileInterval(0.0, LINEAR_WHITE_PERCENTILE).get_limits(sample)
        black, white = float(black), float(white)
        return black, (white if white > black else black + 1.0), 0.5

    median = float(np.median(sample))
    floor = float(np.min(sample))
    madn = 1.4826 * float(np.median(np.abs(sample - median)))
    # Clip the shadows to just under the noise, not to the sensor's offset pedestal: the
    # sky sits at a different level per filter and per night, so a fixed pedestal leaves
    # most of the range empty on the brighter ones.
    black = max(floor, median - STF_SHADOW_CLIP * madn) if madn > 0 else floor
    white = FULL_SCALE_ADU if FULL_SCALE_ADU > black else black + 1.0
    x = (median - black) / (white - black)
    if not 0.0 < x < 1.0:
        return black, white, 0.5   # degenerate sample; a flat transfer is the honest one
    mid = _midtone_for(x, STF_TARGET_BACKGROUND[stretch_mode])
    return black, white, min(max(mid, MIDTONE_MIN), MIDTONE_MAX)


def preview_levels(path: Path, stretch_mode: str, key=None) -> dict:
    """Where a preset puts the three handles for this frame, in ADU (midtone in [0,1])."""
    black, white, mid = levels_from_sample(
        load_decimated(path, PREVIEW_MAX_SIZE, key), stretch_mode)
    return {"black": black, "white": white, "mid": mid}


def render_levels_png(small: np.ndarray, black: float, white: float, mid: float) -> bytes:
    return _stretch_to_png(small, black, white, MidtoneStretch(mid))


def render_preview_png(path: Path, stretch_mode: str, key=None) -> bytes:
    small = load_decimated(path, PREVIEW_MAX_SIZE, key)
    return render_levels_png(small, *levels_from_sample(small, stretch_mode))


def render_crop_png(path: Path, x: int, y: int, w: int, h: int,
                    levels, key=None) -> bytes:
    """One 1:1 region of the undecimated frame, stretched exactly like its preview."""
    full = load_full(path, key)
    fh, fw = full.shape
    x = max(0, min(int(x), max(0, fw - 1)))
    y = max(0, min(int(y), max(0, fh - 1)))
    w = max(1, min(int(w), CROP_MAX_SIZE, fw - x))
    h = max(1, min(int(h), CROP_MAX_SIZE, fh - y))
    crop = np.asarray(full[y:y + h, x:x + w], dtype=np.float32)
    return render_levels_png(crop, *levels)


def render_thumb_png(path: Path, black: float, white: float, mid: float, key=None) -> bytes:
    """Like render_preview_png, but stretched against externally-supplied limits (shared
    across a whole panel/filter group) instead of the frame's own min/max, so a grid of
    these can be visually compared frame-to-frame."""
    return render_levels_png(load_decimated(path, THUMB_MAX_SIZE, key), black, white, mid)


HIST_HIGH_PERCENTILE = 99.9   # see render_histogram
HIST_HEADROOM = 0.15          # extra span past the plotted range, for dragging white up


def render_histogram(path: Path, bins: int = 200, key=None) -> dict:
    """Pixel-value histogram of the (decimated) frame, plus where each preset puts its
    black, midtone and white points.

    Plotted over the frame's floor up to a high percentile rather than up to its maximum.
    A single saturated star sits near 65535 while the sky and everything shaped like it
    sits within a couple of hundred ADU of the sensor's offset pedestal, so a full-range
    axis put 99%% of the pixels - the entire sky distribution, and with it any sign of the
    offset - inside the first bin of two hundred, leaving only the star tail visible.
    """
    small = load_decimated(path, PREVIEW_MAX_SIZE, key)
    full_min, full_max = float(small.min()), float(small.max())
    presets = {mode: preview_levels(path, mode, key) for mode in STRETCH_MODES}

    # The plotted range covers the data and the handles that need fine control - the black
    # point and the midtone. Deliberately NOT the white point: the autostretch parks it at
    # the sensor's full scale so nothing clips, and stretching the axis out to 65535 would
    # put the entire sky distribution back inside one bin, which is the thing this axis
    # exists to avoid. A white point out beyond the plot is drawn pinned at the edge.
    high = float(np.percentile(small, HIST_HIGH_PERCENTILE))
    midtones = [p["black"] + p["mid"] * (p["white"] - p["black"]) for p in presets.values()]
    lo = min([full_min] + [p["black"] for p in presets.values()])
    hi = max([high] + [m for m in midtones if m <= high * 2])
    if hi <= lo:
        hi = lo + 1.0
    hi += HIST_HEADROOM * (hi - lo)

    counts, _edges = np.histogram(small, bins=bins, range=(lo, hi))
    return {"counts": counts.tolist(), "min": lo, "max": hi,
            "full_min": full_min, "full_max": full_max,
            "presets": presets}


def _compute_group_stretch(live, stretch_mode: str):
    if not live:
        return 0.0, 1.0, 0.5
    # Pool pixels across frames and fit ONE interval to that pooled population, rather
    # than averaging each frame's own zscale limits: sessions can sit on slightly
    # different absolute backgrounds (sky glow, bias drift), and a median of per-frame
    # limits just snaps to whichever session has more frames, clipping every frame from
    # the other session to solid white/black.
    #
    # Only a sample of the group is pooled. The result is a two-number summary, and
    # reading all ~35 frames of a group before the first thumbnail could paint cost
    # seconds of cold disk I/O. The sample is spread evenly over the group so it still
    # spans every session rather than over-representing the first one.
    n = min(len(live), GROUP_STRETCH_SAMPLE_FRAMES)
    picks = np.unique(np.linspace(0, len(live) - 1, n).round().astype(int))

    samples = []
    for i in picks:
        key = live[i]
        try:
            small = load_decimated(Path(key[0]), THUMB_MAX_SIZE, key)
        except Exception:
            continue
        samples.append(small.ravel()[::4])  # subsample: pooling is for the stretch, not precision
    if not samples:
        return 0.0, 1.0, 0.5
    return levels_from_sample(np.concatenate(samples), stretch_mode)


def group_stretch_limits(ds: "Dataset", panel: str, filt: str, stretch_mode: str, exposure=None,
                         source: str = SOURCE_RAW):
    """The shared (black, white, midtone) used to render every thumbnail in one group,
    so frames line up for comparison instead of each auto-stretching to its own contrast.
    Fitted per exposure length, since that is what the group is split on - one interval
    pooled over 30 s and 45 s subs would sit between their two sky levels and clip both.
    Keyed on a fingerprint of the group's current (non-deleted) members, so a delete/
    restore or a rescan invalidates it rather than serving stale limits."""
    live = []
    for fid in ds.group_file_ids(panel, filt, exposure):
        if ds.files[fid].get("deleted"):
            continue
        # Only frames that really are in this source contribute to its shared stretch.
        if ds.effective_source(fid, source) != source:
            continue
        try:
            live.append(file_key(ds.current_path(fid, source)))
        except OSError:
            continue
    return group_stretches.get((panel, filt, exposure, stretch_mode, source, tuple(live)),
                               lambda: _compute_group_stretch(live, stretch_mode))


PAGE_CSS = stats.BASE_CSS + stats.MATRIX_CSS + """
html, body { height:100%; overflow:hidden; }
tbody tr { cursor:pointer; }
tbody tr.deleted { opacity:0.45; text-decoration: line-through; }
tbody tr.missing { opacity:0.3; font-style:italic; }
tbody tr.sigma-clipped { display:none; }
.app-shell { height:100vh; display:grid; grid-template-columns:320px minmax(0, 1fr); }
.filter-sidebar { position:sticky; top:0; height:100vh; overflow-y:auto; padding:24px 24px 32px; border-right:1px solid var(--border); background:var(--bg); }
.data-pane { min-width:0; height:100vh; overflow:auto; padding:24px 32px 64px; }
.data-pane > h2:first-child { margin-top:0; }
.filter-sidebar .sub { margin-bottom:26px; }
.clip-panel { display:flex; align-items:stretch; gap:10px; flex-direction:column; margin:0 0 22px; padding:14px 0; border-top:1px solid var(--border); border-bottom:1px solid var(--border); }
.clip-panel label { color:var(--muted); font-size:12.5px; }
.clip-panel select, .clip-panel input[type=range] { accent-color:var(--accent); width:100%; }
.clip-panel select { background:var(--panel); color:var(--text); border:1px solid var(--border); border-radius:6px; padding:5px 8px; font:inherit; font-size:12.5px; }
.clip-value { color:var(--text); font-size:12.5px; min-width:54px; font-variant-numeric:tabular-nums; }
.metric-row { display:flex; flex-direction:column; gap:6px; margin:0; }

#modal-overlay { position:fixed; inset:0; background:rgba(0,0,0,0.6); display:none; align-items:center; justify-content:center; z-index:60; }
#modal-overlay.open { display:flex; }
#modal { background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:16px; width:min(94vw, 960px); height:min(92vh, 900px); display:flex; flex-direction:column; gap:10px; }
#modal-title { font-size:13px; color:var(--muted); word-break:break-all; }
#modal-headline { font-size:13px; color:var(--text); display:flex; gap:16px; flex-wrap:wrap; }
#modal-headline b { color:var(--accent); }
#modal-img-wrap { position:relative; flex:1 1 auto; min-height:260px; display:flex; align-items:center; justify-content:center; background:#000; border-radius:6px; overflow:hidden; }
#modal-img-wrap img { width:100%; height:100%; object-fit:contain; display:block; }
#modal-img-wrap.loading img { opacity:0.42; }
#modal-histogram { width:100%; height:96px; display:block; border-radius:6px; background:var(--head); cursor:ew-resize; touch-action:none; }
#modal-levels { display:flex; align-items:center; gap:14px; font-size:11.5px; color:var(--muted); font-variant-numeric:tabular-nums; }
#modal-levels b { color:var(--text); font-weight:600; }
#modal-levels .lv-black b { color:#8a93a6; }
#modal-levels .lv-mid b { color:#d9a441; }
#modal-levels .lv-white b { color:var(--accent); }
#modal-levels button { font:inherit; font-size:11px; padding:2px 9px; border-radius:5px; border:1px solid var(--border); background:var(--head); color:var(--muted); cursor:pointer; }
#modal-levels button:hover:not(:disabled) { border-color:var(--accent); color:var(--text); }
#modal-levels button:disabled { opacity:0.35; cursor:default; }
/* Fixed so the row never changes height and resizes the preview above it. */
#modal-levels { min-height:22px; }
#modal-stats { font-size:12px; color:var(--muted); display:flex; flex-wrap:wrap; gap:14px; }
#modal-stats b { color:var(--text); }
.modal-actions { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
button.btn { font:inherit; font-size:12.5px; padding:6px 12px; border-radius:6px; border:1px solid var(--border); background:var(--head); color:var(--text); cursor:pointer; }
button.btn:hover { border-color:var(--accent); }
button.btn.active { border-color:var(--accent); color:var(--accent); }
button.btn.danger { border-color: var(--bad); color:var(--bad); }
button.btn.danger:hover { background: color-mix(in srgb, var(--bad) 14%, transparent); }
.spacer { flex:1; }
#loading { color:var(--muted); font-size:12px; }
.trail-overlay { position:absolute; inset:0; width:100%; height:100%; pointer-events:none; z-index:1; }
.trail-box { fill:none; stroke:#ff5a5a; stroke-width:1.5; vector-effect:non-scaling-stroke; }
/* Pinned to the opposite corner from the zoom readout, which shares this class and sat
   directly on top of it at 1:1. */
.trail-status { color:#ff8a8a; inset:12px auto auto 12px; }
.preview-status { position:absolute; inset:auto auto 12px 12px; z-index:2; padding:4px 8px; color:var(--text); background:rgba(0,0,0,0.62); border-radius:6px; font-size:12px; pointer-events:none; }
#modal-img-wrap.zoomable img { cursor:zoom-in; }
#modal-img-wrap.zoomed { cursor:grab; }
#modal-img-wrap.zoomed img { position:absolute; width:auto; height:auto; max-width:none; max-height:none; object-fit:none; cursor:grab; }
#modal-img-wrap.panning, #modal-img-wrap.panning img { cursor:grabbing; }
.row-select { cursor:pointer; }
td:has(> .row-select) { width:1%; text-align:center; }
#bulk-delete-btn:disabled, #grid-delete-btn:disabled { opacity:0.4; cursor:default; }
#bulk-delete-btn:disabled:hover, #grid-delete-btn:disabled:hover { border-color:var(--border); background:none; }
.grid-btn { font-size:11px; padding:2px 8px; margin-left:8px; font-weight:400; vertical-align:middle; }
h3.flash { animation: flash-section 1.4s ease-out; border-radius:4px; }
@keyframes flash-section {
  from { background: color-mix(in srgb, var(--accent) 30%, transparent); }
  to { background: transparent; }
}
#grid-overlay { position:fixed; inset:0; background:rgba(0,0,0,0.6); display:none; align-items:center; justify-content:center; z-index:55; }
#grid-overlay.open { display:flex; }
#grid-modal { background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:16px; width:min(96vw, 1400px); height:min(92vh, 1100px); display:flex; flex-direction:column; gap:10px; }
#grid-header { display:flex; align-items:center; gap:14px; flex-wrap:wrap; }
#grid-title { font-size:14px; font-weight:600; color:var(--text); }
#grid-count { font-size:12px; color:var(--muted); }
#grid-wrap { flex:1; min-height:0; overflow-y:auto; display:grid; grid-template-columns:repeat(auto-fill, minmax(170px, 1fr)); gap:12px; align-content:start; padding-right:4px; }
.grid-item { display:flex; flex-direction:column; gap:4px; }
.grid-item .thumb-wrap { position:relative; }
.grid-item img { width:100%; aspect-ratio:1/1; object-fit:contain; background:#000; border-radius:6px; cursor:pointer; box-sizing:border-box; }
.grid-cap { display:flex; flex-wrap:wrap; align-items:baseline; gap:2px 6px; font-size:11.5px; color:var(--muted); line-height:1.3; font-variant-numeric:tabular-nums; }
.grid-q { font-size:12px; }
.grid-metrics { white-space:nowrap; }
.grid-sess { margin-left:auto; font-size:10px; opacity:0.8; white-space:nowrap; }
.trail-badge { position:absolute; top:4px; right:4px; font-size:14px; line-height:1; background:rgba(0,0,0,0.6); border-radius:50%; width:22px; height:22px; display:flex; align-items:center; justify-content:center; pointer-events:none; }
.grid-select { position:absolute; top:4px; left:4px; width:18px; height:18px; z-index:2; cursor:pointer; }
.grid-item.clipped img { opacity:0.28; }
.grid-item.clipped .grid-cap { opacity:0.5; }
.clip-badge { position:absolute; inset:auto 4px 4px auto; font-size:9.5px; letter-spacing:0.04em; text-transform:uppercase; background:rgba(0,0,0,0.66); color:#d9a441; border-radius:4px; padding:1px 5px; pointer-events:none; }
@media (max-width: 820px) {
  html, body { height:auto; overflow:auto; }
  .app-shell { height:auto; display:block; }
  .filter-sidebar { position:static; height:auto; border-right:0; border-bottom:1px solid var(--border); }
  .data-pane { height:auto; overflow:visible; padding:24px 20px 48px; }
  #modal { height:min(92vh, 760px); }
  #modal-img-wrap { min-height:220px; }
}
"""

PAGE_JS = """
const FILES = __FILES_JSON__;
const FOLDERS = __FOLDERS_JSON__;
// One entry per (folder, exposure length). Everything derived below aggregates on these
// rather than on folders: a folder holding 30 s and 45 s subs is two incomparable sets.
const UNITS = __UNITS_JSON__;
const GROUPS = __GROUPS_JSON__;
const MATRIX_METRICS = __MATRIX_METRICS_JSON__;
const FILTER_COLORS = __FILTER_COLORS_JSON__;
const DEFAULT_FILTER_COLOR = __DEFAULT_FILTER_COLOR_JSON__;
const DASH = '\\u2014';

function unitKey(panel, session, filt, exposure) {
  return [panel, session, filt, exposure].join('\\u0000');
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
      const box = BOX_BY_ID.get(f.id);
      if (box) {
        // A frame deleted from the preview must drop out of the bulk selection too,
        // or "Delete selected" keeps counting frames that are already gone.
        box.disabled = deleted;
        // Deferred: renderRows() refreshes the buttons once when it finishes.
        if (deleted && box.checked) setSelected(f.id, false, true);
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
  if (value) value.textContent = clipSigma <= 0 ? 'Off' : clipSigma.toFixed(2).replace(/\\.00$/, '.0') + '\\u03c3';
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
      badge.title = 'Excluded by the sigma-clipping slider; not counted in this set\\u2019s statistics';
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

function openModal(id) {
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
    return 'tracking-like, r=' + f.elong_uniform_r.toFixed(2) + ' @ ' + fmtNum(f.elong_uniform_angle_deg, 0) + '\\u00b0';
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
      (f.quality_reasons && f.quality_reasons.length ? ' \\u2014 ' + f.quality_reasons.join(', ') : '')
    : DASH;
  box.replaceChildren(
    statSpan('Quality', qualityLabel),
    statSpan('ADU', fmtNum(f.adu_min, 0) + ' \\u2013 ' + fmtNum(f.adu_max, 0) + ' (mean ' + fmtNum(f.adu_mean, 1) + ')'),
    statSpan('Signal', fmtNum(f.signal_adu, 1) + ' ADU'),
    statSpan('Signal rate', fmtNum(f.signal_adu_per_sec, 2) + ' ADU/s'),
    statSpan('HFR', fmtNum(f.hfr_avg, 2)),
    statSpan('FWHM', fmtNum(f.fwhm_avg, 2) + ' px'),
    statSpan('Stars', String(f.num_stars ?? 0)),
    statSpan('Roundness', fmtPct(f.roundness_avg, 1)),
    statSpan('Elongation dir', elongationSummary(f)),
  );

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
    item('lv-white', 'White', fmtAdu(lv.white) + (offPlot ? ' \\u00b7 full scale' : '')),
  ];
  const spacer = document.createElement('span');
  spacer.className = 'spacer';
  nodes.push(spacer);
  const note = document.createElement('span');
  note.textContent = levelsAuto ? 'following ' + curStretch : 'adjusted \\u2014 drag the handles';
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
    textContent: requestedZoom ? 'Loading 1:1\\u2026' : 'Rendering preview\\u2026'});
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
      badge.textContent = '\\u{1F6F0}\\uFE0F';
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
    status.textContent = '\\u{1F6F0}\\uFE0F trail ' + trailAngleDeg(geometry).toFixed(0) +
      '\\u00b0 \\u00b7 ' + geometry.width.toFixed(1) + ' px wide';
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

async function applyMove(action, ids) {
  // One request for the whole batch: the server moves them under a single lock and
  // rewrites the recovery state file once, rather than once per frame.
  const res = await fetch('/api/' + action, {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ids})
  });
  const detail = await res.json().catch(() => ({}));
  if (!res.ok) {
    alert('Action failed: ' + (detail.error || res.status));
    return null;
  }
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
    alert(failedIds.length + ' frame(s) could not be moved:\\n' +
          failedIds.map(id => FILES[id].name + ': ' + failed[id]).join('\\n'));
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

async function toggleDelete() {
  const f = FILES[curId];
  const action = f.deleted ? 'restore' : 'delete';
  if (action === 'delete' && !confirm('Move this frame to _deleted?\\n' + f.name + '\\n\\nThis is reversible (Restore), the file is moved not erased.')) {
    return;
  }
  const delBtn = document.getElementById('delete-btn');
  delBtn.disabled = true;
  try {
    if (await applyMove(action, [f.id]) !== null) renderModal();
  } finally {
    delBtn.disabled = false;
  }
}

function nextFile(delta) {
  if (curId === null || !FILES.length) return;
  const n = FILES.length;
  // Skip over frames the current view hides, so arrowing follows what is on screen.
  for (let step = 1; step <= n; step++) {
    const i = ((curId + delta * step) % n + n) % n;
    if (!FILES[i].sigmaClipped || i === curId) { openModal(i); return; }
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
  return g.panel + ' \\u00b7 Filter ' + g.filter + ' \\u00b7 ' + exposureSuffix(g.exposure);
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
    panel + ' \\u2014 Filter ' + filt + ' \\u00b7 ' + exposureSuffix(groupExposure) + ' (all sessions)';
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
    (clipped ? ' \\u00b7 ' + clipped + ' clipped' : '') + ' \\u00b7 same stretch for all' +
    (gridGroupIndex >= 0 ? ' \\u00b7 group ' + (gridGroupIndex + 1) + '/' + GROUPS.length : '');
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
    img.addEventListener('click', () => openModal(id));
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
      badge.textContent = '\\u{1F6F0}\\uFE0F';
      thumbWrap.append(badge);
    }
    if (f.sigmaClipped) {
      const badge = document.createElement('span');
      badge.className = 'clip-badge';
      badge.title = 'Excluded by the sigma-clipping slider; not counted in this set\\u2019s statistics';
      badge.textContent = 'clipped';
      thumbWrap.append(badge);
    }
    // The filename is 90 characters of camera settings and repeats across the whole grid,
    // so it moves to the hover title and the caption carries what you actually compare
    // thumbnails on.
    const cap = document.createElement('div');
    cap.className = 'grid-cap';
    cap.title = f.name + (f.quality_reasons && f.quality_reasons.length
      ? '\\n\\n' + f.quality_reasons.join('; ') : '');
    const q = document.createElement('span');
    q.className = 'grid-q q-' + (f.quality_verdict || '');
    q.textContent = numeric(f.quality_score) ? 'Q' + Math.round(f.quality_score) : 'Q' + DASH;
    const rest = document.createElement('span');
    rest.className = 'grid-metrics';
    rest.textContent = 'A' + fmtNum(f.adu_mean, 0) + ' \\u00b7 \\u2605' + (f.num_stars ?? DASH);
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
    else return;
    e.preventDefault();
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
  if (row) openModal(Number(row.dataset.id));
});

const selectedIds = new Set();

function updateDeleteButtons() {
  const btn = document.getElementById('bulk-delete-btn');
  btn.disabled = selectedIds.size === 0;
  btn.textContent = 'Delete selected (' + selectedIds.size + ')';
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
  if (!confirm('Move ' + ids.length + ' frame(s) selected in this view to _deleted?\\n\\n' +
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
  const ids = Array.from(selectedIds);
  if (!ids.length) return;
  if (!confirm('Move ' + ids.length + ' selected frame(s) to _deleted?\\n\\nThis is reversible (Restore in the preview), files are moved not erased.')) {
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
"""


def stats_anchor_id(panel: str, filt: str, exposure: str = "") -> str:
    """Anchor of one statistics section. Exposure is part of it: a filter shot at two
    lengths has a section per length, and "Statistics" has to land on the right one."""
    slug = re.sub(r"[^a-z0-9]+", "-", f"{panel}-{filt}-{exposure or 'noexp'}".lower()).strip("-")
    return f"stats-{slug}"


def render_table_head_with_select(parts, with_session=False):
    parts.append('<div class="wrap"><table class="sortable-table"><thead><tr>')
    parts.append('<th></th>')
    if with_session:
        parts.append('<th class="sortable">Session</th>')
    parts.extend(stats._table_header_cell(h) for h in stats.TABLE_HEADERS)
    parts.append("</tr></thead><tbody>")


def render_folder_rows(folder_entry, ds: Dataset, unit_ids=()):
    file_ids = folder_entry["file_ids"]
    parts = [f'<div class="meta" data-unit-summary="{",".join(str(i) for i in unit_ids)}">'
             f'{len(file_ids)}/{len(file_ids)} included</div>']
    render_table_head_with_select(parts)

    for fid in file_ids:
        r = ds.files[fid]
        deleted = bool(r.get("deleted"))
        row_cls = "deleted missing" if r.get("missing") else ("deleted" if deleted else "")
        title = ' title="file no longer exists - removed outside the reviewer"' if r.get("missing") else ""
        disabled = " disabled" if deleted else ""
        parts.append(f'<tr data-id="{fid}" class="{row_cls}"{title}>')
        parts.append(f'<td><input type="checkbox" class="row-select" data-id="{fid}"{disabled}'
                      f' onclick="event.stopPropagation()"></td>')
        parts.extend(stats.render_row_cells(r))
        parts.append("</tr>")

    parts.append("</tbody></table></div>")
    return parts


def render_sessions_rows(sess_list, ds: Dataset):
    """One table for a panel+filter+exposure combining every session's frames, with a
    leading Session column so rows stay comparable across sessions.

    sess_list is [(session, unit)] - already one exposure length, so the summary line
    above the table counts and averages only frames that compare to each other.
    """
    unit_ids = [unit["id"] for _, unit in sess_list]
    total = sum(len(unit["file_ids"]) for _, unit in sess_list)
    parts = [f'<div class="meta" data-unit-summary="{",".join(str(i) for i in unit_ids)}">'
             f'{total}/{total} included across {len(sess_list)} session(s)</div>']
    render_table_head_with_select(parts, with_session=True)

    for session, folder_entry in sess_list:
        for fid in folder_entry["file_ids"]:
            r = ds.files[fid]
            deleted = bool(r.get("deleted"))
            row_cls = "deleted missing" if r.get("missing") else ("deleted" if deleted else "")
            title = ' title="file no longer exists - removed outside the reviewer"' if r.get("missing") else ""
            disabled = " disabled" if deleted else ""
            parts.append(f'<tr data-id="{fid}" class="{row_cls}"{title}>')
            parts.append(f'<td><input type="checkbox" class="row-select" data-id="{fid}"{disabled}'
                          f' onclick="event.stopPropagation()"></td>')
            parts.append(f'<td>{html.escape(session)}</td>')
            parts.extend(stats.render_row_cells(r))
            parts.append("</tr>")

    parts.append("</tbody></table></div>")
    return parts


CLIP_OPTIONS = [
    ("quality_score", "Quality score"),
    ("signal_adu_per_sec", "Signal ADU/s"),
    ("adu_mean", "ADU mean"),
    ("signal_adu", "Signal ADU"),
    ("hfr_avg", "HFR"),
    ("fwhm_avg", "FWHM"),
    ("elong_uniform_r", "Elongation: same direction"),
    ("elong_radial_r", "Elongation: radial (coma-like)"),
]

# Only the fields the client actually reads; the raw rows also carry path/date_obs/adu_dev.
CLIENT_FILE_FIELDS = ("id", "name", "filter", "folder_id", "exposure", "adu_min", "adu_max",
                      "adu_mean", "signal_adu", "signal_adu_per_sec", "hfr_avg", "fwhm_avg",
                      "num_stars", "roundness_avg", "deleted", "has_trail",
                      "elong_n", "elong_uniform_r", "elong_uniform_angle_deg",
                      "elong_radial_r", "elong_radial_diff_deg",
                      "quality_score", "quality_reasons", "quality_verdict",
                      "exp_key", "unit_id", "missing", "has_calibrated", "calibration")


def build_page(ds: Dataset) -> str:
    sessions = sorted({f["session"] for f in ds.folders if f["session"]}, key=stats.natural_num)
    panels = sorted({f["panel"] for f in ds.folders if f["panel"]}, key=stats.natural_num)
    matrix_rows = ds.matrix_rows()

    parts = [f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>FITS LIGHT frame review</title>
<style>{PAGE_CSS}</style></head><body>
<div class="app-shell">
<aside class="filter-sidebar">
<h1>FITS LIGHT frame review</h1>
<div class="sub">Root: {html.escape(ds.root.name)} &nbsp;&middot;&nbsp; Generated {html.escape(time.strftime('%Y-%m-%d %H:%M'))} &nbsp;&middot;&nbsp; click a row to preview / delete, or check rows to bulk-delete</div>
<div class="sub"><button class="btn danger" id="bulk-delete-btn" onclick="bulkDeleteSelected()" disabled>Delete selected (0)</button></div>
"""]

    parts.append('<div class="clip-panel">')
    parts.append('<label for="clip-sigma">Sigma clipping</label>')
    parts.append('<input id="clip-sigma" type="range" min="0" max="100" step="0.1" value="0" list="clip-sigma-marks" oninput="setClipSigma(this.value)">')
    parts.append('<datalist id="clip-sigma-marks"><option value="0" label="Off"></option><option value="1" label="5"></option><option value="21.2" label="4"></option><option value="41.4" label="3"></option><option value="61.6" label="2"></option><option value="81.8" label="1"></option><option value="100" label="0.1"></option></datalist>')
    parts.append('<span class="clip-value" id="clip-value">Off</span>')
    parts.append('<label for="clip-metric">metric</label>')
    parts.append('<select id="clip-metric" onchange="setClipMetric(this.value)">')
    for value, label in CLIP_OPTIONS:
        selected = " selected" if value == "signal_adu_per_sec" else ""
        parts.append(f'<option value="{html.escape(value, quote=True)}"{selected}>{html.escape(label)}</option>')
    parts.append('</select>')
    parts.append('<span class="meta" id="clip-total">0 clipped</span>')
    parts.append('</div>')

    stats.render_metric_buttons(parts)
    parts.append('</aside>')
    parts.append('<main class="data-pane">')
    parts.append('<h2>Panel &times; Session comparison</h2>')

    def grid_button(panel, filt, exposure, stop_propagation):
        """A matrix row is now exactly one thumbnail group, so it needs exactly one button."""
        stop = "event.stopPropagation(); " if stop_propagation else ""
        # JSON-encode for the JS string, then HTML-escape for the attribute. HTML-escaping
        # alone is undone by the parser before the script runs, so a folder named
        # "Bode's Galaxy" closed the quoted argument early and the button did nothing.
        args = ", ".join(json.dumps(v) for v in (panel, filt, exposure))
        return (f'<button class="btn grid-btn" onclick="{stop}openGroupGrid('
                f'{html.escape(args, quote=True)})">All thumbnails</button>')

    def row_extra(panel, filt, exposure):
        return (grid_button(panel, filt, exposure, stop_propagation=True)
                + f'<button class="btn grid-btn" onclick="event.stopPropagation(); '
                  f'goToStats(\'{stats_anchor_id(panel, filt, exposure)}\')">Statistics</button>')

    stats.render_matrix_skeleton(parts, panels, sessions, matrix_rows, row_extra=row_extra)

    units_by_folder = {}
    for unit in ds.units:
        units_by_folder.setdefault(unit["folder_id"], []).append(unit)

    grouped, leftovers = ds.grouped_folders()
    group_order = []
    for panel, filt_list in grouped:
        parts.append(f'<h2>{html.escape(panel)}</h2>')
        for filt, sess_list in filt_list:
            # Regroup the panel+filter's folders into one section per exposure length,
            # keeping session order within each.
            by_exposure = {}
            for session, folder_entry in sess_list:
                for unit in units_by_folder.get(folder_entry["id"], []):
                    by_exposure.setdefault(unit["exposure"], []).append((session, unit))
            for exposure in sorted(by_exposure, key=stats.exposure_sort_key):
                parts.append(
                    f'<h3 id="{stats_anchor_id(panel, filt, exposure)}">Filter {html.escape(filt)} '
                    f'<span class="row-exp">{html.escape(stats.exposure_label(exposure))}</span>'
                    + grid_button(panel, filt, exposure, stop_propagation=False) + '</h3>')
                parts.extend(render_sessions_rows(by_exposure[exposure], ds))
                # Same order as the sections, so the grid's Prev/Next walk the page.
                group_order.append({"panel": panel, "filter": filt, "exposure": exposure})

    for folder_entry in leftovers:
        parts.append(f'<h2>{html.escape(folder_entry["folder"])}</h2>')
        parts.extend(render_folder_rows(folder_entry, ds,
                                        [u["id"] for u in units_by_folder.get(folder_entry["id"], [])]))

    js = (stats.MATRIX_TOTALS_JS
          + PAGE_JS
          .replace("__FILES_JSON__", stats.json_for_script(
              [{k: f.get(k) for k in CLIENT_FILE_FIELDS} for f in ds.files]))
          .replace("__FOLDERS_JSON__", stats.json_for_script(ds.folders))
          .replace("__UNITS_JSON__", stats.json_for_script(ds.units))
          .replace("__GROUPS_JSON__", stats.json_for_script(group_order))
          .replace("__MATRIX_METRICS_JSON__", stats.json_for_script(stats.COMPARISON_METRICS))
          .replace("__FILTER_COLORS_JSON__", stats.json_for_script(stats.FILTER_COLORS))
          .replace("__DEFAULT_FILTER_COLOR_JSON__", stats.json_for_script(stats.DEFAULT_FILTER_COLOR))
          + stats.SORT_TABLE_JS)

    parts.append(f"""
<div id="modal-overlay">
  <div id="modal">
    <div id="modal-title"></div>
    <div id="modal-headline"></div>
    <div id="modal-img-wrap"></div>
    <canvas id="modal-histogram" height="96"></canvas>
    <div id="modal-levels"></div>
    <div id="modal-stats"></div>
    <div class="modal-actions">
      <button class="btn" id="linear-btn" onclick="setStretch('linear')">Linear</button>
      <button class="btn" id="auto-btn" onclick="setStretch('auto')">Auto stretch</button>
      <button class="btn" id="super-btn" onclick="setStretch('super')" title="Auto stretch, turned up: the sky background lifted to a brighter grey">Super stretch</button>
      <button class="btn" id="raw-btn" onclick="setSource('raw')" title="Show the frame as the camera recorded it" hidden>Raw</button>
      <button class="btn" id="calibrated-btn" onclick="setSource('calibrated')" title="Show the dark-subtracted, flat-divided frame the metrics were measured on" hidden>Calibrated</button>
      <button class="btn" id="zoom-btn" onclick="toggleZoom()" title="Inspect star shapes at 1:1 - click the preview to zoom there, drag to pan">100%</button>
      <button class="btn" id="trail-btn" onclick="toggleTrail()" title="Outline the detected satellite trail" hidden>&#128752; Trail</button>
      <span class="spacer"></span>
      <button class="btn" onclick="nextFile(-1)">&larr; Prev</button>
      <button class="btn" onclick="nextFile(1)">Next &rarr;</button>
      <button class="btn danger" id="delete-btn" onclick="toggleDelete()"></button>
      <button class="btn" onclick="closeModal()">Close</button>
    </div>
  </div>
</div>
<div id="grid-overlay">
  <div id="grid-modal">
    <div id="grid-header">
      <div id="grid-title"></div>
      <div id="grid-count"></div>
      <button class="btn" id="grid-prev-btn" onclick="navigateGroup(-1)">&lsaquo; Prev</button>
      <button class="btn" id="grid-next-btn" onclick="navigateGroup(1)">Next &rsaquo;</button>
      <span class="spacer"></span>
      <button class="btn danger" id="grid-delete-btn" onclick="deleteGridSelected()" disabled>Delete selected (0)</button>
      <button class="btn" id="grid-raw-btn" onclick="setSource('raw')" title="Show the frames as the camera recorded them" hidden>Raw</button>
      <button class="btn" id="grid-calibrated-btn" onclick="setSource('calibrated')" title="Show the calibrated frames" hidden>Calibrated</button>
      <button class="btn" id="grid-linear-btn" onclick="setGroupStretch('linear')">Linear</button>
      <button class="btn" id="grid-auto-btn" onclick="setGroupStretch('auto')">Auto stretch</button>
      <button class="btn" id="grid-super-btn" onclick="setGroupStretch('super')" title="Auto stretch, turned up: the sky background lifted to a brighter grey">Super stretch</button>
      <button class="btn" onclick="closeGroupGrid()">Close</button>
    </div>
    <div id="grid-wrap"></div>
  </div>
</div>
<script>{js}</script>
</main>
</div>
</body></html>""")

    return "\n".join(parts)


class Handler(BaseHTTPRequestHandler):
    ds: Dataset = None      # type: ignore[assignment]  # set externally before server starts
    allowed_hosts: frozenset = frozenset()

    # Keep-alive: the page pulls a fresh PNG per frame, and a new TCP connection
    # per preview is pure latency. Every response below sets Content-Length.
    protocol_version = "HTTP/1.1"

    # Parameter named to match BaseHTTPRequestHandler.log_message exactly: the base class
    # and log_error() both pass it positionally, but an override that renames it breaks any
    # caller that uses the keyword, and type checkers flag it as an incompatible override.
    def log_message(self, format, *args):   # noqa: A002 - name fixed by the base class
        sys.stderr.write("%s - %s\n" % (self.address_string(), format % args))

    def _send_bytes(self, body: bytes, content_type: str, status=200, extra_headers=()):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for name, value in extra_headers:
            self.send_header(name, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(self, obj, status=200):
        self._send_bytes(json.dumps(obj).encode(), "application/json", status)

    def _send_empty(self, status):
        self.send_response(status)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _is_same_origin(self) -> bool:
        """A page on another origin can POST JSON here without a preflight, and these
        endpoints move files around. Require the request to name this server."""
        host = (self.headers.get("Host") or "").strip()
        if host.lower() not in self.allowed_hosts:
            return False
        origin = self.headers.get("Origin")
        if origin and urlparse(origin).netloc.lower() not in self.allowed_hosts:
            return False
        return True

    def _parse_fid(self, value) -> int:
        fid = int(value)
        if not 0 <= fid < len(self.ds.files):
            raise ValueError(f"unknown frame id {fid}")
        return fid

    def _frame(self, qs):
        """(id, path, cache key) for the ?id= of a render request. One stat() call: the
        key doubles as the existence check the endpoints used to make separately.

        The key carries the path, so raw and calibrated renders of one frame never collide
        in any of the caches or ETags."""
        fid = self._parse_fid(qs.get("id", ["-1"])[0])
        path = self.ds.current_path(fid, self._parse_source(qs))
        try:
            return fid, path, file_key(path)
        except OSError:
            raise FileNotFoundError(str(path))

    @staticmethod
    def _parse_source(qs) -> str:
        source = qs.get("source", [SOURCE_RAW])[0] or SOURCE_RAW
        if source not in SOURCES:
            raise ValueError(f"unknown source {source!r}")
        return source

    @staticmethod
    def _parse_levels(qs, path, stretch_mode, key):
        """(black, white, mid) for a render: the handles the client is showing if it sent
        them, otherwise where the named preset puts them for this frame."""
        if "black" in qs and "white" in qs:
            black, white = float(qs["black"][0]), float(qs["white"][0])
            mid = float(qs.get("mid", ["0.5"])[0])
            if not (math.isfinite(black) and math.isfinite(white) and math.isfinite(mid)):
                raise ValueError("black, white and mid must be finite numbers")
            if white <= black:
                raise ValueError("white point must be above the black point")
            return black, white, min(max(mid, MIDTONE_MIN), MIDTONE_MAX)
        levels = preview_levels(path, stretch_mode, key)
        return levels["black"], levels["white"], levels["mid"]

    @staticmethod
    def _levels_tag(levels) -> str:
        return "%.4f-%.4f-%.5f" % levels

    @staticmethod
    def _parse_stretch(qs) -> str:
        mode = qs.get("stretch", ["auto"])[0]
        if mode not in STRETCH_MODES:
            raise ValueError(f"unknown stretch {mode!r}")
        return mode

    def _send_revalidated(self, body: bytes, content_type: str, etag: str):
        self._send_bytes(body, content_type, extra_headers=[
            ("ETag", etag),
            ("Cache-Control", "private, max-age=0, must-revalidate"),
        ])

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_bytes(build_page(self.ds).encode(), "text/html; charset=utf-8",
                             extra_headers=[("Cache-Control", "no-store")])
            return

        if parsed.path == "/api/preview":
            qs = parse_qs(parsed.query)
            try:
                fid, path, key = self._frame(qs)
                stretch_mode = self._parse_stretch(qs)
                levels = self._parse_levels(qs, path, stretch_mode, key)
                etag = f'"{fid}-{self._levels_tag(levels)}-{key[1]}-{key[2]}"'
                if self.headers.get("If-None-Match") == etag:
                    self._send_empty(304)
                    return
                png = previews.get(key + levels, lambda: render_levels_png(
                    load_decimated(path, PREVIEW_MAX_SIZE, key), *levels))
            except Exception as e:
                self._send_json({"error": str(e) or repr(e)}, status=400)
                return
            self._send_revalidated(png, "image/png", etag)
            return

        if parsed.path == "/api/histogram":
            qs = parse_qs(parsed.query)
            try:
                fid, path, key = self._frame(qs)
                etag = f'"{fid}-hist-{key[1]}-{key[2]}"'
                if self.headers.get("If-None-Match") == etag:
                    self._send_empty(304)
                    return
                histogram = histograms.get(key, lambda: render_histogram(path, key=key))
            except Exception as e:
                self._send_json({"error": str(e) or repr(e)}, status=400)
                return
            self._send_revalidated(json.dumps(histogram).encode(), "application/json", etag)
            return

        if parsed.path == "/api/trail":
            qs = parse_qs(parsed.query)
            try:
                fid, path, key = self._frame(qs)
                frame = self.ds.files[fid]
                # Revalidate before measuring, as every other endpoint here does: the
                # geometry below costs ~0.7 s on a cold frame, and a 304 needs none of it.
                etag = f'"{fid}-trail-{key[1]}-{key[2]}"'
                if self.headers.get("If-None-Match") == etag:
                    self._send_empty(304)
                    return
                # Current scans store the line itself. A dataset from an older scan stored only
                # the flag, so there the line is re-measured for the one frame being looked
                # at, with its own FWHM so the width gate matches. ~0.7 s, then cached.
                if not frame.get("has_trail"):
                    geometry = None
                elif frame.get("trail"):
                    geometry = frame["trail"]
                else:
                    geometry = trails.get(
                        key, lambda: stats.locate_trail(path, frame.get("fwhm_avg")))
            except Exception as e:
                self._send_json({"error": str(e) or repr(e)}, status=400)
                return
            self._send_revalidated(json.dumps({"trail": geometry}).encode(),
                                   "application/json", etag)
            return

        if parsed.path == "/api/crop":
            qs = parse_qs(parsed.query)
            try:
                fid, path, key = self._frame(qs)
                stretch_mode = self._parse_stretch(qs)
                levels = self._parse_levels(qs, path, stretch_mode, key)
                x, y = int(qs.get("x", ["0"])[0]), int(qs.get("y", ["0"])[0])
                w, h = int(qs.get("w", ["800"])[0]), int(qs.get("h", ["600"])[0])
                etag = f'"{fid}-crop-{self._levels_tag(levels)}-{x}-{y}-{w}-{h}-{key[1]}-{key[2]}"'
                if self.headers.get("If-None-Match") == etag:
                    self._send_empty(304)
                    return
                png = crops.get(key + levels + (x, y, w, h),
                                lambda: render_crop_png(path, x, y, w, h, levels, key))
            except Exception as e:
                self._send_json({"error": str(e) or repr(e)}, status=400)
                return
            self._send_revalidated(png, "image/png", etag)
            return

        if parsed.path == "/api/frameinfo":
            qs = parse_qs(parsed.query)
            try:
                fid, path, key = self._frame(qs)
                fh, fw = frame_shape(path, key)
                etag = f'"{fid}-info-{key[1]}-{key[2]}"'
                if self.headers.get("If-None-Match") == etag:
                    self._send_empty(304)
                    return
            except Exception as e:
                self._send_json({"error": str(e) or repr(e)}, status=400)
                return
            self._send_revalidated(json.dumps({"width": fw, "height": fh}).encode(),
                                   "application/json", etag)
            return

        if parsed.path == "/api/thumb":
            # keep_blank_values: "exp=" is the group of frames whose header carried no
            # usable exposure, which is a real group. Dropped as blank it would read as
            # "exp absent" and silently pool every length into that group's stretch.
            qs = parse_qs(parsed.query, keep_blank_values=True)
            try:
                fid, path, key = self._frame(qs)
                stretch_mode = self._parse_stretch(qs)
                panel = qs.get("panel", [""])[0]
                filt = qs.get("filter", [""])[0]
                # Absent (an older URL, or a filter shot at a single length) means "every
                # exposure", which for such a group is the whole group anyway.
                exposure = qs.get("exp", [None])[0]
                # Fitted for the source this frame is actually being shown in, so a
                # frame falling back to raw is stretched against the other raw frames.
                effective = self.ds.effective_source(fid, self._parse_source(qs))
                levels = group_stretch_limits(self.ds, panel, filt, stretch_mode, exposure,
                                              effective)
                etag = f'"{fid}-thumb-{self._levels_tag(levels)}-{key[1]}-{key[2]}"'
                if self.headers.get("If-None-Match") == etag:
                    self._send_empty(304)
                    return
                png = thumbs.get(key + levels,
                                 lambda: render_thumb_png(path, *levels, key=key))
            except Exception as e:
                self._send_json({"error": str(e) or repr(e)}, status=400)
                return
            self._send_revalidated(png, "image/png", etag)
            return

        self._send_empty(404)

    do_HEAD = do_GET

    def _read_body(self) -> bytes:
        """Read the whole request body before replying to anything.

        This connection is keep-alive (HTTP/1.1), so answering a POST without consuming
        its body leaves those bytes in the stream to be parsed as the next request line -
        which is exactly what the early 404/403 returns below used to do.
        """
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = -1
        if not 0 <= length <= MAX_POST_BYTES:
            self.close_connection = True   # cannot resync the stream; drop it instead
            raise ValueError("missing, malformed or oversized request body")
        return self.rfile.read(length) if length else b""

    def do_POST(self):
        parsed = urlparse(self.path)
        try:
            body = self._read_body()
        except ValueError as e:
            self._send_json({"error": str(e)}, status=413)
            return

        action = {"/api/delete": self.ds.delete_many,
                  "/api/restore": self.ds.restore_many}.get(parsed.path)
        if action is None:
            self._send_empty(404)
            return
        if not self._is_same_origin():
            self._send_json({"error": "cross-origin request rejected"}, status=403)
            return

        try:
            payload = json.loads(body or b"{}")
            requested = payload["ids"] if "ids" in payload else [payload["id"]]
            fids = [self._parse_fid(v) for v in requested]
        except Exception as e:
            self._send_json({"error": str(e) or repr(e)}, status=400)
            return

        failed = action(fids)
        for fid, message in failed.items():
            self.log_error("%s failed for id %s: %s", parsed.path, fid, message)
        if failed and len(failed) == len(fids):
            self._send_json({"error": next(iter(failed.values())), "failed": failed}, status=500)
            return
        self._send_json({"ok": not failed, "failed": failed})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-j", "--json", default="fits_light_data.json", help="Cached dataset JSON from fits_light_stats.py")
    ap.add_argument("-r", "--root", default=".", help="Root folder the dataset was scanned from")
    ap.add_argument("-p", "--port", type=int, default=8765)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    json_path = Path(args.json)
    if not json_path.is_absolute():
        json_path = root / json_path
    if not json_path.exists():
        print(f"Dataset JSON not found: {json_path}", file=sys.stderr)
        print("Run fits_light_stats.py first with -j to produce it.", file=sys.stderr)
        sys.exit(1)

    Handler.ds = Dataset(json_path, root)
    Handler.allowed_hosts = frozenset({
        f"127.0.0.1:{args.port}", f"localhost:{args.port}", f"[::1]:{args.port}",
    })

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Serving on http://127.0.0.1:{args.port}  (Ctrl+C to stop)", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.", file=sys.stderr)
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
