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
import errno
import io
import json
import math
import re
import shutil
import sys
import threading
import time
import html
import socket
import webbrowser
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
GROUP_MEMBERS_TTL_S = 30.0   # how long a group's member list is trusted without a re-stat


class Dataset:
    """The scanned frames plus their deleted/restored state. Mutating methods are
    serialised: ThreadingHTTPServer runs each request on its own thread."""

    def __init__(self, json_path: Path, root: Path):
        self.root = root
        # Identifies this load of the JSON. The page sends it back with every delete or
        # restore, so ids from a page built before a rescan can never hit the frames that
        # the same ids name after it.
        self.version = str(json_path.stat().st_mtime_ns)
        self.deleted_dir = root / DELETED_SUBDIR
        self.lock = threading.Lock()
        # Bumped by every delete/restore that moved something; see group_members().
        self.generation = 0
        self._members_lock = threading.Lock()
        self._members = {}
        self.raw = stats.add_dataset_derived_fields(
            json.loads(json_path.read_text(encoding="utf-8")))

        self.folders = []
        self.files = []  # flat list, index == id
        # Frames the scan could not read. Kept, and shown at the top of the page: dropping
        # them made a failed frame indistinguishable from one that was never there.
        self.errors = []
        for folder_entry in self.raw:
            folder_name = folder_entry["folder"]
            parsed = stats.parse_folder(folder_name)
            folder_id = len(self.folders)
            file_ids = []
            for f in folder_entry["files"]:
                if "error" in f:
                    self.errors.append({"folder": folder_name, "name": f.get("name", ""),
                                        "path": f.get("path", ""), "error": f["error"]})
                    continue
                f = dict(f)
                # The path the scan recorded. State is keyed on it, since `path` itself
                # changes when a restore has to use another name.
                f["source_path"] = f["path"]
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

        # Frames a restore had to bring back under another name (see _move_from_deleted).
        for source, current in (state.get("renamed") or {}).items():
            f = self.by_path.get(source)
            if f is not None and Path(current).exists():
                self.by_path.pop(f["path"], None)
                f["path"] = current
                self.by_path[current] = f

        for key, info in (state.get("deleted") or {}).items():
            if not isinstance(info, dict) or not info.get("deleted_path"):
                continue
            f = self.by_path.get(info.get("path") or key) or next(
                (g for g in self.files if g["source_path"] == (info.get("path") or key)), None)
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
                f["source_path"]: {"path": f["source_path"], "deleted_path": f["deleted_path"]}
                for f in self.files if f.get("deleted")
            },
            # A restore that had to pick a new name. Without this a restart pointed the
            # frame back at its original path - by then another file's.
            "renamed": {
                f["source_path"]: f["path"]
                for f in self.files if f["path"] != f["source_path"]
            },
        }
        stats.write_json_atomic(self.state_path, payload)

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
            ) from None
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
                self.generation += 1
                self.save_state()
        return failed

    def purge_deleted(self):
        """Permanently remove every frame currently in _deleted. Returns (count, failures).

        Only files the reviewer itself moved there, and only inside the _deleted folder,
        are touched. The frames stay listed as deleted and become `missing`.
        """
        removed, failed = 0, {}
        deleted_root = self.deleted_dir.resolve()
        with self.lock:
            for f in self.files:
                if not f.get("deleted") or f.get("missing"):
                    continue
                target = Path(f["deleted_path"]).resolve()
                if deleted_root not in target.parents:
                    failed[f["id"]] = "not inside _deleted; left alone"
                    continue
                try:
                    target.unlink()
                except OSError as e:
                    failed[f["id"]] = str(e)
                    continue
                f["missing"] = True
                removed += 1
                # Tidy the now-empty mirror folders, up to _deleted itself.
                d = target.parent
                while d != deleted_root and deleted_root in d.parents:
                    try:
                        d.rmdir()
                    except OSError:
                        break
                    d = d.parent
            if removed:
                self.generation += 1
                self.save_state()
        return removed, failed

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

    def group_members(self, panel: str, filt: str, exposure, source: str):
        """file_key() of every live frame in one thumbnail group that really is in `source`.

        Every thumbnail request needs this list to find its group's shared stretch, and
        building it stats every member - so one grid of n tiles cost n*n stat calls. It is
        kept per delete/restore generation, and re-checked after GROUP_MEMBERS_TTL_S so a
        file rewritten on disk (a rescan with --save-calibrated) is still noticed.
        """
        key = (panel, filt, exposure, source)
        now = time.monotonic()
        with self._members_lock:
            hit = self._members.get(key)
            if hit and hit[0] == self.generation and now - hit[1] < GROUP_MEMBERS_TTL_S:
                return hit[2]
        generation = self.generation
        live = []
        for fid in self.group_file_ids(panel, filt, exposure):
            if self.files[fid].get("deleted"):
                continue
            # Only frames that really are in this source contribute to its shared stretch.
            if self.effective_source(fid, source) != source:
                continue
            try:
                live.append(file_key(self.current_path(fid, source)))
            except OSError:
                continue
        members = tuple(live)
        with self._members_lock:
            self._members[key] = (generation, now, members)
        return members

    def group_exposures(self, panel: str, filt: str):
        """The distinct exposure lengths one panel+filter was shot at, shortest first."""
        return sorted({stats.exposure_key(self.files[fid].get("exposure"))
                       for fid in self.group_file_ids(panel, filt)},
                      key=stats.exposure_sort_key)


class DatasetHolder:
    """The current Dataset, reloaded when the JSON on disk changes.

    A rescan used to need a server restart before the page showed anything new. The
    file's mtime is checked at most every RELOAD_CHECK_S; a JSON that fails to load (a
    scan halfway through writing it, say) keeps the previous dataset in service.
    """

    RELOAD_CHECK_S = 2.0

    def __init__(self, json_path: Path, root: Path):
        self.json_path, self.root = json_path, root
        self.lock = threading.Lock()
        self.current = Dataset(json_path, root)
        self._checked = time.monotonic()

    def get(self) -> Dataset:
        now = time.monotonic()
        if now - self._checked < self.RELOAD_CHECK_S:
            return self.current
        with self.lock:
            self._checked = now
            try:
                changed = str(self.json_path.stat().st_mtime_ns) != self.current.version
            except OSError:
                return self.current
            if changed:
                try:
                    fresh = Dataset(self.json_path, self.root)
                except (OSError, ValueError) as e:
                    print(f"Keeping the loaded dataset; {self.json_path} did not load: {e}",
                          file=sys.stderr)
                else:
                    with self.current.lock:    # let an in-flight delete finish first
                        self.current = fresh
                    print(f"Reloaded {self.json_path} ({len(fresh.files)} frames)",
                          file=sys.stderr)
            return self.current


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
    live = ds.group_members(panel, filt, exposure, source)
    return group_stretches.get((panel, filt, exposure, stretch_mode, source, live),
                               lambda: _compute_group_stretch(live, stretch_mode))


PAGE_CSS = stats.BASE_CSS + stats.MATRIX_CSS + stats.static_text("review.css")

PAGE_JS = stats.static_text("review.js")


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
        disabled = " disabled" if r.get("missing") else ""
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
            disabled = " disabled" if r.get("missing") else ""
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
CLIENT_FILE_FIELDS = ("id", "name", "path", "date_obs", "filter", "folder_id", "exposure",
                      "adu_min", "adu_max", "hfr_arcsec", "fwhm_arcsec", "scale_arcsec",
                      "calibration_warnings", "rel_sky", "rel_stars", "rel_fwhm",
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
<div class="sub bulk-actions"><button class="btn danger" id="bulk-delete-btn" onclick="bulkDeleteSelected()" disabled>Delete selected (0)</button>
<button class="btn" id="bulk-restore-btn" onclick="bulkRestoreSelected()" disabled>Restore selected (0)</button></div>
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

    parts.append('<div class="clip-panel side-tools">')
    parts.append('<label for="select-by">Select</label>')
    parts.append('<select id="select-by" onchange="selectBy(this.value); this.value=\'\'">'
                 '<option value="">Add to the selection&hellip;</option>'
                 '<option value="reject">every &ldquo;reject&rdquo;</option>'
                 '<option value="warn">every &ldquo;warn&rdquo; or worse</option>'
                 '<option value="trail">every frame with a trail</option>'
                 '<option value="clipped">every sigma-clipped frame</option>'
                 '<option value="none">clear the selection</option></select>')
    parts.append('<label class="check"><input type="checkbox" id="show-deleted" checked '
                 'onchange="setShowDeleted(this.checked)"> Show deleted frames</label>')
    parts.append('<button class="btn danger" id="purge-btn" onclick="purgeDeleted()" disabled>'
                 'Empty _deleted (0)</button>')
    parts.append('<div class="side-row"><button class="btn" onclick="exportCsv()" '
                 'title="Every frame with its metrics, verdict and review state">Export CSV</button>'
                 '<button class="btn" onclick="exportRejectList()" title="Paths of every frame '
                 'deleted here or currently selected, one per line - for a stacker&rsquo;s '
                 'rejection or exclusion list">Export reject list</button></div>')
    if sessions:
        parts.append('<label>Nights</label><div class="side-row">')
        for session in sessions:
            parts.append(f'<button class="btn" onclick="openTimeline({html.escape(json.dumps(session), quote=True)})">'
                         f'{html.escape(session)}</button>')
        parts.append('</div>')
    parts.append('</div>')

    stats.render_metric_buttons(parts)
    parts.append('</aside>')
    parts.append('<main class="data-pane">')
    parts.append('<div id="reload-banner" hidden>This dataset was rescanned since the page '
                 'loaded. <button class="btn" onclick="location.reload()">Reload</button></div>')
    if ds.errors:
        parts.append(f'<h2>Unreadable frames ({len(ds.errors)})</h2>'
                     '<div class="meta">The scan could not measure these; they are left '
                     'out of every statistic below.</div>'
                     '<div class="wrap"><table><thead><tr><th>Folder</th><th>File</th>'
                     '<th>Error</th></tr></thead><tbody>')
        for e in ds.errors:
            parts.append(f'<tr class="error-row"><td>{html.escape(e["folder"])}</td>'
                         f'<td title="{html.escape(e["path"], quote=True)}">{html.escape(e["name"])}</td>'
                         f'<td class="err">{html.escape(str(e["error"]))}</td></tr>')
        parts.append('</tbody></table></div>')
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

    review_data = {
        "files": [{k: f.get(k) for k in CLIENT_FILE_FIELDS} for f in ds.files],
        "folders": ds.folders,
        "units": ds.units,
        "groups": group_order,
        "matrixMetrics": stats.COMPARISON_METRICS,
        "filterColors": stats.FILTER_COLORS,
        "defaultFilterColor": stats.DEFAULT_FILTER_COLOR,
        "version": ds.version,
        "narrowband": sorted(stats.NARROWBAND_FILTERS),
    }
    js = ("const REVIEW_DATA = " + stats.json_for_script(review_data) + ";\n"
          + stats.MATRIX_TOTALS_JS + PAGE_JS + stats.SORT_TABLE_JS)

    parts.append(f"""
<div id="modal-overlay">
  <div id="modal">
    <div id="modal-title"></div>
    <div id="modal-headline"></div>
    <div id="modal-img-wrap"></div>
    <canvas id="modal-histogram" height="96"></canvas>
    <div id="modal-levels"></div>
    <div id="modal-stats"></div>
    <div id="modal-warnings" hidden></div>
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
    <div class="meta modal-keys">&larr; &rarr; step through the frames in the order they are shown &middot; Del moves this frame to _deleted (reversible) and steps on &middot; Esc closes</div>
  </div>
</div>
<div id="timeline-overlay">
  <div id="timeline-modal">
    <div id="timeline-header">
      <div id="timeline-title"></div>
      <div id="timeline-legend"></div>
      <span class="spacer"></span>
      <button class="btn" onclick="closeTimeline()">Close</button>
    </div>
    <div class="meta">Each point is one frame, by the time it was taken. Deleted frames are left out; click a point to open it.</div>
    <div id="timeline-charts"></div>
    <div id="timeline-tip" hidden></div>
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
      <button class="btn" onclick="selectGridWhere('reject')" title="Tick every frame in this group the quality score rejects">Select rejects</button>
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
    holder: DatasetHolder = None   # type: ignore[assignment]  # set before the server starts
    allowed_hosts: frozenset = frozenset()

    @property
    def ds(self) -> Dataset:
        return self.holder.get()

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

    def _host_ok(self) -> bool:
        """The request names this server. Checked on every request, reads included: with
        DNS rebinding a hostile page can make its own name resolve to 127.0.0.1, and then
        read previews and frame data through the victim's browser - but the Host header
        it sends is still its own name."""
        host = (self.headers.get("Host") or "").strip()
        return host.lower() in self.allowed_hosts

    def _is_same_origin(self) -> bool:
        """A page on another origin can POST JSON here without a preflight, and these
        endpoints move files around. Require the request to name this server."""
        if not self._host_ok():
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
            raise FileNotFoundError(str(path)) from None

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
        if not self._host_ok():
            self._send_json({"error": "unknown Host header"}, status=403)
            return
        if parsed.path == "/":
            self._send_bytes(build_page(self.ds).encode(), "text/html; charset=utf-8",
                             extra_headers=[("Cache-Control", "no-store")])
            return

        if parsed.path == "/api/version":
            self._send_bytes(json.dumps({"version": self.ds.version}).encode(),
                             "application/json", extra_headers=[("Cache-Control", "no-store")])
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

        ds = self.ds
        action = {"/api/delete": ds.delete_many, "/api/restore": ds.restore_many,
                  "/api/purge": None}
        if parsed.path not in action:
            self._send_empty(404)
            return
        if not self._is_same_origin():
            self._send_json({"error": "cross-origin request rejected"}, status=403)
            return
        try:
            payload = json.loads(body or b"{}")
        except ValueError as e:
            self._send_json({"error": str(e)}, status=400)
            return
        if payload.get("version") != ds.version:
            self._send_json({"error": "the dataset was rescanned since this page loaded; "
                                      "reload the page"}, status=409)
            return

        if parsed.path == "/api/purge":
            expected = sum(1 for f in ds.files if f.get("deleted") and not f.get("missing"))
            # The count the user confirmed has to be the count that would go: a stale page
            # must not purge frames someone deleted in another tab after it was shown.
            if payload.get("confirm") != expected:
                self._send_json({"error": f"{expected} frame(s) are in _deleted now; "
                                          f"reload and confirm again"}, status=409)
                return
            removed, failed = ds.purge_deleted()
            self._send_json({"ok": not failed, "removed": removed, "failed": failed})
            return

        try:
            requested = payload["ids"] if "ids" in payload else [payload["id"]]
            fids = [self._parse_fid(v) for v in requested]
        except Exception as e:
            self._send_json({"error": str(e) or repr(e)}, status=400)
            return
        action = action[parsed.path]

        failed = action(fids)  # type: ignore[misc]
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
    ap.add_argument("--host", default="127.0.0.1",
                    help="Address to listen on (default: this machine only). 0.0.0.0 lets "
                         "other machines on the network in - and lets them delete frames")
    ap.add_argument("--allow-host", action="append", default=[], metavar="NAME[:PORT]",
                    help="Another name the page may be opened by (e.g. the capture PC's "
                         "hostname); repeatable. Requests naming any other host are refused")
    ap.add_argument("--open", action="store_true", help="Open the page in a browser")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    json_path = Path(args.json)
    if not json_path.is_absolute():
        json_path = root / json_path
    if not json_path.exists():
        print(f"Dataset JSON not found: {json_path}", file=sys.stderr)
        print("Run fits_light_stats.py first with -j to produce it.", file=sys.stderr)
        sys.exit(1)

    Handler.holder = DatasetHolder(json_path, root)
    names = {"127.0.0.1", "localhost", "[::1]"}
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        names |= {args.host, socket.gethostname(), socket.getfqdn()}
        try:
            names |= {ai[4][0] for ai in socket.getaddrinfo(socket.gethostname(), None)
                      if ai[0] == socket.AF_INET}
        except OSError:
            pass
        names.discard("0.0.0.0")
        print("WARNING: listening beyond this machine - anyone who can reach port "
              f"{args.port} can view and delete frames", file=sys.stderr)
    hosts = {f"{n}:{args.port}".lower() for n in names}
    hosts |= {(h if ":" in h else f"{h}:{args.port}").lower() for h in args.allow_host}
    Handler.allowed_hosts = frozenset(hosts)

    try:
        server = ThreadingHTTPServer((args.host, args.port), Handler)
    except OSError as e:
        if e.errno == errno.EADDRINUSE:
            sys.exit(f"Port {args.port} is already in use - is the reviewer already running? "
                     f"Pass -p to pick another port.")
        raise
    url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '::') else args.host}:{args.port}"
    print(f"Serving on {url}  (Ctrl+C to stop)", file=sys.stderr)
    if args.open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.", file=sys.stderr)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
