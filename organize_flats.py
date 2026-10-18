#!/usr/bin/env python3
"""
Put every flat frame where fits_light_stats.py looks for it.

A session is a top-level folder under the root, and its flats belong at

    <session>/<group>/<filter>/FLAT/GAIN<g>_BIN<x>x<y>/EXP_<s>/<file>

- the same depth the lights sit at (<session>/<target>/<filter>/LIGHT/...), which is how
the capture software writes them. <group> can be anything ("FlatWizard m31",
"Targetflats", ...), so folders that already follow this are left alone. A flat is
recognised by IMAGETYP in its header, not by where it sits, and its filter comes from the
FILTER card, so a folder of flats shot through several filters is split up correctly.

Misplaced flats go to the session's existing flat group when there is exactly one, and
to <session>/flats otherwise.

Without --apply nothing is touched: the planned moves are printed for review. With
--apply every move is appended to a log in the root, which --undo LOG reverses.

Usage:
    python3 organize_flats.py [root] [--apply] [--undo LOG]
"""
import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

from astropy.io import fits

FITS_EXTS = {".fit", ".fits", ".fts"}
# Calibration libraries and derived output, never sessions or flats of their own.
EXCLUDED_DIR_NAMES = {"_deleted", ".calibration", "calibrated", "biases", "darks", "__pycache__"}
FLAT_DIR_NAME = "FLAT"
LIGHT_DIR_NAME = "LIGHT"
DEFAULT_GROUP = "flats"
JUNK_FILES = {".DS_Store"}


def walk(directory: Path):
    """os.walk over directory, skipping excluded and hidden folders."""
    for dirpath, dirnames, filenames in os.walk(directory):
        dirnames[:] = sorted(d for d in dirnames
                             if d not in EXCLUDED_DIR_NAMES and not d.startswith("."))
        yield Path(dirpath), dirnames, sorted(filenames)


def sessions_of(root: Path):
    return sorted(p for p in root.iterdir()
                  if p.is_dir() and p.name not in EXCLUDED_DIR_NAMES and not p.name.startswith("."))


def is_flat_type(imagetyp):
    """IMAGETYP says flat - but a dark flat is a dark, not a flat."""
    t = str(imagetyp or "").upper().replace(" ", "").replace("_", "")
    return "FLAT" in t and "DARK" not in t


def canonical_filter(relparts):
    """The filter folder name when relparts (relative to the session) is <group>/<filter>/FLAT/...

    None when the file does not sit at that depth.
    """
    if len(relparts) >= 4 and relparts[2].upper() == FLAT_DIR_NAME:
        return relparts[1]
    return None


def exposure_folder(exptime):
    return f"EXP_{float(exptime):.2f}"


def settings_folder(header):
    gain = header.get("GAIN")
    xb, yb = header.get("XBINNING"), header.get("YBINNING")
    if gain is None or xb is None:
        return None
    return f"GAIN{int(round(float(gain)))}_BIN{int(xb)}x{int(yb or xb)}"


def scan_session(session: Path):
    """Flats and light filters in one session.

    Returns (flats, light_filters, problems): flats is a list of dicts per flat file,
    light_filters {filter: set(target folders)}, problems a list of strings.
    """
    flats, problems = [], []
    light_filters = defaultdict(set)
    for directory, _dirs, filenames in walk(session):
        rel = directory.relative_to(session).parts
        upper = [p.upper() for p in rel]
        in_light = LIGHT_DIR_NAME in upper
        in_flat_dir = FLAT_DIR_NAME in upper
        for name in filenames:
            if Path(name).suffix.lower() not in FITS_EXTS:
                continue
            path = directory / name
            if in_light:
                # Lights only need their filter; the header decides, as in the stats script.
                try:
                    filt = str(fits.getheader(path).get("FILTER", "")).strip()
                except OSError as e:
                    problems.append(f"unreadable: {path} ({e})")
                    continue
                light_filters[filt or "?"].add(rel[0] if rel else "")
                continue
            try:
                header = fits.getheader(path)
            except OSError as e:
                problems.append(f"unreadable: {path} ({e})")
                continue
            typ = header.get("IMAGETYP")
            if not (is_flat_type(typ) or (typ is None and in_flat_dir)):
                continue
            flats.append({"path": path, "rel": rel + (name,),
                          "filter": str(header.get("FILTER", "")).strip(),
                          "exptime": header.get("EXPTIME", header.get("EXPOSURE")),
                          "settings": settings_folder(header)})
    return flats, light_filters, problems


def plan_session(session: Path, flats):
    """(moves, problems): moves is [(src, dst)] for every misplaced flat."""
    groups = sorted({f["rel"][0] for f in flats if canonical_filter(f["rel"])})
    group = groups[0] if len(groups) == 1 else DEFAULT_GROUP
    moves, problems = [], []
    for f in flats:
        placed = canonical_filter(f["rel"])
        if placed is not None and placed.upper() == f["filter"].upper():
            continue
        if not f["filter"]:
            problems.append(f"no FILTER in header, left in place: {f['path']}")
            continue
        if f["exptime"] is None:
            problems.append(f"no EXPTIME in header, left in place: {f['path']}")
            continue
        dst = session / group / f["filter"] / FLAT_DIR_NAME
        if f["settings"]:
            dst = dst / f["settings"]
        dst = dst / exposure_folder(f["exptime"]) / f["path"].name
        moves.append((f["path"], dst))
    return moves, problems


def flat_sets_after(session: Path, flats, moves):
    """{filter: {FLAT folder: [exposures]}} as the session will look once moves are applied.

    Exposure subfolders are not separate sets: flat wizards vary the exposure frame by frame.
    """
    moved = {src: dst for src, dst in moves}
    sets = defaultdict(lambda: defaultdict(list))
    for f in flats:
        final = moved.get(f["path"], f["path"])
        rel = final.relative_to(session).parts
        filt = canonical_filter(rel)
        if filt is None:
            continue
        sets[filt.upper()][str(Path(*rel[:3]))].append(f["exptime"])
    return sets


def describe_set(folder, exposures):
    known = sorted(float(e) for e in exposures if e is not None)
    span = "" if not known else (f" at {known[0]:g}s" if known[0] == known[-1]
                                 else f" at {known[0]:g}-{known[-1]:g}s")
    return f"{len(exposures)} flats in {folder}{span}"


def remove_empty_dirs(start: Path, stop: Path):
    """Remove start and its parents up to (not including) stop while they hold nothing
    but Finder litter."""
    d = start
    while d != stop and stop in d.parents:
        entries = list(d.iterdir())
        if any(e.name not in JUNK_FILES for e in entries):
            return
        for e in entries:
            e.unlink()
        d.rmdir()
        d = d.parent


def apply_moves(root: Path, moves):
    log = root / f"flat_moves_{time.strftime('%Y%m%d-%H%M%S')}.tsv"
    done = 0
    with log.open("a", encoding="utf-8") as out:
        for src, dst in moves:
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.rename(src, dst)  # same volume, so this is instant and never copies
            out.write(f"{src}\t{dst}\n")
            out.flush()
            done += 1
            remove_empty_dirs(src.parent, root)
    print(f"moved {done} flats; log: {log}")


def undo(root: Path, log: Path):
    lines = [l.rstrip("\n").split("\t") for l in log.read_text(encoding="utf-8").splitlines() if l]
    clashes = [src for src, _dst in lines if Path(src).exists()]
    missing = [dst for _src, dst in lines if not Path(dst).exists()]
    if clashes or missing:
        for p in clashes:
            print(f"  original location is occupied: {p}")
        for p in missing:
            print(f"  moved file is gone: {p}")
        sys.exit("undo aborted, nothing changed")
    for src, dst in reversed(lines):
        Path(src).parent.mkdir(parents=True, exist_ok=True)
        os.rename(dst, src)
        remove_empty_dirs(Path(dst).parent, root)
    print(f"moved {len(lines)} flats back")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", default=".", help="Folder holding the session folders")
    ap.add_argument("--apply", action="store_true", help="Actually move the files (default: dry run)")
    ap.add_argument("--undo", metavar="LOG", help="Reverse the moves recorded in a flat_moves_*.tsv log")
    args = ap.parse_args()
    root = Path(args.root).resolve()

    if args.undo:
        undo(root, Path(args.undo))
        return

    all_moves, remarks = [], []
    for session in sessions_of(root):
        flats, light_filters, problems = scan_session(session)
        if not flats and not light_filters:
            continue
        moves, plan_problems = plan_session(session, flats)
        problems += plan_problems
        clashes = [dst for _src, dst in moves if dst.exists()]
        seen = set()
        for _src, dst in moves:
            if dst in seen:
                clashes.append(dst)
            seen.add(dst)
        for dst in clashes:
            problems.append(f"destination already exists, would overwrite: {dst}")

        sets = flat_sets_after(session, flats, moves)
        print(f"\n== {session.name}")
        for filt in sorted(set(k.upper() for k in light_filters) | set(sets)):
            targets = sorted(t for k, v in light_filters.items() if k.upper() == filt for t in v)
            fl = sets.get(filt, {})
            print(f"  {filt:3} lights in {len(targets)} panel(s); "
                  + ("; ".join(describe_set(d, e) for d, e in sorted(fl.items())) or "NO FLATS"))
            if targets and not fl:
                remarks.append(f"{session.name}: {filt} lights ({', '.join(targets)}) have no flats")
            if len(fl) > 1:
                remarks.append(f"{session.name}: {filt} flats sit in {len(fl)} folders "
                               f"({', '.join(sorted(fl))}); they are combined into one master")
        if moves:
            print(f"  {len(moves)} flats to move:")
            by_dir = defaultdict(list)
            for src, dst in moves:
                by_dir[(src.parent.relative_to(root), dst.parent.relative_to(root))].append(src)
            for (s, d), files in sorted(by_dir.items()):
                print(f"    {len(files):3}  {s}/  ->  {d}/")
        for p in problems:
            print(f"  ! {p}")
        remarks += [f"{session.name}: {p}" for p in problems]
        all_moves += [(s, d) for s, d in moves if d not in clashes]

    print()
    for r in remarks:
        print(f"REMARK: {r}")
    if not all_moves:
        print("All flats are already in place.")
    elif args.apply:
        apply_moves(root, all_moves)
    else:
        print(f"Dry run: {len(all_moves)} flats would be moved. Rerun with --apply to move them.")


if __name__ == "__main__":
    main()
