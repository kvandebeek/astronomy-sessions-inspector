"""The whole scan: measurement, incremental rescans, checkpoints, WCS export, the CLI."""
import json
import subprocess
import sys

import numpy as np
from astropy.io import fits

import fits_light_stats as st
import synth

SCRIPT = str(st.Path(st.__file__))


def scan(project, **kwargs):
    kwargs.setdefault("workers", 1)
    kwargs.setdefault("progress", False)
    return st.build_dataset(project, darks_dir=project / "darks", **kwargs)


def rows(dataset):
    return [r for e in dataset for r in e["files"]]


def test_scan_measures_and_calibrates(project):
    dataset = scan(project)
    assert [e["folder"] for e in dataset] == ["night 1/Panel 1/H/LIGHT", "night 2/Panel 1/H/LIGHT"]
    for r in rows(dataset):
        assert "error" not in r
        assert r["calibration"] == "dark 30s + flat" and r["calibration_warnings"] == []
        assert r["num_stars"] > 20 and 3.0 < r["fwhm_avg"] < 4.5
        assert r["fwhm_arcsec"] is not None and r["scale_arcsec"] > 0
    trails = [r["name"] for r in rows(dataset) if r["has_trail"]]
    assert trails == ["L_1_002.fits", "L_1_002.fits"]


def test_incremental_rescan_reuses_unchanged_frames(project, monkeypatch):
    first = scan(project)
    previous = {r["path"]: r for r in rows(first)}
    calls = []
    real = st.process_file_safe
    monkeypatch.setattr(st, "process_file_safe", lambda p, *a: calls.append(p) or real(p, *a))
    scan(project, previous=previous)
    assert calls == []
    # Touch one frame: only that one is measured again.
    changed = next((project / "night 2").rglob("LIGHT/*.fits"))
    with fits.open(changed, mode="update") as h:
        h[0].header["COMMENT"] = "edited"
    scan(project, previous=previous)
    assert calls == [changed]


def test_new_flats_invalidate_their_session_only(project, monkeypatch):
    previous = {r["path"]: r for r in rows(scan(project))}
    extra = project / "night 1" / "flats" / "H" / "FLAT" / "flat_9.fits"
    extra.write_bytes((extra.parent / "flat_0.fits").read_bytes())
    calls = []
    real = st.process_file_safe
    monkeypatch.setattr(st, "process_file_safe", lambda p, *a: calls.append(p) or real(p, *a))
    scan(project, previous=previous)
    assert calls and all("night 1" in str(p) for p in calls)


def test_header_wcs_is_exported(tmp_path):
    root = synth.make_project(tmp_path / "p", sessions=("night 1",), frames=1)
    light = next(root.rglob("LIGHT/*.fits"))
    with fits.open(light, mode="update") as h:
        for k, v in (("CTYPE1", "RA---TAN"), ("CTYPE2", "DEC--TAN"), ("CRPIX1", 128.5),
                     ("CRPIX2", 128.5), ("CRVAL1", 10.68), ("CRVAL2", 41.27),
                     ("CD1_1", -0.0003), ("CD1_2", 0.0), ("CD2_1", 0.0), ("CD2_2", 0.0003),
                     ("OBJECT", "M31")):
            h[0].header[k] = v
    row = rows(st.build_dataset(root, darks_dir=root / "darks", workers=1, progress=False))[0]
    wcs = row["wcs"]
    assert wcs["source"] == "header" and row["object"] == "M31"
    assert abs(wcs["center_ra"] - 10.68) < 1e-3 and abs(wcs["center_dec"] - 41.27) < 1e-3
    assert np.allclose(wcs["pixel_scale_arcsec"], 1.08, atol=1e-3)
    assert abs(wcs["rotation_deg"]) < 1e-6 and wcs["flipped"] is False
    assert len(wcs["corners"]) == 4 and wcs["cards"]["CTYPE1"] == "RA---TAN"


def test_cli_scan_rescan_and_checkpoint_cleanup(project):
    cache = project / "c.json"
    run = [sys.executable, SCRIPT, str(project), "-j", str(cache), "-o",
           str(project / "r.html"), "--darks", str(project / "darks"), "-w", "1"]
    subprocess.run(run, check=True, capture_output=True)
    data = json.loads(cache.read_text())        # strict JSON
    assert sum(len(e["files"]) for e in data) == 10
    out = subprocess.run(run + ["--rescan"], check=True, capture_output=True, text=True)
    assert "10 frame(s) unchanged" in out.stderr
    assert not (project / "c.json.partial").exists()
    assert "<table" in (project / "r.html").read_text()


def test_checkpoint_rows_are_picked_up(project):
    dataset = scan(project)
    checkpoint = project / "c.json.partial"
    st.write_json_atomic(checkpoint, {"rows": rows(dataset)[:3]})
    previous = st.load_previous_rows(project / "missing.json", checkpoint)
    assert len(previous) == 3


def test_unreadable_frame_becomes_an_error_row(project):
    bad = project / "night 1" / "Panel 1" / "H" / "LIGHT" / "L_1_zzz.fits"
    bad.write_bytes(b"not a fits file")
    dataset = scan(project)
    err = [r for r in rows(dataset) if "error" in r]
    assert [r["name"] for r in err] == ["L_1_zzz.fits"]
    st.write_json_atomic(project / "c.json", dataset)    # an all-error folder must not NaN
