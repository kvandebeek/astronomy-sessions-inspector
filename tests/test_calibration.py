"""XISF reading, settings matching and the calibration plan."""
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

import fits_light_stats as st
import synth


@pytest.mark.parametrize("compress", [False, True])
def test_read_xisf_float_with_bounds(tmp_path, compress):
    img = np.linspace(400, 4000, synth.SIZE * synth.SIZE).reshape(synth.SIZE, synth.SIZE)
    path = tmp_path / "m.xisf"
    synth.write_xisf(path, img, compress=compress)
    out = st.read_xisf(path)
    assert out.dtype == np.float32 and out.flags.writeable
    assert np.allclose(out, img, rtol=1e-5)


def test_read_xisf_without_bounds_reads_adu(tmp_path, capsys):
    path = tmp_path / "m.xisf"
    synth.write_xisf(path, np.full((synth.SIZE, synth.SIZE), 920.0), bounds=False)
    assert np.allclose(st.read_xisf(path), 920.0)
    assert "declares no sample bounds" in capsys.readouterr().err


def test_xisf_settings_from_keywords_and_name(tmp_path):
    path = tmp_path / "masterDark_BIN-2_GAIN-200_EXPOSURE-60.00s.xisf"
    synth.write_xisf(path, np.zeros((synth.SIZE, synth.SIZE)), {"GAIN": 100, "CCD-TEMP": -10})
    settings = st.xisf_master_settings(path)
    # The keyword wins over the name where both say; the name fills in the rest.
    assert settings["gain"] == 100 and settings["binning"] == "2x2" and settings["temp"] == -10


def manifest(**extra):
    m = {"root": "/r", "biases": [], "darks": [], "flats": {}, "calibrated_dir": None}
    m.update(extra)
    return m


def test_plan_skips_a_dark_at_another_gain_and_uses_the_matching_bias():
    m = manifest(
        darks=[{"path": "d60", "exposure": 60.0, "settings": {"gain": 200}}],
        biases=[{"path": "b200", "settings": {"gain": 200}, "level": 900},
                {"path": "b100", "settings": {"gain": 100}, "level": 500}])
    plan = st.calibration_plan(m, Path("/r/n/p/H/LIGHT/x.fits"), 60.0, "H", {"gain": 100})
    assert plan["dark"] == "b100" and plan["label"] == "bias" and plan["pedestal"] == 500
    assert any("gain 200 vs 100" in w for w in plan["warnings"])


def test_plan_warns_on_temperature_gap():
    m = manifest(darks=[{"path": "d30", "exposure": 30.0, "settings": {"temp": -20.0}}])
    plan = st.calibration_plan(m, Path("/r/n/p/H/LIGHT/x.fits"), 30.0, "H", {"temp": -10.0})
    assert plan["dark"] == "d30"
    assert any("off the frame's sensor temperature" in w for w in plan["warnings"])


def test_plan_matches_flat_case_insensitively():
    m = manifest(flats={("/r/n", "HA"): {"path": "f", "settings": {}}})
    plan = st.calibration_plan(m, Path("/r/n/p/Ha/LIGHT/x.fits"), 30.0, "Ha", {})
    assert plan["flat"] == "f"


def test_name_tag_keeps_punctuation_variants_apart():
    assert st._name_tag("night 1", "H") != st._name_tag("night_1", "H")


def test_prepare_calibration_end_to_end(project):
    m = st.prepare_calibration(project, project / "darks", progress=False)
    assert len(m["biases"]) == 1 and len(m["darks"]) == 1 and len(m["flats"]) == 2
    flat = np.load(next(iter(m["flats"].values()))["path"])
    assert abs(float(np.median(flat)) - 1.0) < 0.02 and flat.min() >= 0.05
    # A second run builds nothing and prunes nothing it still uses.
    cache = project / st.CALIB_DIRNAME
    before = sorted(p.name for p in cache.glob("*.npy"))
    st.prepare_calibration(project, project / "darks", progress=False)
    assert sorted(p.name for p in cache.glob("*.npy")) == before


def test_stale_master_flat_is_pruned(project):
    st.prepare_calibration(project, project / "darks", progress=False)
    cache = project / st.CALIB_DIRNAME
    flats_before = set(cache.glob("flat_*.npy"))
    extra = project / "night 1" / "flats" / "H" / "FLAT" / "flat_9.fits"
    extra.write_bytes((extra.parent / "flat_0.fits").read_bytes())
    st.prepare_calibration(project, project / "darks", progress=False)
    flats_after = set(cache.glob("flat_*.npy"))
    assert len(flats_after) == 2 and len(flats_before & flats_after) == 1


def test_missing_darks_folder_warns(project, capsys):
    st.prepare_calibration(project, project / "nope", progress=False)
    assert "does not exist" in capsys.readouterr().err


def test_flat_without_bias_subtracts_the_pedestal_first(tmp_path):
    root = synth.make_project(tmp_path / "p", sessions=("night 1",), with_masters=False)
    light = next((root / "night 1").rglob("LIGHT/*.fits"))
    calib = st.prepare_calibration(root, None, progress=False)
    row = st.process_file(light, calib, {"pedestal": synth.PEDESTAL})
    assert row["calibration"] == "flat"
    assert any("nominal" in w for w in row["calibration_warnings"])
    # Sky is 150 ADU over the pedestal; divided by a ~0.9 flat it stays near that, rather
    # than the ~600 a pedestal divided through the flat would add.
    assert 100 < row["signal_adu"] < 250


def test_calibrated_copy_is_rewritten_when_masters_change(project, tmp_path):
    out_dir = tmp_path / "cal"
    calib = st.prepare_calibration(project, project / "darks", progress=False,
                                   calibrated_dir=out_dir)
    light = next((project / "night 1").rglob("LIGHT/*.fits"))
    row = st.process_file(light, calib, {})
    first = fits.getheader(row["calibrated_path"])["CALSIG"]
    calib["flats"] = {}                     # the flat stops applying
    row = st.process_file(light, calib, {})
    assert fits.getheader(row["calibrated_path"])["CALSIG"] != first
