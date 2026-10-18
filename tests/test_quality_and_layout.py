"""Pure functions: quality score, folder parsing, sorting, exposure keys, JSON output."""
import json
import math

import fits_light_stats as st


def good_row(**overrides):
    row = {"hfr_avg": 2.0, "fwhm_avg": 3.0, "num_stars": 300, "roundness_avg": 0.85,
           "elong_uniform_r": 0.1, "elong_radial_r": 0.1, "filter": "L", "has_trail": False}
    row.update(overrides)
    return row


def test_clean_frame_scores_good():
    score, reasons, verdict = st.compute_quality(good_row())
    assert verdict == "good" and score > 95 and reasons == []


def test_trail_costs_points_and_caps_verdict_at_warn():
    clean, _, _ = st.compute_quality(good_row())
    score, reasons, verdict = st.compute_quality(good_row(has_trail=True))
    assert clean - score == st.QUALITY_TRAIL_PENALTY
    assert verdict == "warn"
    assert reasons[0] == "satellite trail"


def test_no_stars_is_a_reject():
    _, reasons, verdict = st.compute_quality(good_row(num_stars=0, hfr_avg=None, fwhm_avg=None))
    assert verdict == "reject" and "no stars detected" in reasons


def test_reference_scale_scores_angular_size(monkeypatch):
    # 4 px at 2"/px is 8": against thresholds tuned at 1"/px that is a bad FWHM.
    row = good_row(fwhm_avg=4.0, pixel_scale_arcsec=2.0)
    unscaled, _, _ = st.compute_quality(row)
    monkeypatch.setattr(st, "REFERENCE_PIXEL_SCALE_ARCSEC", 1.0)
    scaled, reasons, _ = st.compute_quality(row)
    assert scaled < unscaled
    assert any("FWHM" in r for r in reasons)


def test_parse_folder_layouts():
    assert st.parse_folder("night 1/M31/H/LIGHT") == ("night 1", "M31", "H")
    assert st.parse_folder("2024/night 1/M31/H/light") == ("2024/night 1", "M31", "H")
    assert st.parse_folder("M31/H/LIGHT") == (st.ROOT_LABEL, "M31", "H")
    assert st.parse_folder("H/LIGHT") == (st.ROOT_LABEL, st.ROOT_LABEL, "H")
    assert st.parse_folder("LIGHT") is None
    assert st.parse_folder("night 1/M31/H/FLAT") is None


def test_natural_sort():
    names = ["Panel 10", "Panel 2", "Bode's Galaxy", "panel 1", "2024-11-03", "2024-10-12"]
    assert sorted(names, key=st.natural_num) == [
        "2024-10-12", "2024-11-03", "Bode's Galaxy", "panel 1", "Panel 2", "Panel 10"]


def test_exposure_key():
    assert st.exposure_key(30.0) == "30"
    assert st.exposure_key(0.5) == "0.5"
    assert st.exposure_key(True) == "" and st.exposure_key("30") == "" and st.exposure_key(0) == ""


def test_json_output_is_strict(tmp_path):
    out = tmp_path / "x.json"
    st.write_json_atomic(out, {"a": float("nan"), "b": [1.0, float("inf")], "c": {"d": -0.5}})
    text = out.read_text()
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text) == {"a": None, "b": [1.0, None], "c": {"d": -0.5}}
    assert not (tmp_path / "x.json.tmp").exists()


def test_json_for_script_cannot_close_the_script_tag():
    assert "</script>" not in st.json_for_script({"name": "</script><b>"})
    assert "NaN" not in st.json_for_script({"v": math.nan})
