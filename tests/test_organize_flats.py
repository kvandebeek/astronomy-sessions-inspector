import pytest
from astropy.io import fits

import organize_flats as of
import synth


def test_exposure_seconds_tolerates_bad_cards():
    assert of.exposure_seconds(fits.Header({"EXPTIME": 1.5})) == 1.5
    assert of.exposure_seconds(fits.Header({"EXPTIME": "abc"})) is None
    assert of.exposure_seconds(fits.Header()) is None


def test_misplaced_flats_are_planned_into_the_flat_layout(tmp_path):
    session = tmp_path / "night 1"
    field = synth.Field(size=64)
    synth.write_fits(session / "M31" / "H" / "LIGHT" / "l.fits", field.light(), FILTER="H")
    for i, filt in enumerate(("H", "O")):
        synth.write_fits(session / "dump" / f"f{i}.fits", field.flat(), FILTER=filt,
                         IMAGETYP="FLAT", EXPTIME=1.0, GAIN=100, XBINNING=1)
    flats, light_filters, problems = of.scan_session(session)
    assert problems == [] and set(light_filters) == {"H"}
    moves, problems = of.plan_session(session, flats)
    targets = sorted(str(dst.relative_to(session)) for _src, dst in moves)
    assert targets == ["flats/H/FLAT/GAIN100_BIN1x1/EXP_1.00/f0.fits",
                       "flats/O/FLAT/GAIN100_BIN1x1/EXP_1.00/f1.fits"]


def test_apply_and_undo_round_trip(tmp_path):
    session = tmp_path / "night 1"
    field = synth.Field(size=64)
    src = synth.write_fits(session / "dump" / "f.fits", field.flat(), FILTER="H",
                           IMAGETYP="FLAT", EXPTIME=1.0)
    flats, _lf, _p = of.scan_session(session)
    moves, _ = of.plan_session(session, flats)
    of.apply_moves(tmp_path, moves)
    assert not src.exists() and moves[0][1].exists()
    log = next(tmp_path.glob("flat_moves_*.tsv"))
    of.undo(tmp_path, log)
    assert src.exists()


def test_move_never_overwrites(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_text("a")
    b.write_text("b")
    with pytest.raises(FileExistsError):
        of.move_file(a, b)
    assert b.read_text() == "b"
