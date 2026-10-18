"""Synthetic FITS/XISF data for the tests: small star fields, masters and flats laid out
the way the scripts expect, so every test runs end to end in seconds without real data."""
import zlib
from pathlib import Path

import numpy as np
from astropy.io import fits

SIZE = 256
PEDESTAL = 500.0


def write_xisf(path: Path, image, keywords=None, compress=False, bounds=True, size=SIZE):
    """A minimal monolithic XISF file as PixInsight writes masters: float32, normalised to
    bounds 0:1 unless bounds=False, optionally zlib-compressed with byte shuffling."""
    arr = np.asarray(image, dtype=np.float32)
    if bounds:
        arr = arr / np.float32(65535.0)
    raw = arr.tobytes()
    comp = ""
    if compress:
        shuffled = np.frombuffer(raw, np.uint8).reshape(-1, 4).T.tobytes()
        comp = f' compression="zlib+sh:{len(raw)}:4"'
        raw = zlib.compress(shuffled)
    kw = "".join(f'<FITSKeyword name="{k}" value="{v}" comment=""/>'
                 for k, v in (keywords or {}).items())
    bounds_attr = ' bounds="0:1"' if bounds else ""
    offset = 8192
    xml = (f'<?xml version="1.0"?><xisf><Image geometry="{size}:{size}:1" '
           f'sampleFormat="Float32"{bounds_attr}{comp} location="attachment:{offset}:{len(raw)}">'
           f'{kw}</Image></xisf>').encode()
    with open(path, "wb") as f:
        f.write(b"XISF0100")
        f.write(len(xml).to_bytes(4, "little"))
        f.write(b"\0" * 4)
        f.write(xml)
        f.write(b"\0" * (offset - 16 - len(xml)))
        f.write(raw)


class Field:
    """One fixed star field; every frame drawn from it shows the same stars."""

    def __init__(self, seed=1, n_stars=40, size=SIZE):
        self.rng = np.random.default_rng(seed)
        self.size = size
        self.yy, self.xx = np.mgrid[:size, :size]
        self.stars = [(self.rng.uniform(25, size - 25), self.rng.uniform(25, size - 25),
                       self.rng.uniform(800, 6000)) for _ in range(n_stars)]

    def light(self, sky=150.0, pedestal=PEDESTAL, trail=False, sigma=1.6):
        img = np.full((self.size, self.size), pedestal + sky, dtype=np.float64)
        for x, y, a in self.stars:
            img += a * np.exp(-((self.xx - x) ** 2 + (self.yy - y) ** 2) / (2 * sigma ** 2))
        if trail:
            d = np.abs(self.yy - (0.3 * self.xx + 60)) / np.hypot(1, 0.3)
            img += 60 * np.exp(-d ** 2 / (2 * 1.5 ** 2))
        img += self.rng.normal(0, 12, img.shape)
        return np.clip(img, 0, 65535).astype(np.uint16)

    def flat(self, level=20000.0, pedestal=PEDESTAL):
        c = self.size / 2
        vignette = 1 - 0.2 * ((self.xx - c) ** 2 + (self.yy - c) ** 2) / c ** 2
        img = pedestal + level * vignette + self.rng.normal(0, 40, (self.size, self.size))
        return img.astype(np.uint16)


def header(**cards):
    h = fits.Header()
    for k, v in cards.items():
        h[k.replace("_", "-")] = v
    return h


def write_fits(path: Path, data, **cards):
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(data, header=header(**cards)).writeto(path, overwrite=True)
    return path


def make_project(root: Path, sessions=("night 1", "night 2"), panels=("Panel 1",),
                 frames=5, with_masters=True, trail_index=2):
    """<root>/darks + <session>/<panel>/H/LIGHT and <session>/flats/H/FLAT, at gain 100."""
    field = Field()
    if with_masters:
        darks = root / "darks"
        darks.mkdir(parents=True, exist_ok=True)
        write_xisf(darks / "masterBias_BIN-1_GAIN-100.xisf", np.full((SIZE, SIZE), PEDESTAL),
                   {"GAIN": 100, "XBINNING": 1})
        write_xisf(darks / "masterDark_BIN-1_EXPOSURE-30.00s.xisf",
                   np.full((SIZE, SIZE), PEDESTAL + 10), {"GAIN": 100})
    for s, session in enumerate(sessions):
        for panel in panels:
            for i in range(frames):
                write_fits(root / session / panel / "H" / "LIGHT" / f"L_{panel[-1]}_{i:03d}.fits",
                           field.light(trail=(i == trail_index)),
                           FILTER="H", EXPTIME=30.0, GAIN=100, XBINNING=1, CCD_TEMP=-10.0,
                           FOCALLEN=400.0, XPIXSZ=3.76,
                           DATE_OBS=f"2024-10-{11 + s:02d}T21:{10 + 3 * i:02d}:00")
        for i in range(3):
            write_fits(root / session / "flats" / "H" / "FLAT" / f"flat_{i}.fits", field.flat(),
                       FILTER="H", EXPTIME=1.0, GAIN=100, XBINNING=1, IMAGETYP="FLAT")
    return root
