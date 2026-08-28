"""INSAT-3D/3DR L1B ingestion onto the common 4 km grid.

Read with satpy's `insat3d_img_l1b_h5` reader, resample with pyresample.
satpy 0.60 exposes VIS/SWIR/WV/MIR/TIR1/TIR2 and confirms the resolutions
this project was built around: VIS 1000 m, TIR1/TIR2 4000 m, WV 8000 m.
That asymmetry is why channels are matched individually rather than
uniformly resized.

Sanity checks come FIRST
------------------------
A reader that silently drifts from MOSDAC's current product format does not
raise -- it returns plausible-looking arrays in the wrong units, the wrong
calibration, or flipped geometry. `check_physics` is the cheap guard: real
TIR-1 brightness temperatures live in roughly 190-320 K, and water vapour is
narrow and always cold. Anything outside that is a format problem to solve
today, not in week six.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from .grids import india_area

# Native resolutions, straight from the satpy reader spec.
INSAT_RESOLUTION_M = {"VIS": 1000, "SWIR": 1000, "MIR": 4000,
                      "TIR1": 4000, "TIR2": 4000, "WV": 8000}

# Physically plausible brightness-temperature ranges (kelvin).
# TIR-1 is a window channel: it sees the warm surface and cold cloud tops.
# WV is an absorption channel: it sees only the upper troposphere, so it is
# always cold and spans a much narrower range -- the same signature we
# confirmed in SEVIR's ir069 (-61.9 to -34.0 degC).
EXPECTED_BT_K = {
    "TIR1": (190.0, 320.0),
    "TIR2": (190.0, 320.0),
    "MIR": (190.0, 340.0),
    "WV": (190.0, 275.0),
}
BT_CHANNELS = tuple(EXPECTED_BT_K)


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str

    def __str__(self) -> str:
        return f"[{'PASS' if self.passed else 'FAIL'}] {self.name}: {self.detail}"


@dataclass
class ScanCheck:
    results: list = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    def add(self, name, passed, detail):
        self.results.append(CheckResult(name, passed, detail))

    def report(self) -> str:
        head = "INSAT scan sanity checks: " + ("ALL PASS" if self.passed else "FAILURES")
        return "\n".join([head, ""] + [f"  {r}" for r in self.results])

    def raise_if_failed(self):
        if not self.passed:
            bad = [r for r in self.results if not r.passed]
            raise ValueError("INSAT scan failed sanity checks:\n" +
                             "\n".join(f"  {r}" for r in bad))


def check_physics(arrays: dict, min_valid_frac: float = 0.05) -> ScanCheck:
    """Validate decoded brightness temperatures before anything downstream.

    `arrays` maps channel name -> 2-D array in kelvin.
    """
    chk = ScanCheck()
    for name, arr in arrays.items():
        a = np.asarray(arr, dtype=np.float64)
        finite = np.isfinite(a)
        frac = float(finite.mean())
        chk.add(f"{name} coverage", frac >= min_valid_frac,
                f"{100 * frac:.1f}% finite pixels")
        if not finite.any():
            chk.add(f"{name} range", False, "no finite pixels to check")
            continue

        lo, hi = np.nanmin(a), np.nanmax(a)
        if name in EXPECTED_BT_K:
            elo, ehi = EXPECTED_BT_K[name]
            ok = lo >= elo and hi <= ehi
            chk.add(f"{name} range", ok,
                    f"{lo:.1f}-{hi:.1f} K (expected {elo:.0f}-{ehi:.0f} K)"
                    + ("" if ok else "  <-- wrong units, calibration, or format drift"))

    # The window/absorption contrast: WV must be colder and narrower than TIR1.
    if "TIR1" in arrays and "WV" in arrays:
        t, w = np.asarray(arrays["TIR1"]), np.asarray(arrays["WV"])
        if np.isfinite(t).any() and np.isfinite(w).any():
            tspan = float(np.nanmax(t) - np.nanmin(t))
            wspan = float(np.nanmax(w) - np.nanmin(w))
            chk.add("WV narrower than TIR1", wspan < tspan,
                    f"WV span {wspan:.1f} K vs TIR1 span {tspan:.1f} K")
            chk.add("WV colder than TIR1", np.nanmax(w) < np.nanmax(t),
                    f"WV max {np.nanmax(w):.1f} K vs TIR1 max {np.nanmax(t):.1f} K")
    return chk


def read_scan(path, channels: Sequence[str] = ("TIR1", "WV"),
              calibration: str = "brightness_temperature"):
    """Open one INSAT L1B file with satpy. Returns a satpy Scene."""
    from satpy import Scene

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"no INSAT file at {p}")
    scn = Scene(filenames=[str(p)], reader="insat3d_img_l1b_h5")

    available = set(scn.available_dataset_names())
    missing = [c for c in channels if c not in available]
    if missing:
        raise KeyError(
            f"reader exposes {sorted(available)} but {missing} were requested. "
            f"If the product format has drifted, this is where it shows up.")
    scn.load(list(channels), calibration=calibration)
    return scn


def resample_to_grid(scene, channels: Sequence[str] = ("TIR1", "WV"),
                     resampler: str = "nearest",
                     radius_of_influence: float = 12000.0) -> dict:
    """Resample loaded channels onto the common India 4 km grid.

    Nearest by default with a 12 km search radius -- generous enough to fill
    the 4 km grid from the 8 km water-vapour channel without inventing
    structure, which is the same honesty as the nearest-upsampling used on
    the SEVIR side.
    """
    resampled = scene.resample(india_area(), resampler=resampler,
                               radius_of_influence=radius_of_influence)
    return {c: np.asarray(resampled[c].values, dtype=np.float32) for c in channels}


def ingest_scan(path, channels: Sequence[str] = ("TIR1", "WV"),
                strict: bool = True) -> tuple[dict, ScanCheck]:
    """Read, sanity-check at native resolution, then resample.

    Checks run BEFORE resampling: resampling a mis-decoded field produces a
    smooth, plausible-looking array and makes the original problem harder to
    see.
    """
    scn = read_scan(path, channels)
    native = {c: np.asarray(scn[c].values, dtype=np.float64) for c in channels}
    chk = check_physics(native)
    if strict:
        chk.raise_if_failed()
    return resample_to_grid(scn, channels), chk


def fetch_scan(*_args, **_kw):
    """Not implemented -- deliberately.

    MOSDAC serves INSAT products through an order/download portal whose
    endpoint and auth flow are not stable enough to hard-code blind. Guessing
    a URL here would produce a client that fails confusingly against the real
    service.

    Download one scan by hand from https://mosdac.gov.in (credentials in
    .env), then point `ingest_scan` at the local file. Once the actual
    request shape is confirmed against the live portal, implement it here
    using nowcast_data.credentials.get("MOSDAC_USERNAME"/"MOSDAC_PASSWORD").
    """
    raise NotImplementedError(
        "MOSDAC fetch is not implemented -- download a scan manually from "
        "https://mosdac.gov.in and pass the local path to ingest_scan(). "
        "See the docstring for why this is not guessed.")
