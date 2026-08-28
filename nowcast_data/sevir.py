"""SEVIR loader, targeted at INSAT-native resolution and cadence.

SEVIR (Storm EVent ImageRy, MIT Lincoln Laboratory) is GOES-16 + NEXRAD
over CONUS: ~20k four-hour storm events on a 384x384 km domain, 5-minute
frames. It is the pretraining set, used while MOSDAC and NCMRWF account
approvals are pending. It is not the target domain.

Design decisions, each of which costs something and is deliberate
-----------------------------------------------------------------
1. Every channel is degraded to its INSAT counterpart's resolution before
   being placed on the analysis grid -- water vapour to 8 km, thermal
   infrared to 4 km -- rather than uniformly resized. See grid.py. This
   throws away real SEVIR detail on purpose: detail the operational feed
   cannot supply is detail the model must not learn to depend on.

2. Frames are subsampled to the INSAT scan cadence (30 min), discarding
   five of every six. A backbone trained on 5-minute motion learns storm
   evolution at a timescale it will never see operationally.

3. Samples carry their storm DAY, and `day_groups()` hands it to the
   harness's block bootstrap. Windows from one synoptic setup are not
   independent draws; without this the validation intervals come out too
   narrow, invisibly.

Known limitation: the 4-hour horizon
------------------------------------
A SEVIR event is 4 hours long. Spend 1 h on context and 3 h of targets
remain -- so the 2-6 h claim CANNOT be pretrained end to end here. The far
half of that window has to come from IMDAA/INSAT fine-tuning. Better known
now than in week six.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from .grid import match_sensor_resolution, subsample_time

SEVIR_CADENCE_MIN = 5.0
SEVIR_FRAMES = 49          # 4 hours at 5 min, inclusive
SEVIR_DOMAIN_KM = 384.0


@dataclass(frozen=True)
class ChannelSpec:
    """One SEVIR image type and the INSAT channel it stands in for."""
    img_type: str
    native_km: float        # SEVIR's own resolution
    sensor_km: float        # what INSAT-3D/3DR actually resolves
    scale: float = 1.0      # raw -> physical units
    offset: float = 0.0
    verified: bool = True    # False -> decode needs checking against the docs


# INSAT-3D/3DR counterparts. TIR is 4 km, WV is 8 km -- the asymmetry that
# makes uniform downsampling wrong.
DEFAULT_CHANNELS: dict[str, ChannelSpec] = {
    # thermal IR 10.7 um -> INSAT TIR1. Drives the CTT drop rate.
    "ir107": ChannelSpec("ir107", native_km=2.0, sensor_km=4.0, scale=0.01),
    # water vapour 6.9 um -> INSAT WV. 8 km native: the IWV cornerstone
    # genuinely has no 4 km structure, and must not appear to.
    "ir069": ChannelSpec("ir069", native_km=2.0, sensor_km=8.0, scale=0.01),
    # visible -> INSAT VIS. Daytime only; a night-time model cannot use it.
    "vis": ChannelSpec("vis", native_km=0.5, sensor_km=1.0, scale=1e-4),
}

# The label channel. NEXRAD-derived vertically integrated liquid at 1 km.
#
# TWO WARNINGS, both load-bearing:
#
#   * `scale` here is a placeholder. SEVIR stores VIL with a NON-LINEAR
#     uint8 encoding, so a single multiplier does NOT recover kg/m^2.
#     Verify the decoding against the SEVIR documentation before defining
#     any label threshold on it -- the label definition determines what the
#     model learns, and a wrong decode silently mislabels every sample.
#
#   * India has no NEXRAD. Nothing at 1 km will be available operationally;
#     the Indian truth source (INSAT QPE, IMERG at ~11 km, or IMD's gridded
#     gauge product at ~25 km) is far coarser. If labels come in at 25 km, a
#     20-30 km^2 cloudburst is below one label pixel, and label resolution --
#     not input resolution -- becomes the binding constraint on "hyper-local".
VIL_CHANNEL = ChannelSpec("vil", native_km=1.0, sensor_km=4.0,
                          scale=1.0, verified=False)


@dataclass
class SEVIRConfig:
    data_root: Path
    catalog: Path
    inputs: Sequence[ChannelSpec] = field(
        default_factory=lambda: [DEFAULT_CHANNELS["ir107"], DEFAULT_CHANNELS["ir069"]])
    target: ChannelSpec = VIL_CHANNEL

    target_km: float = 4.0          # INSAT TIR native; the analysis grid
    cadence_min: float = 30.0       # INSAT-3D full disk
    context_frames: int = 2         # 1 h of history at 30 min
    horizon_frames: int = 6         # 3 h ahead -- SEVIR's ceiling, see module docs
    max_pct_missing: float = 5.0

    def __post_init__(self):
        self.data_root = Path(self.data_root)
        self.catalog = Path(self.catalog)
        need = (self.context_frames + self.horizon_frames) * (
            self.cadence_min / SEVIR_CADENCE_MIN)
        if need > SEVIR_FRAMES:
            raise ValueError(
                f"context+horizon needs {need:.0f} SEVIR frames but an event has "
                f"only {SEVIR_FRAMES} (4 hours). Reduce horizon_frames: a 6-hour "
                f"lead time cannot be pretrained on SEVIR.")

    @property
    def grid_size(self) -> int:
        return int(round(SEVIR_DOMAIN_KM / self.target_km))

    @property
    def lead_minutes(self) -> list[float]:
        return [self.cadence_min * (i + 1) for i in range(self.horizon_frames)]


@dataclass
class SEVIRSample:
    x: np.ndarray            # (C, context_frames, H, W)
    y: np.ndarray            # (horizon_frames, H, W)
    event_id: str
    day: str                 # YYYY-MM-DD -- the bootstrap blocking unit
    time_utc: str
    channels: list[str]


class SEVIRCatalog:
    """The CATALOG.csv index, filtered to events usable for training."""

    REQUIRED = ("id", "file_name", "file_index", "img_type")

    def __init__(self, path: Path, max_pct_missing: float = 5.0):
        import pandas as pd
        self.path = Path(path)
        df = pd.read_csv(self.path, low_memory=False)
        missing = [c for c in self.REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(f"{self.path} is missing columns {missing}; "
                             f"found {list(df.columns)[:12]}")
        if "pct_missing" in df.columns:
            df = df[df["pct_missing"].fillna(0) <= max_pct_missing]
        self.df = df

    def events_with(self, img_types: Sequence[str]) -> list[str]:
        """Event ids present for EVERY requested channel.

        An event missing one channel is dropped rather than zero-filled --
        a zero-filled water vapour field is not missing data to the model,
        it is a confident statement that the air is dry.
        """
        sub = self.df[self.df["img_type"].isin(list(img_types))]
        counts = sub.groupby("id")["img_type"].nunique()
        return sorted(counts[counts == len(set(img_types))].index.tolist())

    def rows_for(self, event_id: str, img_type: str):
        m = (self.df["id"] == event_id) & (self.df["img_type"] == img_type)
        rows = self.df[m]
        if rows.empty:
            raise KeyError(f"no catalog row for event {event_id!r} / {img_type!r}")
        return rows.iloc[0]

    def day_of(self, event_id: str) -> str:
        rows = self.df[self.df["id"] == event_id]
        if "time_utc" in rows.columns and not rows.empty:
            return str(rows.iloc[0]["time_utc"])[:10]
        return "unknown"


class SEVIRLoader:
    """Reads events, matches them to INSAT resolution/cadence, windows them."""

    def __init__(self, config: SEVIRConfig):
        self.cfg = config
        self.catalog = SEVIRCatalog(config.catalog, config.max_pct_missing)
        if not config.target.verified:
            warnings.warn(
                f"channel {config.target.img_type!r} has an unverified decode "
                f"(scale={config.target.scale}). SEVIR VIL uses a non-linear "
                f"uint8 encoding; confirm it before defining label thresholds.",
                stacklevel=2)

    # --- raw access --------------------------------------------------------
    def _read_raw(self, event_id: str, spec: ChannelSpec) -> np.ndarray:
        """(frames, H, W) in physical units, still at SEVIR resolution."""
        import h5py
        row = self.catalog.rows_for(event_id, spec.img_type)
        path = self.cfg.data_root / str(row["file_name"])
        if not path.exists():
            raise FileNotFoundError(f"{path} (event {event_id}, {spec.img_type})")

        with h5py.File(path, "r") as fh:
            if spec.img_type not in fh:
                raise KeyError(f"{path} has no dataset {spec.img_type!r}; "
                               f"found {list(fh.keys())}")
            arr = np.asarray(fh[spec.img_type][int(row["file_index"])])

        # SEVIR stores (H, W, frames); everything downstream wants frames first.
        if arr.ndim != 3:
            raise ValueError(f"expected 3 dims for {spec.img_type}, got {arr.shape}")
        arr = np.moveaxis(arr, -1, 0)
        return arr.astype(np.float64) * spec.scale + spec.offset

    def load_channel(self, event_id: str, spec: ChannelSpec) -> np.ndarray:
        """Physical units, INSAT resolution, INSAT cadence. (T, H, W)."""
        raw = self._read_raw(event_id, spec)
        timed = subsample_time(raw, SEVIR_CADENCE_MIN, self.cfg.cadence_min, axis=0)
        return match_sensor_resolution(timed, spec.native_km, spec.sensor_km,
                                       self.cfg.target_km)

    # --- windowing ---------------------------------------------------------
    def sample(self, event_id: str, start: int = 0) -> SEVIRSample:
        cfg = self.cfg
        need = start + cfg.context_frames + cfg.horizon_frames

        chans = [self.load_channel(event_id, s) for s in cfg.inputs]
        tgt = self.load_channel(event_id, cfg.target)

        avail = min([c.shape[0] for c in chans] + [tgt.shape[0]])
        if need > avail:
            raise ValueError(
                f"event {event_id}: need {need} frames from index {start}, "
                f"only {avail} available at {cfg.cadence_min:g}-min cadence")

        ctx = slice(start, start + cfg.context_frames)
        hor = slice(start + cfg.context_frames, need)
        return SEVIRSample(
            x=np.stack([c[ctx] for c in chans]),
            y=tgt[hor],
            event_id=event_id,
            day=self.catalog.day_of(event_id),
            time_utc=str(self.catalog.rows_for(event_id, cfg.inputs[0].img_type)
                         .get("time_utc", "")),
            channels=[s.img_type for s in cfg.inputs],
        )

    def event_ids(self) -> list[str]:
        types = [s.img_type for s in self.cfg.inputs] + [self.cfg.target.img_type]
        return self.catalog.events_with(types)

    def iter_samples(self, event_ids: Sequence[str] | None = None,
                     stride: int = 0) -> Iterator[SEVIRSample]:
        """Yield windows. stride=0 gives one window per event.

        Overlapping windows from one event share almost all their data, so
        anything above stride 0 makes the day-blocking in `day_groups`
        essential rather than merely advisable.
        """
        cfg = self.cfg
        for eid in (event_ids if event_ids is not None else self.event_ids()):
            if stride <= 0:
                yield self.sample(eid, 0)
                continue
            n = int(round(SEVIR_FRAMES * SEVIR_CADENCE_MIN / cfg.cadence_min))
            last = n - cfg.context_frames - cfg.horizon_frames
            for s in range(0, max(last, 0) + 1, stride):
                yield self.sample(eid, s)

    # --- the bridge to the harness ----------------------------------------
    def day_groups(self, samples: Sequence[SEVIRSample]) -> np.ndarray:
        """Storm day per sample, for `bootstrap_ci(..., groups=...)`.

        Pass this to the harness or the validation intervals will be too
        narrow: windows cut from one storm day share a synoptic setup and
        are not independent draws.
        """
        return np.array([s.day for s in samples])


def stack_samples(samples: Sequence[SEVIRSample]) -> tuple[np.ndarray, np.ndarray]:
    """(N, C, T, H, W) inputs and (N, T_out, H, W) targets."""
    if not samples:
        raise ValueError("no samples")
    return (np.stack([s.x for s in samples]), np.stack([s.y for s in samples]))
