"""Torch dataset over SEVIR, with an explicit missing-data policy.

Two choices are made here rather than left implicit.

1. NaN policy. 7% of real SEVIR pixels are missing (the int16 sentinel in
   the satellite channels, byte 255 in VIL). A network cannot consume NaN,
   so SOMETHING happens to those pixels -- the only question is whether it
   is stated. `NaNPolicy` states it.

2. Target resolution. VIL is coarsened to ~12 km before thresholding,
   matching the resolution of the Indian fine-tuning labels (IMERG 0.1deg
   ~ 11 km) rather than SEVIR's native 1 km. Pretraining a decoder on 1 km
   structure teaches it detail the fine-tuning target physically cannot
   contain, which fine-tuning would then have to unlearn. Measured cost:
   the >5 kg/m^2 population is 98.7% retained, >20 is 75%, >40 is 46%.
   The extreme head is percentile-defined and therefore self-relative, so
   the shift is consistent rather than lossy in the way it looks.

   12 km specifically, not 10: 384 / 12 = 32 exactly and 12 / 4 = 3, so the
   arithmetic stays integral onto the 96-cell analysis grid. 10 divides
   neither.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class NaNPolicy(str, Enum):
    """What happens to missing pixels. Stated, not implied."""

    # Fill inputs with 0 (== the mean after normalisation), append a validity
    # channel so the network can tell "missing" from "average", and exclude
    # missing target cells from the loss. The honest default.
    MASK = "mask"

    # Fill with 0 and say nothing. The network reads missing as average and
    # is scored on invented targets. Cheaper, and wrong at 7% missing.
    FILL_ZERO = "fill_zero"

    # Fill with the channel mean. Same blindness as FILL_ZERO, minus the
    # normalisation artefact.
    FILL_MEAN = "fill_mean"


@dataclass
class ChannelStats:
    mean: list[float]
    std: list[float]

    def save(self, path) -> None:
        Path(path).write_text(json.dumps({"mean": self.mean, "std": self.std}, indent=2))

    @classmethod
    def load(cls, path) -> "ChannelStats":
        d = json.loads(Path(path).read_text())
        return cls(d["mean"], d["std"])


@dataclass
class TargetConfig:
    """How VIL becomes per-head binary labels, for PRETRAINING only.

    These thresholds are SEVIR-specific. The Indian heads are redefined per
    docs/LABELS.md -- IMERG-derived rain rate and extreme rain, station-based
    cloudburst -- and none of these numbers carry over.
    """
    label_km: float = 12.0          # match IMERG, not SEVIR's 1 km
    rain_kgm2: float = 1.0          # "raining at all"
    extreme_kgm2: float = 25.6      # ~99.9th pct of the 12 km field, measured
    n_pseudo_stations: int = 48


@dataclass
class DatasetConfig:
    nan_policy: NaNPolicy = NaNPolicy.MASK
    targets: TargetConfig = field(default_factory=TargetConfig)
    stats: ChannelStats | None = None
    add_validity_channel: bool = True
    seed: int = 0

    def __post_init__(self):
        self.nan_policy = NaNPolicy(self.nan_policy)


class SEVIRDataset(Dataset):
    """Yields (x, targets, masks, coords) for the multi-task model."""

    def __init__(self, loader, event_ids: Sequence[str],
                 config: DatasetConfig | None = None):
        self.loader = loader
        self.ids = list(event_ids)
        self.cfg = config or DatasetConfig()
        self.grid = loader.cfg.grid_size

    def __len__(self) -> int:
        return len(self.ids)

    # --- inputs ------------------------------------------------------------
    def _prepare_inputs(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Normalise, handle NaN, return (x, validity)."""
        valid = np.isfinite(x)
        st = self.cfg.stats
        if st is not None:
            m = np.asarray(st.mean).reshape(-1, 1, 1, 1)
            s = np.asarray(st.std).reshape(-1, 1, 1, 1)
            x = (x - m) / np.maximum(s, 1e-6)

        pol = self.cfg.nan_policy
        if pol is NaNPolicy.FILL_MEAN:
            fill = np.zeros(x.shape[0]) if st is not None else np.nanmean(
                x.reshape(x.shape[0], -1), axis=1)
            x = np.where(valid, x, fill.reshape(-1, 1, 1, 1))
        else:
            # MASK and FILL_ZERO both fill with 0; they differ in whether the
            # network and the loss are told about it.
            x = np.where(valid, x, 0.0)
        return x, valid

    # --- targets -----------------------------------------------------------
    def _targets(self, y: np.ndarray, rng) -> tuple[dict, dict, np.ndarray]:
        t = self.cfg.targets
        # The loader has ALREADY coarsened the target to its channel's
        # sensor_km. Coarsening again here would silently shrink the grid --
        # it did, from 96 to 24, before this check existed. So validate
        # instead of repeating the operation.
        if self.loader.cfg.target.sensor_km != t.label_km:
            raise ValueError(
                f"target channel is coarsened to "
                f"{self.loader.cfg.target.sensor_km} km but TargetConfig asks "
                f"for {t.label_km} km. Set them equal -- the coarsening happens "
                f"once, in the ChannelSpec.")
        valid = np.isfinite(y)
        yf = np.where(valid, y, 0.0)

        targets = {
            "rain_rate": (yf > t.rain_kgm2).astype(np.float32),
            "extreme_rain": (yf > t.extreme_kgm2).astype(np.float32),
        }
        masks = {k: valid.astype(np.float32) for k in targets}

        # Pseudo-stations. SEVIR has no gauge network, so the point head
        # cannot learn the real station relationship here -- only its
        # mechanism (sampling, gradient path) and a sensible initialisation.
        # The actual cloudburst relationship can only come from IMD station
        # data at fine-tune time.
        h, w = yf.shape[-2:]
        xs = rng.integers(0, w, t.n_pseudo_stations)
        ys = rng.integers(0, h, t.n_pseudo_stations)
        coords = np.stack([(2 * xs + 1) / w - 1, (2 * ys + 1) / h - 1], axis=-1)
        targets["cloudburst"] = (yf[:, ys, xs] > t.extreme_kgm2).astype(np.float32)
        masks["cloudburst"] = valid[:, ys, xs].astype(np.float32)
        return targets, masks, coords.astype(np.float32)

    def __getitem__(self, i):
        eid = self.ids[i]
        rng = np.random.default_rng(self.cfg.seed + i)
        s = self.loader.sample(eid)

        x, valid = self._prepare_inputs(s.x)
        if self.cfg.nan_policy is NaNPolicy.MASK and self.cfg.add_validity_channel:
            # one extra channel per input channel, so "missing" is
            # distinguishable from "average" rather than being silently
            # indistinguishable from it
            x = np.concatenate([x, valid.astype(np.float32)], axis=0)

        targets, masks, coords = self._targets(s.y, rng)
        return (
            torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
            {k: torch.from_numpy(v) for k, v in targets.items()},
            {k: torch.from_numpy(v) for k, v in masks.items()},
            torch.from_numpy(coords),
            eid,
        )

    @property
    def in_channels(self) -> int:
        n = len(self.loader.cfg.inputs)
        if self.cfg.nan_policy is NaNPolicy.MASK and self.cfg.add_validity_channel:
            n *= 2
        return n


def collate(batch):
    xs, tg, mk, co, ids = zip(*batch)
    return (torch.stack(xs),
            {k: torch.stack([t[k] for t in tg]) for k in tg[0]},
            {k: torch.stack([m[k] for m in mk]) for k in mk[0]},
            torch.stack(co),
            list(ids))


def compute_channel_stats(loader, event_ids: Sequence[str], n_sample: int = 200,
                          seed: int = 0) -> ChannelStats:
    """Per-channel mean/std over a sample of the TRAIN split only.

    Train only: statistics computed over val or test leak information about
    them into training, which is a quiet way to flatter your own results.
    NaN-aware, since 7% of pixels are missing.
    """
    rng = np.random.default_rng(seed)
    ids = list(event_ids)
    pick = [ids[i] for i in rng.choice(len(ids), min(n_sample, len(ids)), replace=False)]

    sums, sqs, counts = None, None, None
    for eid in pick:
        x = loader.sample(eid).x                       # (C, T, H, W)
        flat = x.reshape(x.shape[0], -1)
        v = np.isfinite(flat)
        s = np.nansum(np.where(v, flat, 0.0), axis=1)
        q = np.nansum(np.where(v, flat ** 2, 0.0), axis=1)
        c = v.sum(axis=1)
        sums = s if sums is None else sums + s
        sqs = q if sqs is None else sqs + q
        counts = c if counts is None else counts + c

    mean = sums / np.maximum(counts, 1)
    var = np.maximum(sqs / np.maximum(counts, 1) - mean ** 2, 1e-12)
    return ChannelStats(mean.tolist(), np.sqrt(var).tolist())
