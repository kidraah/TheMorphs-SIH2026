"""Train/val/test splitting by contiguous calendar blocks.

Why not shuffled events, and not even shuffled days
---------------------------------------------------
Shuffling events puts windows from the same storm on both sides of the
split, so validation partly measures memorisation. Blocking by day fixes
that but not enough: consecutive days share a synoptic setup, so 2018-07-20
in train and 2018-07-21 in val are still meteorologically correlated.

So the split is by CALENDAR ORDER -- earliest ~70% train, next 15% val, last
15% test -- with a buffer discarded either side of each boundary to break
the correlation across it. This also matches how the model will actually be
used: trained on the past, run on the future.

Effective sample size
---------------------
`SplitReport` reports three counts per split, and they differ by orders of
magnitude:

    events    -- what you have
    days      -- what day-blocking gives you
    episodes  -- maximal runs of consecutive storm days: what the error
                 bars actually rest on

On the real SEVIR store the test split holds ~2,065 events, ~75 days and
~9 episodes. Quoting a test CSI as though it had 2,065 independent samples
overstates the evidence by more than two orders of magnitude.

`episodes` is deliberately conservative: a 7-day run may contain two or
three distinct synoptic systems, so true independence sits somewhere between
the episode and day counts. Blocking the bootstrap on episode is the
cautious end of that range, which is the right end to be on.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd


def count_episodes(days: Sequence) -> int:
    """Maximal runs of consecutive calendar days."""
    d = pd.Series(sorted(pd.unique(pd.to_datetime(list(days)).normalize())))
    if d.empty:
        return 0
    return 1 + int((d.diff().dt.days.dropna() > 1).sum())


def episode_ids(days: Sequence) -> np.ndarray:
    """Episode label per input day -- the `groups` argument for the bootstrap."""
    ds = pd.to_datetime(list(days)).normalize()
    uniq = pd.Series(sorted(pd.unique(ds)))
    new_run = (uniq.diff().dt.days.fillna(999) > 1).cumsum()
    lookup = dict(zip(uniq, new_run))
    return np.array([f"ep{lookup[d]:04d}" for d in ds])


@dataclass
class Split:
    name: str
    event_ids: list[str]
    days: list
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None

    @property
    def n_events(self) -> int:
        return len(self.event_ids)

    @property
    def n_days(self) -> int:
        return len(set(self.days))

    @property
    def n_episodes(self) -> int:
        return count_episodes(self.days)

    @property
    def overstatement(self) -> float:
        """How far the raw event count overstates independent evidence."""
        return self.n_events / max(self.n_episodes, 1)

    def groups(self) -> np.ndarray:
        return episode_ids(self.days)


@dataclass
class SplitReport:
    splits: dict[str, Split] = field(default_factory=dict)
    n_discarded: int = 0
    buffer_days: int = 7

    def __getitem__(self, k: str) -> Split:
        return self.splits[k]

    def summary(self) -> str:
        lines = [f"contiguous calendar split, {self.buffer_days}-day buffer "
                 f"either side of each boundary", ""]
        hdr = (f"{'split':>7} {'events':>8} {'days':>6} {'episodes':>9} "
               f"{'overstate':>10}  range")
        lines += [hdr, "-" * len(hdr)]
        for s in self.splits.values():
            rng = (f"{s.start:%Y-%m-%d} .. {s.end:%Y-%m-%d}"
                   if s.start is not None else "")
            lines.append(f"{s.name:>7} {s.n_events:>8,} {s.n_days:>6} "
                         f"{s.n_episodes:>9} {s.overstatement:>9.0f}x  {rng}")
        lines.append(f"{'buffer':>7} {self.n_discarded:>8,}   discarded")
        lines.append("")
        lines.append("EFFECTIVE SAMPLE SIZE is the episode column, not events.")
        thin = [s.name for s in self.splits.values() if s.n_episodes < 20]
        if thin:
            lines.append(f"!!  {', '.join(thin)} below ~20 episodes: bootstrap "
                         f"intervals there are wide AND the bootstrap itself is "
                         f"only marginally reliable. Report them, do not hide them.")
        return "\n".join(lines)


def split_by_time(
    event_ids: Sequence[str],
    days: Sequence,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    buffer_days: int = 7,
) -> SplitReport:
    """Contiguous calendar split with a discarded buffer at each boundary.

    Fractions are of DISTINCT DAYS, not of events, so a few unusually busy
    days cannot drag the boundary.
    """
    if not 0 < train_frac < 1 or not 0 < val_frac < 1:
        raise ValueError("fractions must be in (0, 1)")
    if train_frac + val_frac >= 1.0:
        raise ValueError(f"train+val must leave room for test, got "
                         f"{train_frac} + {val_frac}")
    if len(event_ids) != len(days):
        raise ValueError(f"{len(event_ids)} ids vs {len(days)} days")

    df = pd.DataFrame({"id": list(event_ids),
                       "day": pd.to_datetime(list(days)).normalize()})
    uniq = pd.Series(sorted(df["day"].unique()))
    if len(uniq) < 3:
        raise ValueError(f"need at least 3 distinct days, got {len(uniq)}")

    buf = pd.Timedelta(days=buffer_days)
    q_tr = uniq.iloc[int(len(uniq) * train_frac)]
    q_va = uniq.iloc[int(len(uniq) * (train_frac + val_frac))]

    masks = {
        "train": df["day"] <= q_tr - buf,
        "val": (df["day"] >= q_tr + buf) & (df["day"] <= q_va - buf),
        "test": df["day"] >= q_va + buf,
    }

    report = SplitReport(buffer_days=buffer_days)
    used = 0
    for name, m in masks.items():
        part = df[m]
        used += len(part)
        report.splits[name] = Split(
            name=name,
            event_ids=part["id"].tolist(),
            days=part["day"].tolist(),
            start=part["day"].min() if len(part) else None,
            end=part["day"].max() if len(part) else None,
        )
    report.n_discarded = len(df) - used

    empty = [n for n, s in report.splits.items() if s.n_events == 0]
    if empty:
        raise ValueError(
            f"split(s) {empty} are empty -- the {buffer_days}-day buffer has "
            f"consumed them. Shorten the buffer or widen the date range.")
    return report


def split_sevir(loader, **kw) -> SplitReport:
    """Convenience: build the split straight from a SEVIRLoader."""
    ids = loader.event_ids()
    days = [loader.catalog.day_of(i) for i in ids]
    return split_by_time(ids, days, **kw)
