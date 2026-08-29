"""Sub-basin delineation and aggregation of grid fields onto basins.

Why sub-basins and not a fixed target size
------------------------------------------
Flash floods are basin-scale, and the basin that matters is the one whose
outlet the water arrives at. Chopping the domain into equal-area blocks
would be a grid with extra steps. So the decomposition here is the standard
one: cut the channel network at confluences, and give each reach the area
that drains into it. Basin size then comes from the topology of the river
network rather than from a number chosen for convenience, which is also why
basin areas vary by orders of magnitude -- see `aggregate`, and see the
weighting problem in nowcast_eval.

Everything is a label raster plus a property table, not vector polygons.
The label raster IS the polygon set at the analysis resolution, it needs no
geometry library, and it aggregates with a bincount rather than a spatial
join.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .flow import D8_OFFSETS, FlowGrid, _topological_order, flow_accumulate

# Diagonal D8 steps are longer than cardinal ones. Ignoring this
# underestimates channel length by up to 41%, and Kirpich takes L^0.77, so
# the error lands straight in the arrival time.
_DIAGONAL = {2, 8, 32, 128}


@dataclass
class Basins:
    labels: np.ndarray             # (H, W) int, -1 = unassigned
    ids: np.ndarray                # (B,) sorted unique labels
    area_km2: np.ndarray           # (B,)
    channel_length_m: np.ndarray   # (B,)
    channel_slope: np.ndarray      # (B,) dimensionless, nan if no elevation
    outlet_index: np.ndarray       # (B,) flat index of each reach outlet
    mean_elev_above_outlet_m: np.ndarray | None = None   # (B,) basin relief

    @property
    def n(self) -> int:
        return len(self.ids)

    def __len__(self) -> int:
        return self.n


def delineate(fg: FlowGrid, upa_km2: np.ndarray, channel_km2: float = 25.0,
              order: np.ndarray | None = None) -> np.ndarray:
    """Label every cell by the reach catchment it belongs to.

    `channel_km2` sets where the channel network starts. It is a judgment
    call, not a constant: raising it makes fewer, larger basins. 25 km^2 is
    a common flash-flood working value and is recorded in the result.
    """
    order = _topological_order(fg) if order is None else order
    down = fg.downstream
    channel = (np.asarray(upa_km2, dtype=np.float64) >= channel_km2).ravel()

    # channel inflows per cell -> sources (0) and confluences (>=2) start reaches
    inflow = np.zeros(fg.n, dtype=np.int32)
    src = np.flatnonzero(channel & (down >= 0))
    np.add.at(inflow, down[src], 1)

    reach = np.full(fg.n, -1, dtype=np.int64)
    next_id = 0
    for i in order:                                   # headwaters first
        if not channel[i]:
            continue
        up_here = inflow[i]
        if up_here == 1:
            # inherit from the single upstream channel cell, already labelled
            j = _sole_channel_parent(fg, channel, i)
            reach[i] = reach[j] if j >= 0 and reach[j] >= 0 else -1
        if reach[i] < 0:                              # source or confluence
            reach[i] = next_id
            next_id += 1

    # hillslopes take the label of the first channel cell they reach:
    # process downstream-first so the target is already resolved
    labels = np.full(fg.n, -1, dtype=np.int64)
    for i in order[::-1]:
        if channel[i]:
            labels[i] = reach[i]
        else:
            j = down[i]
            labels[i] = labels[j] if j >= 0 else -1
    return labels.reshape(fg.shape)


def _sole_channel_parent(fg: FlowGrid, channel: np.ndarray, i: int) -> int:
    """The one channel cell flowing into i. Only called when inflow == 1."""
    h, w = fg.shape
    r, c = divmod(int(i), w)
    for code, (dr, dc) in D8_OFFSETS.items():
        rr, cc = r - dr, c - dc                       # a neighbour pointing AT i
        if 0 <= rr < h and 0 <= cc < w:
            j = rr * w + cc
            if fg.direction.ravel()[j] == code and channel[j]:
                return j
    return -1


def properties(fg: FlowGrid, labels: np.ndarray, cell_area_km2,
               cell_size_m: float, upa_km2: np.ndarray,
               elevation_m: np.ndarray | None = None,
               channel_km2: float = 25.0) -> Basins:
    """Area, channel length and channel slope per basin.

    `elevation_m` is required for slope, and slope is required by Kirpich.
    MERIT `elv` is the matching layer; without it `channel_slope` is nan and
    every downstream arrival time is nan rather than a plausible default. A
    default slope would produce confident, wrong warning times.
    """
    lab = np.asarray(labels).ravel()
    ids = np.unique(lab[lab >= 0])
    if ids.size == 0:
        return Basins(labels, ids, *(np.zeros(0) for _ in range(3)),
                      np.zeros(0, dtype=np.int64), np.zeros(0))
    index = np.searchsorted(ids, lab.clip(0))
    index = np.where(lab >= 0, index, -1)
    keep = index >= 0

    area_cell = (np.full(fg.n, float(cell_area_km2)) if np.isscalar(cell_area_km2)
                 else np.asarray(cell_area_km2, dtype=np.float64).ravel())
    area = np.bincount(index[keep], weights=area_cell[keep], minlength=ids.size)

    # channel length: sum of step lengths over the reach's channel cells
    channel = (np.asarray(upa_km2, dtype=np.float64).ravel() >= channel_km2)
    step = np.where(np.isin(fg.direction.ravel(), list(_DIAGONAL)),
                    cell_size_m * np.sqrt(2.0), cell_size_m)
    on_ch = keep & channel
    length = np.bincount(index[on_ch], weights=step[on_ch], minlength=ids.size)

    # outlet = the channel cell of the reach with the largest upstream area
    upa_flat = np.asarray(upa_km2, dtype=np.float64).ravel()
    outlet = np.full(ids.size, -1, dtype=np.int64)
    ch_idx = np.flatnonzero(on_ch)
    ordering = np.lexsort((upa_flat[ch_idx], index[ch_idx]))
    ch_sorted = ch_idx[ordering]
    grp = index[ch_sorted]
    last = np.r_[grp[1:] != grp[:-1], True]           # last of each group = max upa
    outlet[grp[last]] = ch_sorted[last]

    slope = np.full(ids.size, np.nan)
    relief = None
    if elevation_m is not None:
        elev = np.asarray(elevation_m, dtype=np.float64).ravel()
        hi = np.full(ids.size, -np.inf)
        np.maximum.at(hi, index[on_ch], elev[on_ch])
        # The drop is measured to where the channel LEAVES the reach, not to
        # the outlet cell itself. Measuring within the reach gives every
        # single-cell reach a slope of exactly zero -- and therefore an
        # undefined Kirpich time -- which at the channel threshold is a large
        # and entirely artificial share of the headwater basins.
        exit_cell = np.where(outlet >= 0, fg.downstream[outlet.clip(0)], -1)
        exit_cell = np.where(exit_cell >= 0, exit_cell, outlet)
        lo = np.where(exit_cell >= 0, elev[exit_cell.clip(0)], np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            slope = (hi - lo) / length
        slope = np.where((length > 0) & np.isfinite(hi) & np.isfinite(lo),
                         slope, np.nan)
        # Giandotti uses basin RELIEF, not channel slope: the mean elevation
        # of the whole basin above its outlet. Having a method that fails on
        # a different quantity is what makes the ensemble spread meaningful
        # rather than three views of the same error.
        cnt = np.bincount(index[keep], minlength=ids.size)
        tot = np.bincount(index[keep], weights=elev[keep], minlength=ids.size)
        with np.errstate(invalid="ignore", divide="ignore"):
            relief = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan) - lo
    return Basins(np.asarray(labels), ids, area, length, slope, outlet, relief)


def aggregate(field: np.ndarray, basins: Basins, how: str = "mean",
              cell_area_km2=None) -> np.ndarray:
    """Grid field -> one value per basin.

    "mean"  area-weighted mean (e.g. basin-average runoff depth, mm)
    "sum"   total (e.g. runoff VOLUME, if the field is already a volume)
    "max"   the extreme cell (e.g. peak local intensity)

    Area weighting is not decoration: cells are equal-area on the analysis
    grid but a plain mean over an irregular basin is only correct because of
    that -- pass `cell_area_km2` when it is not true.
    """
    f = np.asarray(field, dtype=np.float64).ravel()
    lab = basins.labels.ravel()
    keep = (lab >= 0) & np.isfinite(f)
    idx = np.searchsorted(basins.ids, lab.clip(0))
    idx = np.where(keep, idx, -1)
    ok = idx >= 0
    nb = basins.n
    if how == "max":
        out = np.full(nb, -np.inf)
        np.maximum.at(out, idx[ok], f[ok])
        return np.where(np.isfinite(out), out, np.nan)
    w = (np.ones(f.size) if cell_area_km2 is None
         else (np.full(f.size, float(cell_area_km2)) if np.isscalar(cell_area_km2)
               else np.asarray(cell_area_km2, dtype=np.float64).ravel()))
    tot = np.bincount(idx[ok], weights=(f * w)[ok], minlength=nb)
    if how == "sum":
        return tot
    if how == "mean":
        den = np.bincount(idx[ok], weights=w[ok], minlength=nb)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, tot / den, np.nan)
    raise ValueError(f"how must be 'mean', 'sum' or 'max', got {how!r}")
