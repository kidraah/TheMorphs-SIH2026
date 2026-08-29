"""D8 flow routing: direction -> accumulation -> sub-basins -> channel geometry.

Direction encoding
------------------
MERIT Hydro's `dir` layer uses the ESRI D8 convention, powers of two
clockwise from east:

        32  64 128
        16   X   1
         8   4   2

with 0 = river mouth, -1 = inland depression, -9999 = ocean.

THAT CONVENTION IS ASSERTED, NOT VERIFIED. Nothing in this repo has read a
real `dir` tile yet -- they are not downloaded (see docs/FLOOD_DATA.md). A
transposed or counter-clockwise reading would still route, still produce
basins, and still return plausible numbers; it would simply route water the
wrong way. So `verify_against_upa()` exists and the flood track is gated on
it: recompute accumulation from `dir` and check it reproduces MERIT's own
`upa`, which is on disk. Two independently derived layers agreeing is the
only cheap proof the convention is right.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# code -> (drow, dcol). Row 0 is NORTH (standard raster orientation), so
# north is drow = -1.
D8_OFFSETS = {
    1:   (0, 1),     # east
    2:   (1, 1),     # south-east
    4:   (1, 0),     # south
    8:   (1, -1),    # south-west
    16:  (0, -1),    # west
    32:  (-1, -1),   # north-west
    64:  (-1, 0),    # north
    128: (-1, 1),    # north-east
}
MOUTH, DEPRESSION, OCEAN = 0, -1, -9999
TERMINAL = (MOUTH, DEPRESSION, OCEAN)


@dataclass(frozen=True)
class FlowGrid:
    """A D8 network. `downstream` is a flat index, or -1 for a terminal cell."""
    direction: np.ndarray          # (H, W) int, D8 codes
    downstream: np.ndarray         # (H*W,) int64, -1 at terminals
    shape: tuple[int, int]

    @property
    def n(self) -> int:
        return self.shape[0] * self.shape[1]

    @property
    def is_terminal(self) -> np.ndarray:
        return self.downstream < 0


def build_flow_grid(direction: np.ndarray) -> FlowGrid:
    """Precompute the downstream index of every cell, once.

    Off-grid neighbours become terminals: a basin that drains out of the
    domain is truncated, not wrapped. Wrapping would connect Gujarat to
    Assam, and the resulting basin would look entirely ordinary.
    """
    d = np.asarray(direction)
    h, w = d.shape
    rows, cols = np.divmod(np.arange(h * w), w)
    down = np.full(h * w, -1, dtype=np.int64)

    for code, (dr, dc) in D8_OFFSETS.items():
        m = (d.ravel() == code)
        if not m.any():
            continue
        r, c = rows[m] + dr, cols[m] + dc
        inside = (r >= 0) & (r < h) & (c >= 0) & (c < w)
        idx = np.flatnonzero(m)
        down[idx[inside]] = r[inside] * w + c[inside]
    return FlowGrid(direction=d, downstream=down, shape=(h, w))


def topological_levels(fg: FlowGrid) -> list[np.ndarray]:
    """Kahn waves: level k contains cells whose upstream is all in < k.

    Returned as WAVES rather than a flat order because accumulation then
    vectorises. A per-cell Python loop is O(N) interpreter steps, and a
    single MERIT tile is 6000x6000 = 36 million cells -- correct, and far
    too slow to ever run on real data, which is its own kind of untested.

    Iterative on purpose either way: a recursive walk overflows the stack,
    since the Ganga's main stem is one chain hundreds of thousands of cells
    long.
    """
    n = fg.n
    down = fg.downstream
    indeg = np.zeros(n, dtype=np.int64)
    has_down = down >= 0
    np.add.at(indeg, down[has_down], 1)

    waves, seen = [], 0
    frontier = np.flatnonzero(indeg == 0)
    while frontier.size:
        waves.append(frontier)
        seen += frontier.size
        nxt = down[frontier]
        nxt = nxt[nxt >= 0]
        if nxt.size == 0:
            break
        np.subtract.at(indeg, nxt, 1)
        frontier = np.unique(nxt[indeg[nxt] == 0])
    if seen != n:
        raise ValueError(
            f"flow direction contains a cycle: {n - seen} of {n} cells are "
            f"unreachable headwaters-first. A D8 field must be acyclic; check "
            f"the direction encoding and nodata handling.")
    return waves


def _topological_order(fg: FlowGrid) -> np.ndarray:
    """Flat headwaters-first order. Kept for callers that need a sequence."""
    return np.concatenate(topological_levels(fg))


def flow_accumulate(fg: FlowGrid, weight: np.ndarray | None = None,
                    levels: list | None = None) -> np.ndarray:
    """Accumulate `weight` downstream. Default weight 1 => cell counts.

    This is the workhorse: with weight = cell area it reproduces `upa`, and
    with weight = runoff volume it produces the routed flood volume, which
    is the whole point of having it.

    Vectorised over Kahn waves: one `np.add.at` per wave instead of one
    Python step per cell. A cell enters a wave only once every upstream
    contributor has been processed, so its value is final when its wave is
    pushed downstream -- the same invariant the per-cell loop relied on.
    """
    levels = topological_levels(fg) if levels is None else levels
    acc = (np.ones(fg.n, dtype=np.float64) if weight is None
           else np.asarray(weight, dtype=np.float64).ravel().astype(np.float64))
    acc = acc.copy()
    down = fg.downstream
    for wave in levels:
        j = down[wave]
        ok = j >= 0
        if ok.any():
            np.add.at(acc, j[ok], acc[wave[ok]])
    return acc.reshape(fg.shape)


def boundary_contaminated(fg: FlowGrid, levels: list | None = None) -> np.ndarray:
    """Cells whose catchment extends beyond this tile.

    MERIT's `upa` is computed on the global mosaic, so at a tile edge it
    counts area that is not in the tile. Comparing those cells against a
    tile-local accumulation would report a disagreement that is an artefact
    of the crop, not of the routing -- and it would look exactly like a
    wrong D8 convention, which is the thing the comparison exists to detect.
    """
    levels = topological_levels(fg) if levels is None else levels
    h, w = fg.shape
    flag = np.zeros(fg.n, dtype=bool)
    edge = np.zeros(fg.shape, dtype=bool)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    flag |= edge.ravel()
    down = fg.downstream
    for wave in levels:
        j = down[wave]
        ok = (j >= 0) & flag[wave]
        if ok.any():
            flag[j[ok]] = True
    return flag.reshape(fg.shape)


def verify_against_upa(fg: FlowGrid, upa_km2: np.ndarray, cell_area_km2,
                       tol: float = 0.02, exclude_boundary: bool = True) -> dict:
    """Gate: does accumulation from `dir` reproduce MERIT's own `upa`?

    Agreement is the only cheap evidence that the direction convention above
    is read correctly. A transposed or reversed encoding routes water
    confidently in the wrong direction and everything downstream of it looks
    normal, so this runs before any flood number is believed.
    """
    area = (np.full(fg.shape, float(cell_area_km2))
            if np.isscalar(cell_area_km2) else np.asarray(cell_area_km2))
    levels = topological_levels(fg)
    mine = flow_accumulate(fg, area, levels)
    theirs = np.asarray(upa_km2, dtype=np.float64)
    valid = np.isfinite(theirs) & (theirs > 0) & np.isfinite(mine)
    if exclude_boundary:
        valid &= ~boundary_contaminated(fg, levels)
    if not valid.any():
        return {"passed": False, "reason": "no valid cells to compare"}
    rel = np.abs(mine[valid] - theirs[valid]) / theirs[valid]
    frac = float(np.mean(rel <= tol))

    # Separate a ROUTING error from a UNITS error. If the two accumulations
    # differ by a constant factor, the ratio has near-zero spread and the
    # disagreement is the cell-area constant -- MERIT's own convention, not
    # a wrong D8 reading. A transposed or reversed convention sends water to
    # different cells, which shows up as a wide ratio distribution, not an
    # offset. Reporting only a mean error cannot tell these apart, and they
    # have opposite consequences.
    ratio = mine[valid] / theirs[valid]
    scale = float(np.median(ratio))
    spread = float(np.percentile(ratio, 84) - np.percentile(ratio, 16)) / 2
    rel_after_scale = np.abs(ratio / scale - 1.0)
    return {
        "scale_offset": scale,
        "scale_spread": spread,
        "median_rel_error_after_rescale": float(np.median(rel_after_scale)),
        "verdict": ("routing exact; residual is a cell-area constant"
                    if spread < 0.01 else
                    "ratio is DISPERSED -- water is going to different cells, "
                    "which is a routing error, not a units error"),
        "passed": bool(frac >= 0.95),
        "fraction_within_tol": frac,
        "median_rel_error": float(np.median(rel)),
        "p99_rel_error": float(np.percentile(rel, 99)),
        "tol": tol,
        "n_compared": int(valid.sum()),
        "boundary_excluded": bool(exclude_boundary),
        "reason": ("" if frac >= 0.95 else
                   "accumulation does not reproduce upa -- the D8 convention "
                   "in flow.py is probably wrong (transposed axes or reversed "
                   "rotation). Do not use any routed number until this passes."),
    }
