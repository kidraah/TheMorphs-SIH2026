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


def _topological_order(fg: FlowGrid) -> np.ndarray:
    """Cells ordered headwaters-first (Kahn's algorithm on the D8 tree).

    Iterative on purpose. A recursive walk overflows the stack on a real
    tile -- an India tile is ~10^8 cells and the Ganga's main stem is a
    single chain hundreds of thousands of cells long.
    """
    n = fg.n
    down = fg.downstream
    indeg = np.zeros(n, dtype=np.int64)
    has_down = down >= 0
    np.add.at(indeg, down[has_down], 1)

    order = np.empty(n, dtype=np.int64)
    queue = np.flatnonzero(indeg == 0)
    filled = 0
    while queue.size:
        order[filled:filled + queue.size] = queue
        filled += queue.size
        nxt = down[queue]
        nxt = nxt[nxt >= 0]
        if nxt.size == 0:
            break
        np.subtract.at(indeg, nxt, 1)
        queue = np.unique(nxt[indeg[nxt] == 0])
    if filled != n:
        # A cycle means the direction raster is not a tree -- almost always a
        # wrong convention or a bad nodata fill, so say which.
        raise ValueError(
            f"flow direction contains a cycle: {n - filled} of {n} cells are "
            f"unreachable headwaters-first. A D8 field must be acyclic; check "
            f"the direction encoding and nodata handling.")
    return order[:filled]


def flow_accumulate(fg: FlowGrid, weight: np.ndarray | None = None,
                    order: np.ndarray | None = None) -> np.ndarray:
    """Accumulate `weight` downstream. Default weight 1 => cell counts.

    This is the workhorse: with weight = cell area it reproduces `upa`, and
    with weight = runoff volume it produces the routed flood volume, which
    is the whole point of having it.
    """
    order = _topological_order(fg) if order is None else order
    w = (np.ones(fg.n, dtype=np.float64) if weight is None
         else np.asarray(weight, dtype=np.float64).ravel().copy())
    acc = w.copy()
    down = fg.downstream
    for i in order:                      # headwaters first: each cell final
        j = down[i]                      # when reached
        if j >= 0:
            acc[j] += acc[i]
    return acc.reshape(fg.shape)


def verify_against_upa(fg: FlowGrid, upa_km2: np.ndarray, cell_area_km2,
                       tol: float = 0.02) -> dict:
    """Gate: does accumulation from `dir` reproduce MERIT's own `upa`?

    Agreement is the only cheap evidence that the direction convention above
    is read correctly. A transposed or reversed encoding routes water
    confidently in the wrong direction and everything downstream of it looks
    normal, so this runs before any flood number is believed.
    """
    area = (np.full(fg.shape, float(cell_area_km2))
            if np.isscalar(cell_area_km2) else np.asarray(cell_area_km2))
    mine = flow_accumulate(fg, area)
    theirs = np.asarray(upa_km2, dtype=np.float64)
    valid = np.isfinite(theirs) & (theirs > 0) & np.isfinite(mine)
    if not valid.any():
        return {"passed": False, "reason": "no valid cells to compare"}
    rel = np.abs(mine[valid] - theirs[valid]) / theirs[valid]
    frac = float(np.mean(rel <= tol))
    return {
        "passed": bool(frac >= 0.95),
        "fraction_within_tol": frac,
        "median_rel_error": float(np.median(rel)),
        "p99_rel_error": float(np.percentile(rel, 99)),
        "tol": tol,
        "n_compared": int(valid.sum()),
        "reason": ("" if frac >= 0.95 else
                   "accumulation does not reproduce upa -- the D8 convention "
                   "in flow.py is probably wrong (transposed axes or reversed "
                   "rotation). Do not use any routed number until this passes."),
    }
