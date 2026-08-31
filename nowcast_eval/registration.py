"""Post-hoc registration: is the model's forecast displaced from the labels?

This is a REQUIRED GATE, not a diagnostic. The pre-ingest alignment gate
returned INCONCLUSIVE (LIMITATIONS 15), so ingest proceeded on the fallback
path, and this is the check that pays for that decision.

Why it is required rather than nice-to-have
-------------------------------------------
A learned registration offset costs INTERPRETATION, not skill. The model can
compensate internally and score perfectly on every metric in this harness
while every map it draws is shifted by 15 km. POD, FAR, CSI, SEDI, FSS and
the reliability diagram are all computed cell-against-cell, so a forecast
displaced by the same amount as its labels scores exactly as well as one
that is not. Nothing else in the repo can see this.

Why it is EASIER than the pre-ingest gate
-----------------------------------------
The pre-ingest gate matched two sparse binary masks -- thresholded cold cloud
against thresholded rain -- and binary IoU on sparse masks saturates, which
is why 2/3 of tiles had no interior maximum. Here both fields are dense and
continuous: a predicted probability field against a label field. Normalised
cross-correlation on continuous fields has an interior maximum wherever there
is any structure at all.

PER YEAR, NOT POOLED
--------------------
Registration can DRIFT -- satellite repositioning, a product version change,
a reprocessing campaign. A pooled number averages a drift away and reports a
small offset that is wrong in every individual year. The pre-ingest gate
would have caught drift by construction; this one only catches it if it is
asked per year. `by_year` is therefore the primary output and the pooled
figure is reported beneath it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MAX_OFFSET_PX = 8
PASS_BAR_PX = 1.0
DRIFT_BAR_PX = 1.0          # spread across years that counts as drift


@dataclass
class RegistrationResult:
    dy: float = float("nan")
    dx: float = float("nan")
    peak: float = float("nan")
    zero: float = float("nan")
    n_samples: int = 0
    censored: bool = False
    # Peak height above the REST of the surface, in sigma. Not peak/zero:
    # a perfectly registered field has its peak AT zero offset, so the ratio
    # is exactly 1.000 and a peak/zero test rules out the one case that
    # should pass. Same shape as SEDI being undefined for a perfect forecast.
    z_score: float = float("nan")

    @property
    def offset_px(self) -> float:
        return float(np.hypot(self.dy, self.dx))

    @property
    def identifiable(self) -> bool:
        return (not self.censored and np.isfinite(self.z_score)
                and self.z_score >= 3.0)


@dataclass
class RegistrationGate:
    by_year: dict = field(default_factory=dict)
    pooled: RegistrationResult = field(default_factory=RegistrationResult)
    grid_km: float = 4.0
    verdict: str = "INCONCLUSIVE"
    reasons: list = field(default_factory=list)

    @property
    def drift_px(self) -> float:
        vals = [r.offset_px for r in self.by_year.values() if r.identifiable]
        return float(np.max(vals) - np.min(vals)) if len(vals) > 1 else float("nan")

    def report(self) -> str:
        L = ["=" * 66, f"POST-HOC REGISTRATION GATE: {self.verdict}", "=" * 66,
             f"{'year':>8} {'dy':>7} {'dx':>7} {'offset km':>10} "
             f"{'sigma':>8} {'n':>8}"]
        for y in sorted(self.by_year):
            r = self.by_year[y]
            L.append(f"{y:>8} {r.dy:>+7.2f} {r.dx:>+7.2f} "
                     f"{r.offset_px * self.grid_km:>10.1f} "
                     f"{r.z_score:>8.1f} {r.n_samples:>8,}"
                     + ("" if r.identifiable else "   NOT IDENTIFIABLE"))
        p = self.pooled
        L += ["", f"{'POOLED':>8} {p.dy:>+7.2f} {p.dx:>+7.2f} "
                  f"{p.offset_px * self.grid_km:>10.1f}"]
        if np.isfinite(self.drift_px):
            L.append(f"  drift across years: {self.drift_px * self.grid_km:.1f} km "
                     f"(bar {DRIFT_BAR_PX * self.grid_km:.0f} km)")
        L.append("")
        for r in self.reasons:
            L.append(f"  {r}")
        return "\n".join(L)


def _ncc_surface(pred, obs, mask=None, max_offset: int = MAX_OFFSET_PX):
    """Normalised cross-correlation over integer shifts. Continuous fields."""
    p = np.asarray(pred, dtype=np.float64)
    o = np.asarray(obs, dtype=np.float64)
    valid = np.isfinite(p) & np.isfinite(o)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    if valid.sum() < 100:
        return None, 0
    pz = np.where(valid, p, 0.0)
    pz = pz - pz[valid].mean()
    n = 2 * max_offset + 1
    s = np.zeros((n, n))
    for i, dy in enumerate(range(-max_offset, max_offset + 1)):
        for j, dx in enumerate(range(-max_offset, max_offset + 1)):
            ов = np.roll(np.roll(o, dy, axis=-2), dx, axis=-1)
            vv = valid & np.isfinite(ов)
            if vv.sum() < 100:
                continue
            a = pz[vv]
            b = ов[vv] - ов[vv].mean()
            den = (a.std() * b.std()) or 1e-12
            s[i, j] = float((a * b).mean() / den)
    return s, int(valid.sum())


def measure(pred, obs, mask=None, max_offset: int = MAX_OFFSET_PX) -> RegistrationResult:
    """One (pred, obs) stack -> the displacement between them."""
    s, n = _ncc_surface(pred, obs, mask, max_offset)
    if s is None:
        return RegistrationResult(n_samples=n)
    k = int(np.argmax(s))
    iy, ix = divmod(k, s.shape[0])
    c = max_offset
    dy, dx = iy - c, ix - c
    others = np.delete(s.ravel(), k)
    z = (s[iy, ix] - others.mean()) / (others.std() or 1e-12)
    return RegistrationResult(dy=float(dy), dx=float(dx), peak=float(s[iy, ix]),
                              zero=float(s[c, c]), n_samples=n,
                              censored=(abs(dy) >= max_offset or abs(dx) >= max_offset),
                              z_score=float(z))


def run_gate(pred_by_year: dict, obs_by_year: dict, masks=None,
             grid_km: float = 4.0, max_offset: int = MAX_OFFSET_PX) -> RegistrationGate:
    """The gate. `pred_by_year` / `obs_by_year` map year -> (N, ..., H, W)."""
    g = RegistrationGate(grid_km=grid_km)
    if set(pred_by_year) != set(obs_by_year):
        raise ValueError("pred and obs must cover the same years")

    for y in sorted(pred_by_year):
        m = (masks or {}).get(y)
        g.by_year[y] = measure(pred_by_year[y], obs_by_year[y], m, max_offset)

    allp = np.concatenate([np.asarray(v).reshape(-1, *np.asarray(v).shape[-2:])
                           for v in pred_by_year.values()])
    allo = np.concatenate([np.asarray(v).reshape(-1, *np.asarray(v).shape[-2:])
                           for v in obs_by_year.values()])
    g.pooled = measure(allp, allo, None, max_offset)

    usable = [r for r in g.by_year.values() if r.identifiable]
    if len(usable) < 2:
        g.verdict = "INCONCLUSIVE"
        g.reasons.append(
            f"only {len(usable)} of {len(g.by_year)} years are identifiable. "
            f"A pooled figure alone cannot show drift, which is the failure "
            f"the pre-ingest gate would have caught and this one is standing "
            f"in for.")
        return g

    off = g.pooled.offset_px
    drift = g.drift_px
    ok = True
    if not g.pooled.identifiable:
        g.verdict = "INCONCLUSIVE"
        g.reasons.append("the pooled surface has no interior maximum.")
        return g
    if off > PASS_BAR_PX:
        ok = False
        g.reasons.append(
            f"FAIL: forecasts sit {off:.2f} px ({off * grid_km:.1f} km) from "
            f"their labels. Every metric in the harness is cell-against-cell "
            f"and cannot see this -- the maps are shifted, the scores are not.")
    else:
        g.reasons.append(f"offset {off:.2f} px ({off * grid_km:.1f} km) is within "
                         f"{PASS_BAR_PX:.1f} px.")
    if np.isfinite(drift) and drift > DRIFT_BAR_PX:
        ok = False
        g.reasons.append(
            f"FAIL: registration DRIFTS {drift:.2f} px ({drift * grid_km:.1f} km) "
            f"across years. The pooled figure averages that away and is "
            f"misleading in every individual year -- check for a satellite "
            f"repositioning or a product version change in the archive.")
    elif np.isfinite(drift):
        g.reasons.append(f"drift across years {drift:.2f} px, within "
                         f"{DRIFT_BAR_PX:.1f} px.")
    g.verdict = "PASS" if ok else "FAIL"
    return g
