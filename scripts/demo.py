"""End-to-end demo on synthetic storms -- no real data required.

Run this to see what a scorecard looks like before any of the MOSDAC or
IMDAA accounts come through. It also demonstrates the point of the whole
harness: a plausible-looking model is scored against baselines, and you
find out immediately whether it beat 'assume nothing changes'.

    .venv/bin/python scripts/demo.py
"""
import numpy as np

from nowcast_eval import EvalConfig, evaluate
from nowcast_eval.baselines import climatology, persistence
from nowcast_eval.plots import fss_by_scale, reliability_diagram, skill_vs_lead_time

RNG = np.random.default_rng(42)
N, LEAD, H, W = 40, 6, 96, 96
LEAD_MIN = [30, 60, 120, 240, 360, 420]


def synthetic_storms():
    """Blobs that drift and decay. Stands in for convective cells."""
    obs = np.zeros((N, LEAD, H, W))
    yy, xx = np.mgrid[0:H, 0:W]
    for n in range(N):
        for _ in range(RNG.integers(1, 4)):
            y0, x0 = RNG.uniform(15, H - 15), RNG.uniform(15, W - 15)
            vy, vx = RNG.normal(0, 2.5), RNG.normal(0, 2.5)
            r = RNG.uniform(3, 7)
            for li in range(LEAD):
                d2 = (yy - (y0 + vy * li)) ** 2 + (xx - (x0 + vx * li)) ** 2
                obs[n, li][d2 < r ** 2] = 1.0
    return obs


def toy_model(obs):
    """A deliberately imperfect forecast: right storms, slightly wrong place,
    degrading with lead time. Exactly the failure mode FSS is built to see."""
    pred = np.zeros_like(obs)
    for li in range(LEAD):
        shift = int(round(1.2 * li))
        blur = np.roll(np.roll(obs[:, li], shift, axis=1), shift, axis=2)
        conf = 0.92 - 0.11 * li
        noise = RNG.random(blur.shape)
        pred[:, li] = np.clip(blur * conf + noise * (0.03 + 0.02 * li), 0, 1)
    return pred


def main():
    obs = synthetic_storms()
    pred = toy_model(obs)

    cfg = EvalConfig(name="demo", grid_km=4.0, headline_threshold=0.5)

    model = evaluate(pred, obs, cfg, lead_minutes=LEAD_MIN, meta={"run": "demo"})
    persist = evaluate(np.clip(persistence(obs[:, 0], LEAD), 0, 1), obs, cfg,
                       lead_minutes=LEAD_MIN)
    clim = evaluate(climatology(float(obs.mean()), obs.shape), obs, cfg,
                    lead_minutes=LEAD_MIN)

    for name, r in (("MODEL", model), ("PERSISTENCE", persist), ("CLIMATOLOGY", clim)):
        print(f"\n=== {name} " + "=" * (62 - len(name)))
        print(r.summary_table())

    model.to_json("artifacts/demo_result.json")
    skill_vs_lead_time(model, "artifacts/skill_vs_lead.png",
                       models={"persistence": persist, "climatology": clim})
    fss_by_scale(model, "artifacts/fss_by_scale.png")
    reliability_diagram(model, "artifacts/reliability.png")
    print("\nwrote artifacts/{demo_result.json,skill_vs_lead.png,"
          "fss_by_scale.png,reliability.png}")


if __name__ == "__main__":
    main()
