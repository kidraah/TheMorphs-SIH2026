"""The problem statement's configuration, end to end, on synthetic data.

Three MTL heads, 30-min steps out to 6 h, realistic base rates two orders
of magnitude apart, with bootstrap confidence intervals attached.

Run before you have any real data:  .venv/bin/python scripts/demo_mtl.py
"""
import numpy as np

from nowcast_eval import EvalConfig, attach_confidence_intervals, evaluate_multi

RNG = np.random.default_rng(3)
LEAD_MIN = [30 * i for i in range(1, 13)]        # 30 min -> 6 h
N, H, W = 60, 48, 48

# Base rates as they actually are. Cloudbursts are ~100x rarer than storms,
# which is the whole reason the CI machinery exists.
HAZARDS = {
    "thunderstorm": dict(rate=2e-2, skill=0.88, threshold=0.5),
    "cloudburst":   dict(rate=2e-4, skill=0.70, threshold=0.6),
    "flash_flood":  dict(rate=8e-4, skill=0.75, threshold=0.5),
}


def synth(rate, skill):
    obs = (RNG.random((N, 12, H, W)) < rate).astype(float)
    decay = np.linspace(skill, skill * 0.25, 12)[None, :, None, None]
    pred = np.where(obs > 0,
                    RNG.uniform(0, 1, obs.shape) * decay + (1 - decay),
                    RNG.uniform(0, 1, obs.shape) * (1 - decay) * 0.9)
    return np.clip(pred, 0, 1), obs


def main():
    preds, obss, cfgs = {}, {}, {}
    for name, spec in HAZARDS.items():
        preds[name], obss[name] = synth(spec["rate"], spec["skill"])
        cfgs[name] = EvalConfig(name=name, grid_km=4.0,
                                headline_threshold=spec["threshold"])

    result = evaluate_multi(preds, obss, cfgs, lead_minutes=LEAD_MIN)

    # Gap 4: intervals, so nobody reads a cloudburst CSI as if it were solid.
    for name in result.names:
        attach_confidence_intervals(result[name], preds[name], obss[name],
                                    cfgs[name], n_boot=500)

    print(result.summary_table().split("--- thunderstorm")[0])

    print("=" * 78)
    print("POOLED CSI WITH 95% INTERVALS")
    print("=" * 78)
    for name in result.names:
        ci = result[name].pooled["ci"]
        print(f"\n{name}  ({ci['n_events']} observed events, "
              f"{ci['n_samples']} cases)")
        for m in ("csi", "pod", "far"):
            print(f"    {m.upper():>4}  {ci['point'][m]:.3f} "
                  f"[{ci['lo'][m]:.3f}, {ci['hi'][m]:.3f}]")
        for w in ci["warnings"]:
            print(f"    !!  {w}")


if __name__ == "__main__":
    main()
