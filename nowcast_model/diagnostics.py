"""Attention-range instrumentation, for the full-vs-windowed decision.

We kept full spatial attention on the argument that long-range reasoning
matters -- relating a moisture plume ~200 km upwind to the terrain below.
That argument is testable, and the test decides whether to switch to windowed
attention and take a ~48x compute saving.

The measurement has to be BUILT IN rather than bolted on afterwards, because
it needs a trained model and the intermediate attention weights, and neither
survives the end of a run.

What is measured
----------------
    mass vs distance   how much attention weight lands within r tokens
    effective range    the radius containing 50% / 90% of the mass
    entropy            how diffuse each distribution is, in nats, against
                       log(N) for uniform

How to read it
--------------
One token is 16 km on the 4 km grid at patch 4. Our stated 200 km
justification is 12.5 tokens. If r90 comes back at a handful of tokens, the
model is effectively local and full attention is buying nothing -- switch to
a window that comfortably exceeds r90 and take the compute back.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

TOKEN_KM_DEFAULT = 16.0


@dataclass
class AttentionRange:
    radii_tokens: np.ndarray
    mass_within: np.ndarray        # cumulative attention mass within r
    r50_tokens: float
    r90_tokens: float
    entropy_nats: float
    uniform_entropy_nats: float
    n_tokens: int
    token_km: float

    @property
    def r50_km(self) -> float:
        return self.r50_tokens * self.token_km

    @property
    def r90_km(self) -> float:
        return self.r90_tokens * self.token_km

    @property
    def diffuseness(self) -> float:
        """1.0 means attention is uniform (learning nothing spatial)."""
        return self.entropy_nats / max(self.uniform_entropy_nats, 1e-9)

    def report(self, needed_km: float = 200.0) -> str:
        L = [f"attention range over {self.n_tokens} tokens "
             f"({self.token_km:.0f} km each)"]
        L.append(f"  r50 {self.r50_tokens:5.1f} tokens = {self.r50_km:6.0f} km")
        L.append(f"  r90 {self.r90_tokens:5.1f} tokens = {self.r90_km:6.0f} km")
        L.append(f"  entropy {self.entropy_nats:.2f} nats of "
                 f"{self.uniform_entropy_nats:.2f} uniform "
                 f"(diffuseness {self.diffuseness:.2f})")
        if self.r90_km < needed_km * 0.5:
            L.append(f"  -> EFFECTIVELY LOCAL. 90% of attention lands inside "
                     f"{self.r90_km:.0f} km against a stated need of "
                     f"{needed_km:.0f} km. A window of "
                     f"{2 * int(np.ceil(self.r90_tokens)) or 2} tokens would "
                     f"capture it; full attention is buying range the model is "
                     f"not using.")
        elif self.r90_km < needed_km:
            L.append(f"  -> borderline: r90 {self.r90_km:.0f} km vs stated "
                     f"{needed_km:.0f} km. A window sized on r90 plus margin "
                     f"is probably safe; measure again after more training.")
        else:
            L.append(f"  -> GENUINELY LONG-RANGE. r90 reaches {self.r90_km:.0f} km, "
                     f"beyond the {needed_km:.0f} km justification. Full "
                     f"attention is doing work a small window would lose.")
        if self.diffuseness > 0.95:
            L.append("  !! attention is near-uniform -- the model may not have "
                     "learned spatial structure yet. Re-measure after training.")
        return "\n".join(L)


def _token_positions(hp: int, wp: int) -> np.ndarray:
    r, c = np.meshgrid(np.arange(hp), np.arange(wp), indexing="ij")
    return np.stack([r.ravel(), c.ravel()], axis=1).astype(np.float64)


def attention_range(weights, hp: int, wp: int,
                    token_km: float = TOKEN_KM_DEFAULT,
                    max_radius: int | None = None) -> AttentionRange:
    """Attention mass as a function of distance.

    `weights` is (N, N) or (B, N, N): row i is the distribution token i
    attends over. Rows must sum to 1.
    """
    w = np.asarray(weights.detach().cpu() if torch.is_tensor(weights) else weights,
                   dtype=np.float64)
    if w.ndim == 3:
        w = w.mean(axis=0)
    n = w.shape[0]
    if n != hp * wp:
        raise ValueError(f"weights are {n}x{n} but hp*wp = {hp*wp}")

    pos = _token_positions(hp, wp)
    d = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1))

    rmax = max_radius or int(np.ceil(d.max()))
    radii = np.arange(0, rmax + 1)
    mass = np.array([w[d <= r].sum() / n for r in radii])
    mass = np.clip(mass / max(mass[-1], 1e-12), 0, 1)

    r50 = float(np.interp(0.5, mass, radii))
    r90 = float(np.interp(0.9, mass, radii))

    ent = float(-(w * np.log(np.clip(w, 1e-12, None))).sum(axis=1).mean())
    return AttentionRange(radii, mass, r50, r90, ent, float(np.log(n)), n, token_km)


@torch.no_grad()
def measure_attention_range(model, x, station_coords=None, context=None,
                            layer: int = -1, token_km: float = TOKEN_KM_DEFAULT):
    """Run one diagnostic forward and measure the spatial attention range.

    Use a TILE, not the full grid: this materialises the NxN matrix, which at
    49k tokens would be 2.4e9 floats.
    """
    model.eval()
    model(x, station_coords, context=context, need_weights=True)
    blocks = model.backbone.blocks
    w = getattr(blocks[layer], "last_space_weights", None)
    if w is None:
        raise RuntimeError("no spatial attention weights captured -- the forward "
                           "pass must run with need_weights=True")
    hp = x.shape[-2] // model.backbone.patch
    wp = x.shape[-1] // model.backbone.patch
    return attention_range(w, hp, wp, token_km)
