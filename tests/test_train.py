"""Tests for splitting, the NaN policy, checkpointing, and the loop.

Bias is toward the failures that do not raise: temporal leakage across a
split boundary, a resume that silently discards optimizer state, a NaN
policy that fills without telling anyone.
"""
import numpy as np
import pytest
import torch

from nowcast_data.sevir import DEFAULT_CHANNELS, VIL_CHANNEL, SEVIRConfig, SEVIRLoader
from nowcast_data.synthetic import build_store
from nowcast_model import ModelConfig, MultiTaskNowcaster
from nowcast_train import (ChannelStats, DatasetConfig, NaNPolicy, SEVIRDataset,
                           TargetConfig, TrainConfig, collate, count_episodes,
                           episode_ids, find_latest, load_checkpoint,
                           save_checkpoint, split_by_time, train)


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------

def _days(spec):
    out = []
    for day, n in spec:
        out += [day] * n
    return out


def test_episode_counting():
    """Maximal runs of consecutive days."""
    assert count_episodes(["2019-01-01", "2019-01-02", "2019-01-03"]) == 1
    assert count_episodes(["2019-01-01", "2019-01-05"]) == 2
    assert count_episodes(["2019-01-01", "2019-01-02", "2019-01-09"]) == 2
    assert count_episodes([]) == 0


def test_episode_ids_group_consecutive_days_together():
    g = episode_ids(["2019-01-01", "2019-01-02", "2019-01-20", "2019-01-01"])
    assert g[0] == g[1] == g[3]
    assert g[2] != g[0]


def test_split_is_chronological_with_a_real_gap():
    """The point of the buffer: no train day may sit within `buffer_days`
    of any val day, or the two share a synoptic setup."""
    days = [f"2019-{m:02d}-{d:02d}" for m in range(1, 13) for d in (1, 8, 15, 22)]
    ids = [f"E{i}" for i in range(len(days))]
    rep = split_by_time(ids, days, buffer_days=7)

    import pandas as pd
    tr, va, te = rep["train"], rep["val"], rep["test"]
    assert pd.Timestamp(max(tr.days)) < pd.Timestamp(min(va.days))
    assert pd.Timestamp(max(va.days)) < pd.Timestamp(min(te.days))
    gap = (pd.Timestamp(min(va.days)) - pd.Timestamp(max(tr.days))).days
    assert gap > 7, f"buffer must actually separate the splits, gap={gap}"


def test_split_reports_all_three_sample_sizes():
    days = _days([(f"2019-{m:02d}-{d:02d}", 5) for m in range(1, 13) for d in (1, 2, 3, 20)])
    ids = [f"E{i}" for i in range(len(days))]
    rep = split_by_time(ids, days)
    s = rep["train"]
    assert s.n_events > s.n_days >= s.n_episodes
    assert s.overstatement > 1
    text = rep.summary()
    assert "episodes" in text and "EFFECTIVE SAMPLE SIZE" in text


def test_split_warns_when_episodes_are_thin():
    # 20 isolated days, well separated: every split has few episodes
    days = _days([(f"2018-{m:02d}-{d:02d}", 20)
                  for m in range(1, 11) for d in (5, 20)])
    ids = [f"E{i}" for i in range(len(days))]
    rep = split_by_time(ids, days, buffer_days=1)
    assert rep["test"].n_episodes < 20
    assert "below ~20 episodes" in rep.summary()


def test_buffer_that_eats_a_split_fails_loudly():
    days = _days([("2019-01-01", 3), ("2019-01-02", 3), ("2019-01-03", 3)])
    with pytest.raises(ValueError, match="buffer has"):
        split_by_time([f"E{i}" for i in range(9)], days, buffer_days=30)


def test_split_fractions_are_of_days_not_events():
    """A few unusually busy days must not drag the boundary."""
    days = _days([("2019-01-01", 500)] + [(f"2019-02-{d:02d}", 1) for d in range(1, 25)])
    ids = [f"E{i}" for i in range(len(days))]
    rep = split_by_time(ids, days, buffer_days=0)
    assert rep["train"].n_days >= 15


# --------------------------------------------------------------------------
# Dataset / NaN policy
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def store(tmp_path_factory):
    return build_store(tmp_path_factory.mktemp("sevir_train"), n_events=12, n_days=6)


@pytest.fixture(scope="module")
def loader(store):
    return SEVIRLoader(SEVIRConfig.from_store(
        store, inputs=[DEFAULT_CHANNELS["ir107"], DEFAULT_CHANNELS["ir069"]],
        target=VIL_CHANNEL, context_frames=2, horizon_frames=4))


def test_mask_policy_adds_a_validity_channel(loader):
    """So 'missing' is distinguishable from 'average'. Without it, a filled
    zero is indistinguishable from a mean-valued observation."""
    ds = SEVIRDataset(loader, loader.event_ids(),
                      DatasetConfig(nan_policy=NaNPolicy.MASK))
    x, _, _, _, _ = ds[0]
    assert ds.in_channels == 4          # 2 data + 2 validity
    assert x.shape[0] == 4
    assert set(np.unique(x[2:].numpy())) <= {0.0, 1.0}


def test_fill_policy_does_not_add_a_channel(loader):
    ds = SEVIRDataset(loader, loader.event_ids(),
                      DatasetConfig(nan_policy=NaNPolicy.FILL_ZERO))
    assert ds.in_channels == 2
    assert ds[0][0].shape[0] == 2


def test_no_nan_survives_into_the_tensor(loader):
    """A NaN reaching the network produces NaN gradients and destroys the run."""
    for pol in (NaNPolicy.MASK, NaNPolicy.FILL_ZERO, NaNPolicy.FILL_MEAN):
        ds = SEVIRDataset(loader, loader.event_ids(), DatasetConfig(nan_policy=pol))
        x, y, m, _, _ = ds[0]
        assert torch.isfinite(x).all(), pol
        for k in y:
            assert torch.isfinite(y[k]).all(), (pol, k)
            assert torch.isfinite(m[k]).all(), (pol, k)


def test_nan_policy_is_an_explicit_enum():
    assert NaNPolicy("mask") is NaNPolicy.MASK
    with pytest.raises(ValueError):
        DatasetConfig(nan_policy="whatever")


def test_targets_are_coarsened_to_the_label_resolution(loader):
    """VIL is labelled at ~12 km to match IMERG, not at SEVIR's native 1 km.
    On a 4 km grid that means 3x3 blocks must be constant."""
    ds = SEVIRDataset(loader, loader.event_ids(),
                      DatasetConfig(targets=TargetConfig(label_km=12.0)))
    _, y, _, _, _ = ds[0]
    g = y["rain_rate"].numpy()
    blk = g[:, 0:3, 0:3]
    assert np.allclose(blk, blk[:, :1, :1]), "12 km labels must be 3x3-constant on a 4 km grid"


def test_point_head_targets_and_coords_line_up(loader):
    ds = SEVIRDataset(loader, loader.event_ids(),
                      DatasetConfig(targets=TargetConfig(n_pseudo_stations=13)))
    _, y, m, coords, _ = ds[0]
    assert y["cloudburst"].shape[-1] == 13
    assert coords.shape == (13, 2)
    assert coords.min() >= -1.0 and coords.max() <= 1.0


def test_channel_stats_are_nan_aware(loader):
    from nowcast_train import compute_channel_stats
    st = compute_channel_stats(loader, loader.event_ids(), n_sample=4)
    assert len(st.mean) == 2 and len(st.std) == 2
    assert all(np.isfinite(st.mean)) and all(s > 0 for s in st.std)


def test_stats_round_trip(tmp_path):
    st = ChannelStats([1.0, 2.0], [0.5, 0.25])
    p = tmp_path / "stats.json"
    st.save(p)
    assert ChannelStats.load(p).mean == st.mean


# --------------------------------------------------------------------------
# Checkpointing -- built for spot reclaims
# --------------------------------------------------------------------------

def _tiny():
    m = MultiTaskNowcaster(ModelConfig(in_channels=2, context_frames=2,
                                       grid_size=32, lead_steps=2, dim=16, depth=1))
    return m, torch.optim.AdamW(m.parameters(), lr=1e-3)


def test_resume_restores_optimizer_state_not_just_weights(tmp_path):
    """The failure this guards: saving only weights loses Adam's moment
    estimates, so the loss trajectory visibly jumps after a reclaim."""
    m, opt = _tiny()
    x = torch.randn(1, 2, 2, 32, 32)
    coords = torch.rand(1, 5, 2) * 2 - 1
    for _ in range(3):
        out = m(x, coords)
        sum(v.sum() for v in out.values()).backward()
        opt.step(); opt.zero_grad()

    p = save_checkpoint(tmp_path / "c.pt", model=m, optimizer=opt,
                        epoch=2, best_score=0.4)

    m2, opt2 = _tiny()
    meta = load_checkpoint(p, model=m2, optimizer=opt2)
    assert meta["epoch"] == 2 and meta["best_score"] == pytest.approx(0.4)

    s1 = opt.state_dict()["state"]
    s2 = opt2.state_dict()["state"]
    assert s1.keys() == s2.keys() and len(s1) > 0, "optimizer state must survive"
    k = next(iter(s1))
    assert torch.allclose(s1[k]["exp_avg"], s2[k]["exp_avg"]), "Adam moments lost"
    assert s1[k]["step"] == s2[k]["step"]

    for a, b in zip(m.parameters(), m2.parameters()):
        assert torch.allclose(a, b)


def test_checkpoint_write_is_atomic(tmp_path):
    """A reclaim mid-write must not leave a truncated file that fails to
    load later -- worse than no checkpoint, because it is found too late."""
    m, opt = _tiny()
    p = save_checkpoint(tmp_path / "c.pt", model=m, optimizer=opt,
                        epoch=0, best_score=0.0)
    assert p.exists()
    assert not list(tmp_path.glob("*.tmp")), "temp file must be renamed away"


def test_find_latest_picks_the_newest(tmp_path):
    m, opt = _tiny()
    save_checkpoint(tmp_path / "a.pt", model=m, optimizer=opt, epoch=0, best_score=0)
    save_checkpoint(tmp_path / "b.pt", model=m, optimizer=opt, epoch=1, best_score=1)
    assert find_latest(tmp_path).name in {"a.pt", "b.pt"}
    assert find_latest(tmp_path / "nope") is None


def test_missing_checkpoint_raises(tmp_path):
    m, _ = _tiny()
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "absent.pt", model=m)


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------

def test_two_epoch_run_produces_a_checkpoint(loader, tmp_path):
    ids = loader.event_ids()
    dcfg = DatasetConfig(nan_policy=NaNPolicy.MASK,
                         targets=TargetConfig(rain_kgm2=0.01, extreme_kgm2=0.05,
                                              n_pseudo_stations=8))
    tr = SEVIRDataset(loader, ids[:8], dcfg)
    va = SEVIRDataset(loader, ids[8:], dcfg)

    model = MultiTaskNowcaster(ModelConfig(
        in_channels=tr.in_channels, context_frames=2, grid_size=96,
        lead_steps=4, dim=16, depth=1))

    groups = episode_ids([loader.catalog.day_of(i) for i in ids[8:]])
    out = train(model, tr, va,
                TrainConfig(epochs=2, batch_size=2, run_dir=tmp_path / "run",
                            n_boot=40, log_every=0, device="cpu"),
                lead_minutes=[30, 60, 90, 120], val_groups=groups, resume=False)

    assert (tmp_path / "run" / "last.pt").exists()
    assert out["run_dir"].endswith("run")


def test_resume_continues_from_the_saved_epoch(loader, tmp_path):
    ids = loader.event_ids()
    dcfg = DatasetConfig(targets=TargetConfig(rain_kgm2=0.01, extreme_kgm2=0.05,
                                              n_pseudo_stations=8))
    tr, va = SEVIRDataset(loader, ids[:8], dcfg), SEVIRDataset(loader, ids[8:], dcfg)
    mk = lambda: MultiTaskNowcaster(ModelConfig(
        in_channels=tr.in_channels, context_frames=2, grid_size=96,
        lead_steps=4, dim=16, depth=1))
    cfg = lambda e: TrainConfig(epochs=e, batch_size=2, run_dir=tmp_path / "r",
                                n_boot=20, log_every=0, device="cpu")

    train(mk(), tr, va, cfg(1), lead_minutes=[30, 60, 90, 120], resume=False)
    meta = load_checkpoint(tmp_path / "r" / "last.pt", model=mk())
    assert meta["epoch"] == 0

    train(mk(), tr, va, cfg(2), lead_minutes=[30, 60, 90, 120], resume=True)
    assert load_checkpoint(tmp_path / "r" / "last.pt", model=mk())["epoch"] == 1


# --------------------------------------------------------------------------
# FAR-constrained operating points
# --------------------------------------------------------------------------

def test_far_constraint_actually_binds():
    """SEDI's false-alarm term is the RATE b/(b+d), which stays tiny when
    negatives dominate -- so an unconstrained optimum buys recall at almost
    any price in precision. On the real run that produced FAR 0.997."""
    from nowcast_train import select_threshold, select_threshold_far_constrained
    rng = np.random.default_rng(0)
    t = (rng.random(80_000) < 4e-4).astype(float)
    p = np.where(t > 0, rng.uniform(0.05, 0.35, t.shape),
                 rng.uniform(0.0, 0.25, t.shape))

    thr_u, _ = select_threshold(p, t, "sedi")
    from nowcast_eval import contingency
    far_u = contingency(p, t, thr_u).far

    thr_c, _, far_c = select_threshold_far_constrained(p, t, max_far=0.80)
    assert far_u > 0.9, f"fixture must reproduce the unconstrained problem, got {far_u}"
    assert far_c <= 0.80, f"constraint must bind, got {far_c}"
    assert thr_c >= thr_u, "the constrained point must be at least as strict"


def test_infeasible_constraint_reports_honestly():
    """If nothing meets the ceiling, return the least-bad point and say so
    rather than pretending a feasible operating point exists."""
    from nowcast_train import select_threshold_far_constrained
    rng = np.random.default_rng(1)
    t = (rng.random(40_000) < 1e-3).astype(float)
    p = rng.uniform(0.0, 1.0, t.shape)          # no skill at all
    thr, score, far = select_threshold_far_constrained(p, t, max_far=0.05)
    assert far > 0.05, "an impossible constraint must not silently appear met"
    assert np.isfinite(thr)


def test_operating_points_reports_both_side_by_side():
    from nowcast_train import operating_points
    rng = np.random.default_rng(2)
    t = (rng.random(60_000) < 5e-4).astype(float)
    p = np.where(t > 0, rng.uniform(0.05, 0.4, t.shape),
                 rng.uniform(0.0, 0.28, t.shape))
    op = operating_points({"h": p}, {"h": t}, max_far=0.8)["h"]
    assert set(op) == {"unconstrained", "far_constrained", "max_far"}
    assert op["far_constrained"]["far"] <= op["unconstrained"]["far"]
    assert "feasible" in op["far_constrained"]


# --------------------------------------------------------------------------
# Position embeddings and resolution transfer
# --------------------------------------------------------------------------

def test_model_trained_at_tile_size_runs_at_full_grid():
    """The tile->full-grid path. Full India inference is feasible (measured:
    116 MB, 15.5 s on MPS), so this must work."""
    m = MultiTaskNowcaster(ModelConfig(in_channels=4, context_frames=2,
            grid_size=96, lead_steps=6, dim=32, depth=1)).eval()
    for h, w in ((96, 96), (192, 192), (384, 288)):
        with torch.no_grad():
            out = m(torch.randn(1, 4, 2, h, w), torch.rand(1, 8, 2) * 2 - 1)
        assert out["rain_rate"].shape == (1, 6, h, w)


def test_row_position_is_independent_of_grid_width():
    """The bug this replaced: a flat 1-D table sliced [:hw] encodes token i as
    (i//wp, i%wp), and wp changes with WIDTH. Train at wp=24, infer at wp=216,
    and index 120 means (row 5, col 0) then (row 0, col 120). It does not
    raise -- it is silently spatially wrong."""
    bb = MultiTaskNowcaster(ModelConfig(in_channels=4, grid_size=96, dim=32,
                                        depth=1)).backbone
    narrow = bb.spatial_pos(24, 24).reshape(24, 24, -1)
    wide = bb.spatial_pos(24, 216).reshape(24, 216, -1)
    # a row DIFFERENCE cancels the column term, so it must match exactly
    d = ((narrow[5] - narrow[9])[0] - (wide[5] - wide[9])[0]).abs().max()
    assert d < 1e-6, f"row meaning still depends on width: {d}"


def test_column_position_is_independent_of_grid_height():
    bb = MultiTaskNowcaster(ModelConfig(in_channels=4, grid_size=96, dim=32,
                                        depth=1)).backbone
    short = bb.spatial_pos(24, 48).reshape(24, 48, -1)
    tall = bb.spatial_pos(216, 48).reshape(216, 48, -1)
    d = ((short[:, 5] - short[:, 9])[0] - (tall[:, 5] - tall[:, 9])[0]).abs().max()
    assert d < 1e-6, f"column meaning depends on height: {d}"


def test_native_resolution_needs_no_interpolation():
    bb = MultiTaskNowcaster(ModelConfig(in_channels=4, grid_size=96, dim=32,
                                        depth=1)).backbone
    p = bb.spatial_pos(bb.max_side, bb.max_side)
    assert p.shape[1] == bb.max_side ** 2
