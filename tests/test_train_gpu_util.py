"""Tests for the GPU-utilisation planner in src/train.py.

Covers batch sizing, LR scaling, warmup derivation and the OneCycleLR
step-count fix.  Every numeric expectation here is the arithmetic from
src/model.py's MEASURED activation table, so if that table is re-measured these
tests are supposed to fail and be updated deliberately.

Run:  python3 -m pytest tests/test_train_gpu_util.py -q
"""

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.train import (
    memory_cap_batch, peak_gib_for_batch, plan_gpu_utilisation,
    scale_lr_for_batch, onecycle_pct_start, _cfg_without_ckpt,
    ONECYCLE_MIN_UPDATES, optimiser_extras_gib, should_drop_last,
)


# ── the arithmetic the whole plan rests on ────────────────────────────────

def _cfg(**kw):
    c = Config()
    c.backbone = "tf_efficientnet_b0_ns"
    c.image_size = 224
    c.in_channels = 3
    c.num_classes = 12
    c.gradient_checkpointing = False
    c.grad_checkpointing = False
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def test_measured_footprint_matches_the_published_table():
    """effnet_b0@224 = 0.968 GiB model fixed + 43.10 MiB/sample. All else follows."""
    cap, info = memory_cap_batch(_cfg(use_ema=False, use_lookahead=False),
                                 "t4", headroom=0.85)
    assert info["n_params"] == 4_025_481, info["n_params"]
    assert info["act_mib_per_sample"] == pytest.approx(43.10, abs=0.01)
    assert info["model_fixed_gib"] == pytest.approx(0.9675, abs=0.001)
    assert info["extras_gib"] == 0.0
    assert info["vram_gib"] == 15.0          # a T4 is 15.0 GiB usable, not 16
    assert cap == 279, cap                   # largest batch at 85% of the card


@pytest.mark.parametrize("batch,gib", [
    (8, 1.30), (64, 3.66), (128, 6.35), (256, 11.74),
])
def test_peak_memory_table(batch, gib):
    """The published table, EXCLUDING train.py's own EMA/Lookahead copies."""
    _, info = memory_cap_batch(_cfg(use_ema=False, use_lookahead=False), "t4")
    assert peak_gib_for_batch(info, batch) == pytest.approx(gib, abs=0.02)


def test_optimiser_extras_are_priced_and_are_not_free():
    """EMA and Lookahead each keep a full fp32 parameter copy on the GPU."""
    n = 4_025_481
    assert optimiser_extras_gib(_cfg(use_ema=False, use_lookahead=False), n) == 0.0
    one = optimiser_extras_gib(_cfg(use_ema=True, use_lookahead=False), n)
    two = optimiser_extras_gib(_cfg(use_ema=True, use_lookahead=True), n)
    assert one == pytest.approx(n * 4 / 1024 ** 3)
    assert two == pytest.approx(2 * one)
    # On a 88M-parameter convnext_base the same two copies are 0.65 GiB, which
    # is the difference between an 84.9% plan and an 87.0% OOM.
    big = optimiser_extras_gib(_cfg(use_ema=True, use_lookahead=True), 87_600_000)
    assert big == pytest.approx(0.653, abs=0.01)


def test_convnext_base_384_is_vram_bound_not_data_bound():
    """The production 9h script's real config. Extras are what make it bind."""
    c = _cfg(backbone="convnext_base.fb_in22k_ft_in1k", image_size=384,
             use_ema=False, use_lookahead=True)
    p = plan_gpu_utilisation(c, n_train=519, base_batch=2, base_accum=8)
    assert "VRAM" in p["binding_constraint"], p["binding_constraint"]
    assert p["batch_size"] == 31
    assert p["peak_gib"] <= 15.0 * 0.85


def test_batch_8_uses_under_nine_percent_of_a_t4():
    _, info = memory_cap_batch(_cfg(), "t4")
    pct = 100.0 * peak_gib_for_batch(info, 8) / info["vram_gib"]
    assert 8.0 < pct < 9.0, pct


def test_grad_checkpointing_cuts_activations_by_the_measured_ratio():
    _, on = memory_cap_batch(_cfg(gradient_checkpointing=True,
                                  use_ema=False, use_lookahead=False), "t4")
    _, off = memory_cap_batch(_cfg(gradient_checkpointing=False,
                                   use_ema=False, use_lookahead=False), "t4")
    assert on["grad_checkpointing"] is True
    assert off["grad_checkpointing"] is False
    ratio = on["act_mib_per_sample"] / off["act_mib_per_sample"]
    assert ratio == pytest.approx(0.109, abs=0.002)
    # ...and it saves 0.30 GiB at batch 8, on a 15.0 GiB card.
    saved = peak_gib_for_batch(off, 8) - peak_gib_for_batch(on, 8)
    assert saved == pytest.approx(0.30, abs=0.02)


def test_cfg_without_ckpt_does_not_mutate_the_real_config():
    c = _cfg(gradient_checkpointing=True, grad_checkpointing=True)
    view = _cfg_without_ckpt(c)
    assert view.gradient_checkpointing is False
    assert view.grad_checkpointing is False
    assert view.backbone == c.backbone       # falls through for everything else
    assert c.gradient_checkpointing is True  # original untouched
    assert c.grad_checkpointing is True


# ── the conditional recommendation: 58-study vs weak-label regime ─────────

def test_gold_regime_stays_at_batch_eight():
    """46 training studies cannot fill a bigger batch; the plan must say so."""
    p = plan_gpu_utilisation(_cfg(), n_train=46, base_batch=8, base_accum=2)
    assert p["batch_size"] == 8
    assert p["grad_accum_steps"] == 1
    assert "training-split size" in p["binding_constraint"]
    assert p["updates_per_epoch"] == 6        # ceil(46/8) micro-batches, accum 1


def test_weak_label_regime_scales_up():
    """519 training studies (649 with pixels, 5-fold) support batch 32."""
    p = plan_gpu_utilisation(_cfg(), n_train=519, base_batch=8, base_accum=2)
    assert p["batch_size"] == 32              # 519 // auto_batch_target_steps(16)
    assert p["grad_accum_steps"] == 1
    assert p["updates_per_epoch"] == 17       # ceil(519/32), accum 1


def test_target_steps_is_the_knob_that_trades_updates_for_throughput():
    for target, want in ((16, 32), (8, 64), (4, 128)):
        p = plan_gpu_utilisation(_cfg(auto_batch_target_steps=target),
                                 n_train=519, base_batch=8, base_accum=2)
        assert p["batch_size"] == want, (target, p["batch_size"])


def test_batch_never_exceeds_the_training_split():
    for n in (3, 7, 12, 40):
        p = plan_gpu_utilisation(_cfg(), n_train=n, base_batch=8, base_accum=2)
        assert p["batch_size"] <= max(1, n), (n, p["batch_size"])


def test_auto_batch_max_caps_a_huge_split():
    p = plan_gpu_utilisation(_cfg(), n_train=100_000, base_batch=8, base_accum=2)
    assert p["batch_size"] == 128             # auto_batch_max, not the 279 mem cap


def test_memory_cap_binds_when_the_model_is_big_enough():
    """A batch the card cannot hold must be refused even with data to spare."""
    c = _cfg(auto_batch_max=4096, auto_batch_target_steps=1,
             use_ema=False, use_lookahead=False)
    cap, _ = memory_cap_batch(c, "t4", headroom=0.85)
    p = plan_gpu_utilisation(c, n_train=100_000, base_batch=8, base_accum=2)
    assert p["batch_size"] == cap == 279
    assert "VRAM" in p["binding_constraint"]
    assert p["peak_gib"] <= 15.0 * 0.85


def test_disabled_planner_is_a_no_op():
    p = plan_gpu_utilisation(_cfg(auto_batch_size=False), n_train=519,
                             base_batch=8, base_accum=2)
    assert (p["batch_size"], p["grad_accum_steps"]) == (8, 2)


def test_accumulation_is_folded_away_but_can_be_kept():
    kept = plan_gpu_utilisation(_cfg(auto_grad_accum=False), n_train=46,
                                base_batch=8, base_accum=2)
    assert kept["grad_accum_steps"] == 2


# ── LR scaling ────────────────────────────────────────────────────────────

def test_sqrt_is_the_default_rule():
    assert Config().lr_scale_rule == "sqrt"
    assert Config().lr_base_batch == 16


@pytest.mark.parametrize("rule,eff,mult", [
    ("sqrt",   16,  1.0),
    ("sqrt",   32,  math.sqrt(2)),
    ("sqrt",  128,  math.sqrt(8)),
    ("linear", 32,  2.0),
    ("none",  128,  1.0),
])
def test_lr_scaling_rules(rule, eff, mult):
    lr, info = scale_lr_for_batch(_cfg(lr_scale_rule=rule), 3e-4, eff)
    assert info["multiplier"] == pytest.approx(mult, rel=1e-6)
    assert lr == pytest.approx(3e-4 * mult, rel=1e-6)


def test_linear_scaling_is_capped_so_it_cannot_destroy_a_backbone():
    """8x the batch under the linear rule is 2.4e-3; the cap holds it at 4x."""
    lr, info = scale_lr_for_batch(_cfg(lr_scale_rule="linear", lr_scale_max=4.0),
                                  3e-4, 256)
    assert info["ratio"] == 16.0
    assert info["multiplier"] == 4.0
    assert info["capped_at"] == 4.0
    assert lr == pytest.approx(1.2e-3)


def test_unknown_rule_degrades_to_none_instead_of_crashing():
    lr, info = scale_lr_for_batch(_cfg(lr_scale_rule="cosine?"), 3e-4, 128)
    assert info["rule"] == "none" and lr == 3e-4


def test_lr_scaling_is_idempotent_across_folds():
    """Fold 2 must not scale fold 1's already-scaled LR (compounding bug)."""
    c = _cfg()
    base = c.lr
    seen = []
    for _ in range(5):
        c.lr = base                      # what the fold loop does
        c.lr, _ = scale_lr_for_batch(c, base, 32)
        seen.append(c.lr)
    assert len(set(seen)) == 1
    assert seen[0] == pytest.approx(base * math.sqrt(2))


# ── warmup / OneCycleLR ───────────────────────────────────────────────────

def test_warmup_epochs_finally_reaches_the_scheduler():
    """It was declared in Config and read by nothing; pct_start was hard 0.1."""
    pct, steps = onecycle_pct_start(_cfg(warmup_epochs=1.0), 6, 72)
    assert steps == 6
    assert pct == pytest.approx(6 / 72)
    pct2, steps2 = onecycle_pct_start(_cfg(warmup_epochs=2.0), 6, 72)
    assert steps2 == 12 and pct2 > pct


def test_warmup_has_an_absolute_floor():
    """A big batch shrinks the update count; a fraction alone would vanish."""
    pct, steps = onecycle_pct_start(_cfg(warmup_epochs=0.0, warmup_min_steps=3),
                                    1, 12)
    assert steps == 3
    assert 0.0 < pct < 1.0


def test_pct_start_stays_in_range_for_degenerate_schedules():
    """Every total_steps >= 2 must produce a pct_start OneCycleLR accepts.

    OneCycleLR needs 1 <= pct_start * total_steps < total_steps, or it either
    builds a negative-length warmup or divides by zero.  total_steps == 1 can
    satisfy neither, which is why train.py refuses to build a scheduler there.
    """
    cases = [(1, 3), (1, 4), (100, 100), (6, 3), (6, 72), (32, 384), (9, 108)]
    for upe, total in cases:
        assert total >= ONECYCLE_MIN_UPDATES
        pct, _ = onecycle_pct_start(_cfg(), upe, total)
        assert 0.0 < pct <= 0.5, (upe, total, pct)
        sch = torch.optim.lr_scheduler.OneCycleLR(
            torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1),
            max_lr=0.1, total_steps=total, pct_start=pct)
        for _ in range(total - 1):
            sch.step()


def test_onecycle_really_cannot_be_built_at_two_updates():
    """total_steps=2 admits no pct_start: the only legal one has zero length."""
    pct, _ = onecycle_pct_start(_cfg(), 1, 2)
    with pytest.raises(ZeroDivisionError):
        torch.optim.lr_scheduler.OneCycleLR(
            torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1),
            max_lr=0.1, total_steps=2, pct_start=pct)


def test_one_update_schedule_degenerates_to_no_warmup():
    """total_steps=1 builds, but with pct_start=0 -- no warmup at all.

    That is a schedule in name only, which is the other half of why
    ONECYCLE_MIN_UPDATES is 3 rather than 2.
    """
    pct, steps = onecycle_pct_start(_cfg(), 1, 1)
    assert (pct, steps) == (0.0, 0)


def test_onecycle_step_count_matches_optimiser_updates():
    """The shipped code passed micro-batches and stepped once per UPDATE.

    With 6 micro-batches and accum=2 the scheduler was built for 72 steps and
    received 36, so the cosine never finished and training ended at ~59% of
    max_lr.  Reproduce the old behaviour, then the fixed one.
    """
    n_batches, accum, epochs = 6, 2, 12

    def end_lr(steps_per_epoch, n_steps):
        p = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.SGD([p], lr=1.0)
        sch = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=1.0, epochs=epochs, steps_per_epoch=steps_per_epoch,
            pct_start=0.1, div_factor=10.0, final_div_factor=1e4)
        for _ in range(n_steps):
            sch.step()
        return opt.param_groups[0]["lr"]

    actual_updates = epochs * math.ceil(n_batches / accum)   # 36
    broken = end_lr(n_batches, actual_updates)               # built for 72
    fixed = end_lr(math.ceil(n_batches / accum), actual_updates)

    assert broken > 0.5, broken             # never annealed: ~59% of max_lr
    assert fixed < 0.01, fixed              # actually annealed
    assert broken / fixed > 50, (broken, fixed)


def test_scheduler_survives_a_single_update_fold():
    pct, _ = onecycle_pct_start(_cfg(), 1, 12)
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=1.0)
    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=1.0, epochs=12, steps_per_epoch=1, pct_start=pct)
    for _ in range(12):
        sch.step()


# ── do-not-regress guards ─────────────────────────────────────────────────

def test_amp_dtype_default_is_still_auto():
    assert Config().amp_dtype == "auto"


def test_selection_metric_default_is_unchanged():
    assert getattr(Config(), "selection_metric", "val_loss") in ("val_loss", "auto")


def test_new_fields_are_all_scalars_so_set_kv_works():
    """Config.from_args converts with type(getattr(cfg, k)); containers break it."""
    c = Config()
    for name in ("auto_batch_size", "auto_batch_target_steps", "auto_batch_min",
                 "auto_batch_max", "auto_batch_headroom", "auto_batch_device",
                 "auto_grad_accum", "auto_grad_checkpointing", "lr_scale_rule",
                 "lr_base_batch", "lr_scale_max", "warmup_min_steps"):
        assert isinstance(getattr(c, name), (bool, int, float, str)), name


# ── drop_last ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n,b,dev,want", [
    # BatchNorm safety: a remainder of 1 or 2 goes, on every device.
    (49, 8, "cpu", True),    # 6 full + 1  -- a batch of 1 raises in BN
    (50, 8, "mps", True),    # 6 full + 2  -- statistics from 2 are noise
    (49, 8, "cuda", True),
    # 46 gold studies at batch 8: remainder is 13% of the fold. Too much to
    # discard every epoch just to keep torch.compile from re-specialising.
    (46, 8, "cuda", False),
    (46, 8, "cpu", False),
    # 637 report-labelled studies at batch 39: remainder is 2%. Free.
    (637, 39, "cuda", True),
    (637, 39, "cpu", False),   # nothing to gain off CUDA
    (637, 31, "cuda", True),   # 20 full + 17 = 2.7%
    # Nothing to drop, or nothing left after dropping.
    (48, 8, "cuda", False),    # exact fit
    (5, 8, "cuda", False),     # fewer studies than one batch
    (0, 8, "cuda", False),
])
def test_should_drop_last(n, b, dev, want):
    assert should_drop_last(n, b, dev) is want, (n, b, dev)


def test_drop_last_never_empties_the_epoch():
    for n in range(1, 200):
        for b in (8, 16, 32, 39, 64):
            if should_drop_last(n, b, "cuda"):
                assert n // b >= 1, (n, b)
