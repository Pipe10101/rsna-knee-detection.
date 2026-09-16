"""Tests for src/train.py — label integrity, losses, folds, accumulation, guards.

Run:  python3 -m tests.test_train        (from the repo root)
No pytest required; plain asserts so it works on the Kaggle image too.
"""

import os
import sys
import math
import time
import types

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.train import (
    MaskedBCEWithLogitsLoss, MaskedFocalLoss, build_criterion,
    filter_labelled, assign_folds, multilabel_stratified_folds,
    resolve_amp, WallClock, better, train_one_epoch, validate,
)

PASSED, FAILED = [], []


def check(name, fn):
    try:
        fn()
        PASSED.append(name)
        print(f"  PASS  {name}")
    except AssertionError as e:
        FAILED.append((name, str(e)))
        print(f"  FAIL  {name}: {e}")
    except Exception as e:
        FAILED.append((name, f"{type(e).__name__}: {e}"))
        print(f"  ERROR {name}: {type(e).__name__}: {e}")


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data_subset")
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
           "Contusion", "Fracture"]


# ══════════════════════════════════════════════════════════════════════════
# 1. The bug: NaN targets
# ══════════════════════════════════════════════════════════════════════════

def test_plain_bce_is_nan_on_nan_targets():
    """Baseline: this is what the old code would have done had kaggle_data.py
    not silently nan_to_num'd the labels first."""
    t = torch.tensor([[float("nan")] * 3, [1.0, 0.0, 1.0]])
    out = F.binary_cross_entropy_with_logits(torch.zeros_like(t), t)
    assert torch.isnan(out), "expected plain BCE to be NaN on NaN targets"


def test_masked_bce_finite_on_nan_targets():
    t = torch.tensor([[float("nan")] * 3, [1.0, 0.0, 1.0]])
    logits = torch.randn(2, 3, requires_grad=True)
    loss = MaskedBCEWithLogitsLoss()(logits, t)
    assert torch.isfinite(loss), f"masked BCE not finite: {loss}"
    loss.backward()
    assert torch.isfinite(logits.grad).all(), "gradient contains non-finite values"
    # Gradient must be exactly zero on the masked-out row.
    assert torch.allclose(logits.grad[0], torch.zeros(3)), \
        f"masked row received gradient: {logits.grad[0]}"


def test_masked_focal_finite_on_nan_targets():
    t = torch.tensor([[float("nan")] * 3, [1.0, 0.0, 1.0]])
    logits = torch.randn(2, 3, requires_grad=True)
    loss = MaskedFocalLoss(gamma=2.0, alpha=0.25)(logits, t)
    assert torch.isfinite(loss), f"masked focal not finite: {loss}"
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.allclose(logits.grad[0], torch.zeros(3))


def test_masked_bce_matches_plain_bce_when_fully_labelled():
    torch.manual_seed(0)
    logits, t = torch.randn(8, 12), (torch.rand(8, 12) > 0.5).float()
    a = MaskedBCEWithLogitsLoss()(logits, t)
    b = F.binary_cross_entropy_with_logits(logits, t)
    assert torch.allclose(a, b, atol=1e-6), f"{a} != {b}"


def test_masked_bce_magnitude_independent_of_mask_density():
    """Mean-over-supervised-elements, not mean-over-all: a batch that is half
    unlabelled must not report half the loss."""
    torch.manual_seed(1)
    logits, t = torch.randn(8, 12), (torch.rand(8, 12) > 0.5).float()
    full = MaskedBCEWithLogitsLoss()(logits, t)
    t2 = t.clone()
    t2[4:] = float("nan")
    half = MaskedBCEWithLogitsLoss()(logits, t2)
    ref = F.binary_cross_entropy_with_logits(logits[:4], t[:4])
    assert torch.allclose(half, ref, atol=1e-6), f"{half} != {ref}"
    assert not torch.allclose(half, full * 0.5, atol=1e-3)


# ══════════════════════════════════════════════════════════════════════════
# 2. Focal loss properties
# ══════════════════════════════════════════════════════════════════════════

def test_focal_gamma0_alpha_off_equals_bce():
    torch.manual_seed(2)
    logits, t = torch.randn(16, 12), (torch.rand(16, 12) > 0.5).float()
    f = MaskedFocalLoss(gamma=0.0, alpha=-1.0)(logits, t)
    b = F.binary_cross_entropy_with_logits(logits, t)
    assert torch.allclose(f, b, atol=1e-6), f"focal(gamma=0,no alpha)={f} vs bce={b}"


def test_focal_is_numerically_stable_at_extreme_logits():
    """The plan's `pt = torch.exp(-bce)` formulation underflows to 0 for a
    confidently-wrong logit, which turns the focal weight into exactly 1 and
    silently disables the down-weighting; and the gradient can go non-finite.
    The sigmoid-based p_t used here stays exact."""
    logits = torch.tensor([[-40.0, 40.0, 0.0]], requires_grad=True)
    t = torch.tensor([[1.0, 0.0, 1.0]])
    loss = MaskedFocalLoss(gamma=2.0, alpha=0.25)(logits, t)
    assert torch.isfinite(loss), f"focal not finite at |logit|=40: {loss}"
    loss.backward()
    assert torch.isfinite(logits.grad).all(), f"non-finite grad: {logits.grad}"

    # The naive formulation, for contrast: exp(-bce) with bce=40 underflows.
    bce = F.binary_cross_entropy_with_logits(
        logits.detach(), t, reduction="none")
    naive_pt = torch.exp(-bce)
    exact_pt = torch.sigmoid(logits.detach()) * t + \
        (1 - torch.sigmoid(logits.detach())) * (1 - t)
    # exp(-40) is 4.2e-18: not literally 0 in fp32, but 18 orders of magnitude
    # of mantissa are already gone, and it IS exactly 0 in fp16 (min normal
    # 6.1e-5), which is the dtype AMP runs this in on a T4.
    assert naive_pt[0, 0].item() < 1e-16, "expected exp(-40) to collapse"
    assert naive_pt[0, 0].half().item() == 0.0, "exp(-40) must underflow in fp16"
    assert abs(exact_pt[0, 0].item()) < 1e-16, "exact p_t should also be ~0 here"
    # and the confidently-CORRECT case, where the two disagree materially:
    lg = torch.tensor([[20.0]])
    ty = torch.tensor([[1.0]])
    b2 = F.binary_cross_entropy_with_logits(lg, ty, reduction="none")
    assert abs(torch.exp(-b2).item() - torch.sigmoid(lg).item()) < 1e-6


def test_focal_downweights_easy_examples_relative_to_bce():
    """Sanity: focal must shift relative weight from easy to hard samples."""
    easy = torch.tensor([[6.0]])      # correct, confident
    hard = torch.tensor([[0.1]])      # near the boundary
    t = torch.tensor([[1.0]])
    f = MaskedFocalLoss(gamma=2.0, alpha=-1.0)
    r_focal = f(easy, t).item() / f(hard, t).item()
    r_bce = (F.binary_cross_entropy_with_logits(easy, t).item() /
             F.binary_cross_entropy_with_logits(hard, t).item())
    assert r_focal < r_bce, f"focal ratio {r_focal} should be < bce ratio {r_bce}"


def test_build_criterion_is_switchable():
    # The shipped default is now cfg.loss="asl" (AsymmetricLoss), not bce; what
    # this test guards is that every name still resolves to the right class.
    cfg = Config()
    cfg.loss = "bce"
    crit, name = build_criterion(cfg)
    assert isinstance(crit, MaskedBCEWithLogitsLoss), f"'bce' gave {name}"
    cfg.loss = "focal"
    crit, name = build_criterion(cfg)
    assert isinstance(crit, MaskedFocalLoss), name
    assert "gamma=2.0" in name and "alpha=0.25" in name, name
    # "auto" honours the legacy flag so nothing that sets focal_loss breaks.
    cfg2 = Config()
    cfg2.loss, cfg2.focal_loss = "auto", True
    assert isinstance(build_criterion(cfg2)[0], MaskedFocalLoss)
    cfg2.focal_loss = False
    assert isinstance(build_criterion(cfg2)[0], MaskedBCEWithLogitsLoss)


# ══════════════════════════════════════════════════════════════════════════
# 3. Label filtering against the real CSV
# ══════════════════════════════════════════════════════════════════════════

def test_real_train_csv_filtering():
    path = os.path.join(DATA, "train.csv")
    if not os.path.exists(path):
        print("       (skipped: no data_subset/train.csv)")
        return
    df = pd.read_csv(path)
    assert len(df) > 4000, len(df)
    n_nan = int(df[TARGETS].isna().all(axis=1).sum())
    assert n_nan > 4000, f"expected the bulk of train.csv to be unlabelled, got {n_nan}"

    cfg = Config()
    cfg.labelled_only = True
    kept = filter_labelled(df, TARGETS, cfg)
    assert len(kept) == len(df) - n_nan, f"{len(kept)} != {len(df) - n_nan}"
    assert kept[TARGETS].notna().all().all(), "kept rows still contain NaN targets"
    assert len(kept) == 58, f"expected 58 labelled studies, got {len(kept)}"

    cfg.labelled_only = False
    assert len(filter_labelled(df, TARGETS, cfg)) == len(df)


def test_unlabelled_rows_would_be_all_negative_without_the_filter():
    """Documents exactly why filtering is required: kaggle_data.py's
    nan_to_num turns 'unknown' into 'confidently negative'."""
    block = np.full((4, 12), np.nan, dtype=np.float32)
    as_delivered = torch.tensor(np.nan_to_num(block, nan=0.0))
    assert (as_delivered == 0).all()
    loss = MaskedBCEWithLogitsLoss()(torch.zeros(4, 12), as_delivered)
    assert torch.isfinite(loss) and loss.item() > 0.6, \
        "an all-zero target block produces a real, non-zero training signal"


# ══════════════════════════════════════════════════════════════════════════
# 4. Folds
# ══════════════════════════════════════════════════════════════════════════

def test_assign_folds_prefers_existing_column():
    path = os.path.join(DATA, "train_gold.csv")
    if not os.path.exists(path):
        print("       (skipped: no train_gold.csv)")
        return
    df = pd.read_csv(path)
    original = df["fold"].tolist()
    out = assign_folds(df.copy(), TARGETS, Config())
    assert out["fold"].tolist() == original, "existing fold column was overwritten"


def test_multilabel_stratification_beats_index_mod_5():
    path = os.path.join(DATA, "train_gold.csv")
    if not os.path.exists(path):
        print("       (skipped: no train_gold.csv)")
        return
    df = pd.read_csv(path)
    y = df[TARGETS].values.astype(int)
    folds = multilabel_stratified_folds(y, n_folds=5, seed=42)

    sizes = np.bincount(folds, minlength=5)
    assert sizes.min() >= len(df) // 5 - 2, f"unbalanced fold sizes {sizes}"
    assert sizes.sum() == len(df)

    # Every fold should have both classes for most columns, otherwise that
    # column silently drops out of the macro-AUC for that fold.
    usable = []
    for f in range(5):
        v = y[folds == f]
        usable.append(sum(1 for c in range(y.shape[1])
                          if 0 < v[:, c].sum() < len(v)))
    assert min(usable) >= 10, f"folds lose too many AUC columns: {usable}"


def test_fold_assignment_is_deterministic():
    y = (np.random.RandomState(0).rand(60, 12) > 0.7).astype(int)
    a = multilabel_stratified_folds(y, 5, 42)
    b = multilabel_stratified_folds(y, 5, 42)
    assert (a == b).all(), "fold assignment is not reproducible for a fixed seed"


# ══════════════════════════════════════════════════════════════════════════
# 5. Selection criterion
# ══════════════════════════════════════════════════════════════════════════

def test_better_direction():
    assert better("val_auc", 0.7, 0.6) and not better("val_auc", 0.5, 0.6)
    assert better("val_loss", 0.4, 0.5) and not better("val_loss", 0.6, 0.5)
    assert better("val_loss", 1.0, math.inf), "must accept the first finite score"
    assert better("val_auc", 0.1, -math.inf)
    assert not better("val_auc", float("nan"), 0.5), "NaN must never win"
    assert not better("val_loss", float("nan"), 0.5)


def test_old_selection_logic_could_never_save():
    """Regression note: the previous code used `best_auc = 0.0` with a strict
    `>`, so a fold whose macro-AUC evaluated to exactly 0.0 (every column
    skipped, which is what happens when a fold is single-class) saved no
    checkpoint at all."""
    best_auc, val_auc = 0.0, 0.0
    assert not (val_auc > best_auc), "the old comparison should indeed fail"
    assert better("val_loss", 0.69, math.inf), "the new one must not"


def test_auc_quantisation_is_larger_than_typical_improvements():
    """With 10-13 validation studies a single swapped pair moves macro-AUC by
    ~0.002-0.009; selecting on smaller differences is selecting on noise."""
    path = os.path.join(DATA, "train_gold.csv")
    if not os.path.exists(path):
        print("       (skipped: no train_gold.csv)")
        return
    df = pd.read_csv(path)
    worst = 0.0
    for f in sorted(df["fold"].unique()):
        v = df[df["fold"] == f]
        steps = []
        for c in TARGETS:
            p = int(v[c].sum())
            n = len(v) - p
            if p and n:
                steps.append(1.0 / (p * n))
        worst = max(worst, max(steps) / len(steps))
    assert worst > 0.002, f"expected coarse quantisation, got {worst}"


# ══════════════════════════════════════════════════════════════════════════
# 6. AMP resolution
# ══════════════════════════════════════════════════════════════════════════

def test_resolve_amp_off_when_not_cuda():
    cfg = Config()
    use_amp, dtype, scaler = resolve_amp(cfg, torch.device("cpu"))
    assert use_amp is False and dtype is torch.float32 and scaler is False
    use_amp, _, _ = resolve_amp(cfg, torch.device("mps"))
    assert use_amp is False, "MPS must not take the CUDA autocast path"


def test_resolve_amp_falls_back_to_fp16_on_pre_ampere():
    """T4 is sm_75 and P100 is sm_60 — neither has native bfloat16. Requesting
    bf16 there must degrade to fp16 + a live GradScaler, not stay on bf16."""
    cfg = Config()
    cfg.amp_dtype = "bf16"
    real_avail = torch.cuda.is_available
    real_cap = torch.cuda.get_device_capability
    real_bf16 = torch.cuda.is_bf16_supported
    try:
        torch.cuda.is_available = lambda: True
        torch.cuda.get_device_capability = lambda *a, **k: (7, 5)   # T4
        torch.cuda.is_bf16_supported = lambda *a, **k: False
        use_amp, dtype, need_scaler = resolve_amp(cfg, torch.device("cuda"))
        assert use_amp and dtype is torch.float16 and need_scaler, \
            f"got {dtype}, scaler={need_scaler}"

        cfg.amp_dtype = "auto"
        _, dtype, need_scaler = resolve_amp(cfg, torch.device("cuda"))
        assert dtype is torch.float16 and need_scaler

        torch.cuda.get_device_capability = lambda *a, **k: (8, 0)   # A100
        torch.cuda.is_bf16_supported = lambda *a, **k: True
        _, dtype, need_scaler = resolve_amp(cfg, torch.device("cuda"))
        assert dtype is torch.bfloat16 and not need_scaler, f"got {dtype}"
    finally:
        torch.cuda.is_available = real_avail
        torch.cuda.get_device_capability = real_cap
        torch.cuda.is_bf16_supported = real_bf16


# ══════════════════════════════════════════════════════════════════════════
# 7. Wall clock
# ══════════════════════════════════════════════════════════════════════════

def test_wallclock_uses_real_time_not_monotonic_only():
    c = WallClock(budget_min=0.0)
    assert not c.expired() and c.remaining() == math.inf, "0 must disable the guard"

    c = WallClock(budget_min=10.0)
    assert not c.expired()
    # Simulate a suspend: real time jumps, CLOCK_MONOTONIC does not. A guard
    # built only on monotonic would still think it had 10 minutes left.
    c.t0_wall -= 9.9 * 60
    assert c.expired(reserve_s=60), "must notice sleep-inclusive elapsed time"
    assert c.elapsed() >= 9.9 * 60


# ══════════════════════════════════════════════════════════════════════════
# 8. Gradient accumulation
# ══════════════════════════════════════════════════════════════════════════

class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x):
        return self.fc(x)


def _run_epoch(batch_size, accum, x, y, seed=0, steps_out=None):
    torch.manual_seed(seed)
    model = _Tiny()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    if steps_out is not None:
        # Count real optimizer steps.  With GradScaler(enabled=False),
        # scaler.step(opt) delegates straight through to opt.step(), so wrapping
        # the bound method observes every update the loop actually performs.
        _inner_step = opt.step

        def _counting_step(*a, **kw):
            steps_out.append(1)
            return _inner_step(*a, **kw)

        opt.step = _counting_step
    cfg = Config()
    cfg.grad_accum_steps = accum
    # Mixup/CutMix and label smoothing are ON by default now; both are random
    # or label-altering, which makes "accumulation == one large batch" untestable.
    cfg.mixup_alpha = 0.0
    cfg.label_smoothing = 0.0
    ds = torch.utils.data.TensorDataset(x, y)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    clock = WallClock(0.0)
    # NOTE the `scheduler` positional (None here): train_one_epoch grew it when
    # OneCycleLR moved to per-step stepping, and this call site was never
    # updated, so these three tests had been erroring out rather than running.
    loss, _ = train_one_epoch(model, None, loader, opt, None,
                              MaskedBCEWithLogitsLoss(), scaler,
                              torch.device("cpu"), cfg, False, torch.float32,
                              clock, smoke=False)
    return loss, [p.detach().clone() for p in model.parameters()]


def test_grad_accum_matches_large_batch():
    torch.manual_seed(3)
    x = torch.randn(8, 4)
    y = (torch.rand(8, 3) > 0.5).float()
    l1, p1 = _run_epoch(batch_size=8, accum=1, x=x, y=y)
    l2, p2 = _run_epoch(batch_size=2, accum=4, x=x, y=y)
    for a, b in zip(p1, p2):
        assert torch.allclose(a, b, atol=1e-5), \
            f"grad accumulation diverges from the equivalent large batch:\n{a}\n{b}"
    assert abs(l1 - l2) < 1e-5, f"reported losses differ: {l1} vs {l2}"


def test_partial_accumulation_window_is_flushed():
    """A trailing micro-batch that never fills an accumulation window must still
    be applied, not silently dropped at the end of the epoch.

    Counting real optimizer steps rather than inferring "an update happened"
    from the weights having moved.  The weights-moved form only caught an
    outright removal of the flush, and only via the degenerate accum=99 path;
    it could not see the opposite bug.  Verified by mutation: dropping the
    `pending = 0` reset after a full window (which yields spurious extra steps)
    fails the accum=5 case below and passed the weights-moved assertion.
    """
    torch.manual_seed(4)
    x, y = torch.randn(10, 4), (torch.rand(10, 3) > 0.5).float()

    # 10 samples / batch_size 2 = 5 batches at accum=4: one full window
    # (batches 1-4) plus one pending -> 2 steps, the second from the flush.
    steps = []
    _run_epoch(batch_size=2, accum=4, x=x, y=y, steps_out=steps)
    assert len(steps) == 2, \
        f"5 batches at accum=4 should step twice (1 full + 1 flushed), got {len(steps)}"

    # Degenerate case: the window never fills, so the flush is the ONLY update
    # that can happen.
    steps = []
    _, p_after = _run_epoch(batch_size=2, accum=99, x=x[:2], y=y[:2], steps_out=steps)
    assert len(steps) == 1, \
        f"a never-filled window must still flush exactly once, got {len(steps)}"
    torch.manual_seed(0)
    ref = _Tiny()
    changed = any(not torch.allclose(a, b)
                  for a, b in zip([p.detach() for p in ref.parameters()], p_after))
    assert changed, "a single pending micro-batch produced no parameter update"

    # And the opposite failure: an exactly-divisible epoch must NOT pick up a
    # spurious extra step from the flush path.
    steps = []
    _run_epoch(batch_size=2, accum=5, x=x, y=y, steps_out=steps)
    assert len(steps) == 1, \
        f"5 batches at accum=5 should step exactly once, got {len(steps)}"


def test_nonfinite_loss_batches_are_skipped_not_propagated():
    torch.manual_seed(5)
    x = torch.randn(4, 4)
    y = torch.full((4, 3), float("nan"))
    y[2:] = (torch.rand(2, 3) > 0.5).float()
    loss, params = _run_epoch(batch_size=2, accum=1, x=x, y=y)
    assert math.isfinite(loss), f"epoch loss went non-finite: {loss}"
    for p in params:
        assert torch.isfinite(p).all(), "weights went non-finite"


# ══════════════════════════════════════════════════════════════════════════
# 9. validate()
# ══════════════════════════════════════════════════════════════════════════

def test_validate_reports_criterion_independent_val_loss():
    torch.manual_seed(6)
    x = torch.randn(8, 4)
    y = (torch.rand(8, 3) > 0.5).float()
    y[:, 2] = 1.0                       # single-class column -> must be skipped
    model = _Tiny()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y), batch_size=4)

    vl_b, vc_b, auc_b, cols_b, n = validate(
        model, loader, MaskedBCEWithLogitsLoss(), torch.device("cpu"),
        False, torch.float32, smoke=False)
    vl_f, vc_f, auc_f, cols_f, _ = validate(
        model, loader, MaskedFocalLoss(2.0, 0.25), torch.device("cpu"),
        False, torch.float32, smoke=False)

    assert n == 8
    assert abs(vl_b - vl_f) < 1e-6, "val_loss must not depend on the criterion"
    assert abs(vc_b - vc_f) > 1e-6, "val_crit must track the criterion"
    assert auc_b == auc_f, "AUC must be identical for the same predictions"
    assert cols_b[2] is None, "single-class column should be excluded from macro-AUC"
    assert math.isfinite(auc_b)


def test_validate_auc_is_nan_not_zero_when_undecidable():
    """A fold where no column has both classes must report NaN, so that the
    selection logic rejects it rather than treating 0.0 as a real score."""
    x = torch.randn(4, 4)
    y = torch.ones(4, 3)
    model = _Tiny()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y), batch_size=4)
    _, _, auc, _, _ = validate(model, loader, MaskedBCEWithLogitsLoss(),
                               torch.device("cpu"), False, torch.float32, smoke=False)
    assert math.isnan(auc), f"expected NaN, got {auc}"
    assert not better("val_auc", auc, -math.inf), "NaN AUC must not be selected"


# ══════════════════════════════════════════════════════════════════════════
# 10. Cache isolation across resolutions
# ══════════════════════════════════════════════════════════════════════════

def test_cache_key_now_includes_image_size():
    """kaggle_data.py used to key the cache as f'{study}_{in_channels}.npy',
    with NO resolution in it, so a 224px phase followed by a 384px phase would
    silently reuse the 224px pixels and the 'progressive resolution' would be
    fake.  It now hashes image_size (and the slice policy) into the filename
    via selection_signature().

    train.py's per-resolution cache_dir is therefore no longer load-bearing,
    but it is kept as defence in depth: it costs nothing, and it survives the
    signature being weakened again.  This test tracks which of the two is
    actually doing the work."""
    src = open(os.path.join(REPO, "src", "kaggle_data.py")).read()
    assert "def selection_signature(" in src, "cache signature helper vanished"
    sig = src.split("def selection_signature(")[1].split("\ndef ")[0]
    assert "image_size" in sig, "cache signature no longer covers image_size"
    train_src = open(os.path.join(REPO, "src", "train.py")).read()
    assert 'f"sz{cfg.image_size}_ch{cfg.in_channels}"' in train_src, \
        "the per-resolution cache_dir belt-and-braces disappeared"


def main():
    print("\n=== src/train.py tests ===")
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and isinstance(fn, types.FunctionType):
            check(name[5:], fn)
    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    for n, e in FAILED:
        print(f"  - {n}: {e}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
