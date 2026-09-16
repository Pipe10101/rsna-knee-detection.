"""Selection / early-stopping / weighted-soft-target tests for src/train.py.

Everything here drives the DECISION LOGIC directly on synthetic metric
sequences and tiny tensors.  No model is built and no epoch is trained: the
point of extracting the policy into pure functions was that a decision buried
in the epoch loop cannot be tested, and one that takes a handful of floats can.

Run:  python3 tests/test_train_selection.py     (from the repo root)
No pytest required -- plain asserts, so it also runs on the Kaggle image.
(There is no tests/__init__.py, so `python3 -m tests.<name>` does not work;
tests/test_train.py's header advertises that form but it has never resolved.)
"""

import os
import sys
import math
import types

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.train import (
    IMPROVED, WITHIN_RESOLUTION, REGRESSED, UNDECIDABLE,
    metric_direction, classify_score, auc_macro_resolution, loss_resolution,
    gold_studies_for_auc_resolution, resolve_selection_metric,
    CheckpointSelector, per_row_masked_bce, better,
    AsymmetricLoss, MaskedBCEWithLogitsLoss, MaskedFocalLoss, build_criterion,
    derived_cell_weights, attach_derived_rows, enforce_gold_only_validation,
    apply_gold_weight, _truthy, WallClock,
)

PASSED, FAILED = [], []

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data_subset")
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
           "Contusion", "Fracture"]


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


def _cfg(**kw):
    c = Config()
    for k, v in kw.items():
        setattr(c, k, v)
    return c


# ══════════════════════════════════════════════════════════════════════════
# 1. Metric resolution: the 1/(p*n) maths
# ══════════════════════════════════════════════════════════════════════════

def test_auc_step_is_one_over_p_times_n():
    """A single column with p positives and n negatives moves in 1/(p*n)."""
    y = np.array([[1]] * 3 + [[0]] * 9, dtype=float)          # p=3, n=9
    step, info = auc_macro_resolution(y)
    assert info["n_usable"] == 1
    assert abs(step - 1.0 / 27.0) < 1e-12, step


def test_macro_step_divides_by_the_number_of_computable_labels():
    y = np.zeros((12, 4))
    y[:6, :] = 1.0                                            # p=n=6 in all 4
    step, info = auc_macro_resolution(y)
    assert info["n_usable"] == 4
    assert abs(step - (1.0 / 36.0) / 4) < 1e-12, step


def test_single_class_columns_are_excluded_from_the_denominator():
    y = np.zeros((10, 3))
    y[:5, 0] = 1.0                                            # only col 0 usable
    step, info = auc_macro_resolution(y)
    assert info["usable_columns"] == [0], info["usable_columns"]
    assert abs(step - (1.0 / 25.0) / 1) < 1e-12


def test_all_single_class_gives_zero_delta_and_nan_step():
    """Degenerate fold: macro AUC will be NaN every epoch.  The floor must be
    0.0 (the NaN fallback, not a min_delta, is what saves the checkpoint)."""
    y = np.zeros((10, 5))
    step, info = auc_macro_resolution(y)
    assert step == 0.0 and info["n_usable"] == 0
    assert math.isnan(info["macro_step"])


def test_reduce_min_is_never_larger_than_reduce_mean():
    rng = np.random.RandomState(0)
    y = (rng.rand(13, 12) < 0.4).astype(float)
    lo, _ = auc_macro_resolution(y, reduce="min")
    hi, _ = auc_macro_resolution(y, reduce="mean")
    assert 0 < lo <= hi, (lo, hi)


def test_real_gold_folds_quantise_far_above_the_0_0003_that_killed_a_run():
    """The load-bearing number.  On the REAL fold sizes (12/10/12/11/13) the
    macro-AUC step is 0.0027-0.0041, so a 0.0003 'non-improvement' is 9-14x
    below anything the metric can express."""
    path = os.path.join(DATA, "train_gold.csv")
    if not os.path.exists(path):
        print("       (skipped: no train_gold.csv)")
        return
    df = pd.read_csv(path)
    steps = {}
    for f in sorted(df["fold"].unique()):
        v = df[df["fold"] == f]
        step, info = auc_macro_resolution(v[TARGETS].values.astype(float))
        steps[int(f)] = step
        assert info["n_usable"] == 12, (f, info["n_usable"])
    assert min(steps.values()) > 0.002, steps
    assert max(steps.values()) < 0.010, steps
    for f, s in steps.items():
        assert s / 0.0003 > 8, f"fold {f}: 0.0003 is only {0.0003/s:.2f} steps"


def test_gold_panel_size_needed_for_a_target_resolution():
    path = os.path.join(DATA, "train_gold.csv")
    if not os.path.exists(path):
        print("       (skipped: no train_gold.csv)")
        return
    df = pd.read_csv(path)
    y = df[TARGETS].values.astype(float)
    n = gold_studies_for_auc_resolution(y, target=0.005, n_folds=5)
    assert n is not None and 1 < n < 58, n      # 0.005 is already met at 58
    # Monotone: a tighter target must require strictly more studies, and a
    # genuinely fine target must require more than the panel we have.
    n2 = gold_studies_for_auc_resolution(y, target=0.001, n_folds=5)
    n3 = gold_studies_for_auc_resolution(y, target=0.0001, n_folds=5)
    assert n < n2 < n3, (n, n2, n3)
    assert n3 > 58, n3
    # Sanity-check the closed form against the exact 1/(p*n) count on a
    # synthetic panel of that size.
    rng = np.random.RandomState(11)
    big = (rng.rand(n2 // 5, 12) < y.mean(axis=0)).astype(float)
    step, _ = auc_macro_resolution(big)
    assert step < 0.005, step


def test_paired_loss_resolution_is_smaller_than_the_marginal_sem():
    """Between epochs the val set is identical, so the study-to-study spread
    cancels.  A paired SE must be far below the marginal one."""
    rng = np.random.RandomState(1)
    base = rng.rand(13) * 2.0                    # big study-to-study spread
    cur = base - 0.01                            # a uniform, tiny improvement
    marginal = loss_resolution(cur, None, k=1.0)
    paired = loss_resolution(cur, base, k=1.0)
    assert paired < marginal / 10, (paired, marginal)
    assert paired < 1e-9, paired                 # a constant shift has no spread


def test_loss_resolution_degrades_to_zero_not_to_infinity():
    assert loss_resolution(None) == 0.0
    assert loss_resolution([0.5]) == 0.0                     # n < 2
    assert loss_resolution([0.5, float("nan")]) == 0.0


# ══════════════════════════════════════════════════════════════════════════
# 2. classify_score: three outcomes, and legacy equivalence
# ══════════════════════════════════════════════════════════════════════════

def test_direction_is_maximise_for_auc_and_minimise_otherwise():
    assert metric_direction("val_auc") == +1
    assert metric_direction("gold_auc") == +1        # better() got this wrong
    assert metric_direction("val_loss") == -1
    assert metric_direction("val_crit") == -1
    assert metric_direction("something_else") == -1


def test_sub_resolution_change_is_never_an_improvement():
    """THE regression test for Priority 1.  0.0003 against a 0.003 resolution
    is a tie in BOTH directions -- it neither saves a checkpoint nor charges
    early-stopping patience."""
    d = 0.003
    assert classify_score("val_auc", 0.7003, 0.7000, d) == WITHIN_RESOLUTION
    assert classify_score("val_auc", 0.6997, 0.7000, d) == WITHIN_RESOLUTION
    assert classify_score("val_loss", 0.6997, 0.7000, d) == WITHIN_RESOLUTION
    assert classify_score("val_loss", 0.7003, 0.7000, d) == WITHIN_RESOLUTION
    # Exactly one resolution step is still a tie; strictly more is not.
    base = 0.7
    assert classify_score("val_auc", base + d * 0.999, base, d) == WITHIN_RESOLUTION
    assert classify_score("val_auc", base + d * 1.001, base, d) == IMPROVED
    assert classify_score("val_auc", base - d * 0.999, base, d) == WITHIN_RESOLUTION
    assert classify_score("val_auc", base - d * 1.001, base, d) == REGRESSED


def test_direction_is_respected_for_losses():
    assert classify_score("val_loss", 0.40, 0.50, 0.0) == IMPROVED
    assert classify_score("val_loss", 0.60, 0.50, 0.0) == REGRESSED
    assert classify_score("val_auc", 0.60, 0.50, 0.0) == IMPROVED
    assert classify_score("val_auc", 0.40, 0.50, 0.0) == REGRESSED


def test_nan_is_undecidable_not_a_regression():
    assert classify_score("val_auc", float("nan"), 0.5, 0.0) == UNDECIDABLE
    assert classify_score("val_loss", float("nan"), 0.5, 0.0) == UNDECIDABLE
    # ...and a NaN incumbent (the first epoch) never blocks a finite score.
    assert classify_score("val_auc", 0.5, float("nan"), 0.0) == IMPROVED
    assert classify_score("val_auc", 0.5, -math.inf, 0.0) == IMPROVED
    assert classify_score("val_loss", 0.5, math.inf, 0.0) == IMPROVED


def test_classify_with_zero_delta_is_exactly_the_legacy_better():
    """`better()` is the pre-fix comparison.  min_delta=0 must reproduce it for
    every combination, so `selection_min_delta=0` is a true legacy switch."""
    vals = [-math.inf, math.inf, float("nan"), 0.0, 0.5, 0.5000001, 1.0]
    for metric in ("val_auc", "val_loss", "val_crit"):
        for new in vals:
            for old in vals:
                legacy = better(metric, new, old)
                now = classify_score(metric, new, old, 0.0) == IMPROVED
                assert legacy == now, (metric, new, old, legacy, now)


def test_negative_or_nan_min_delta_is_clamped_to_zero():
    assert classify_score("val_auc", 0.51, 0.50, -5.0) == IMPROVED
    assert classify_score("val_auc", 0.51, 0.50, float("nan")) == IMPROVED


# ══════════════════════════════════════════════════════════════════════════
# 3. resolve_selection_metric
# ══════════════════════════════════════════════════════════════════════════

def test_auto_falls_back_to_val_loss_on_the_real_58_study_panel():
    path = os.path.join(DATA, "train_gold.csv")
    if not os.path.exists(path):
        print("       (skipped: no train_gold.csv)")
        return
    df = pd.read_csv(path)
    for f in sorted(df["fold"].unique()):
        v = df[df["fold"] == f][TARGETS].values.astype(float)
        sel = resolve_selection_metric(_cfg(selection_metric="auto"), v)
        assert sel["metric"] == "val_loss", (f, sel["reason"])
        assert "validation studies" in sel["reason"], sel["reason"]
        # The resolution alone is borderline (0.0027-0.0041 vs a 0.005 target);
        # what actually disqualifies AUC here is that 10-13 studies give it a
        # sampling error ~30x its quantisation.
        assert 0.002 < sel["auc_min_delta"] < 0.005, sel["auc_min_delta"]


def test_auto_switches_to_val_auc_once_the_panel_is_large_enough():
    """Priority 2's payoff: at 649 studies the fold's AUC resolves fine and
    'auto' starts driving control flow with the competition metric itself --
    no code change, no hard-coded study count."""
    rng = np.random.RandomState(7)
    y = (rng.rand(130, 12) < 0.35).astype(float)              # 649/5 folds
    sel = resolve_selection_metric(_cfg(selection_metric="auto"), y)
    assert sel["metric"] == "val_auc", sel
    assert sel["min_delta"] > 0 and sel["min_delta"] < 0.005, sel["min_delta"]


def test_explicit_val_auc_is_honoured_and_still_gets_a_resolution_floor():
    """Backward compatibility: `--set selection_metric=val_auc` still selects
    on AUC, maximising, exactly as before -- but with the guard attached."""
    y = np.zeros((12, 12))
    y[:6, :] = 1.0
    sel = resolve_selection_metric(_cfg(selection_metric="val_auc"), y)
    assert sel["metric"] == "val_auc"
    assert sel["direction"] == +1
    assert sel["min_delta"] > 0


def test_explicit_min_delta_override_wins_including_zero():
    y = np.zeros((12, 12))
    y[:6, :] = 1.0
    sel = resolve_selection_metric(
        _cfg(selection_metric="val_auc", selection_min_delta=0.0), y)
    assert sel["min_delta"] == 0.0 and sel["override"] == 0.0
    sel = resolve_selection_metric(
        _cfg(selection_metric="val_auc", selection_min_delta=0.02), y)
    assert sel["min_delta"] == 0.02


def test_val_loss_is_the_configured_default():
    """Priority 1: the shipped default must not be the quantised rank metric."""
    assert Config().selection_metric == "val_loss"
    y = np.zeros((11, 12)); y[:5, :] = 1.0
    sel = resolve_selection_metric(Config(), y)
    assert sel["metric"] == "val_loss" and sel["direction"] == -1
    # ...and it stays val_loss no matter how the panel is shaped, because the
    # choice was configured rather than inferred.
    big = np.zeros((500, 12)); big[:250, :] = 1.0
    assert resolve_selection_metric(Config(), big)["metric"] == "val_loss"


# ══════════════════════════════════════════════════════════════════════════
# 4. CheckpointSelector on synthetic metric sequences
# ══════════════════════════════════════════════════════════════════════════

def _run(selector, scores, metric="val_auc", other=None, exists_after_first=True):
    """Feed a sequence of scores; return the list of decisions."""
    out, exists = [], False
    for i, s in enumerate(scores):
        m = {"epoch": i, "val_loss": 0.6, "val_crit": 0.6, "val_auc": float("nan")}
        if other:
            m.update({k: v[i] for k, v in other.items()})
        m[metric] = s
        d = selector.step(m, checkpoint_exists=exists)
        if d["save"] and exists_after_first:
            exists = True
        out.append(d)
    return out


def test_selection_follows_the_configured_metric():
    """Same numbers, opposite conclusions, purely because of the metric name."""
    seq = [0.50, 0.60, 0.55]
    up = _run(CheckpointSelector(metric="val_auc"), seq, "val_auc")
    assert [d["status"] for d in up] == [IMPROVED, IMPROVED, REGRESSED]

    down = _run(CheckpointSelector(metric="val_loss"), seq, "val_loss")
    assert [d["status"] for d in down] == [IMPROVED, REGRESSED, REGRESSED]
    assert down[0]["best"] == 0.50


def test_val_crit_is_selectable_too():
    d = _run(CheckpointSelector(metric="val_crit"), [0.9, 0.4], "val_crit")
    assert [x["status"] for x in d] == [IMPROVED, IMPROVED]


def test_sub_resolution_wobble_never_saves_and_never_stops():
    """The exact failure that killed a run: a long tail of +/-0.0003 moves.
    With the resolution guard none of them is an improvement OR a regression,
    so no checkpoint churn and no early stop -- patience stays at 0."""
    sel = CheckpointSelector(metric="val_auc", auc_min_delta=0.003,
                             patience=3, min_epochs=1, stale_patience=99)
    seq = [0.7000, 0.7003, 0.6997, 0.7002, 0.6998, 0.7001, 0.6999, 0.7000]
    ds = _run(sel, seq, "val_auc")
    assert ds[0]["status"] == IMPROVED           # first finite score always wins
    assert all(d["status"] == WITHIN_RESOLUTION for d in ds[1:]), \
        [d["status"] for d in ds]
    assert sel.patience == 0, sel.patience
    assert sel.stop_reason(len(seq) - 1) is None
    assert sum(1 for d in ds if d["save"]) == 1, "no checkpoint churn on noise"


def test_the_same_wobble_kills_the_run_without_the_guard():
    """Control: min_delta=0 is the old arithmetic, and it early-stops on noise
    while the loss is still falling.  This is what the guard prevents."""
    sel = CheckpointSelector(metric="val_auc", auc_min_delta=0.0,
                             patience=3, min_epochs=1, legacy=True)
    seq = [0.7000, 0.6997, 0.6998, 0.6999, 0.69995]
    _run(sel, seq, "val_auc")
    assert sel.patience >= 3, sel.patience
    assert sel.stop_reason(len(seq) - 1) == "early_stopping"


def test_real_improvements_still_register_through_the_guard():
    sel = CheckpointSelector(metric="val_auc", auc_min_delta=0.003, patience=3)
    ds = _run(sel, [0.60, 0.65, 0.70, 0.75], "val_auc")
    assert all(d["status"] == IMPROVED for d in ds)
    assert abs(sel.best - 0.75) < 1e-12
    assert sel.patience == 0


def test_only_measurable_regressions_charge_patience():
    sel = CheckpointSelector(metric="val_auc", auc_min_delta=0.003,
                             patience=2, min_epochs=1, stale_patience=99)
    _run(sel, [0.80, 0.8001, 0.7999, 0.60], "val_auc")   # 2 ties, then a drop
    assert sel.patience == 1, sel.patience
    _run(sel, [0.50], "val_auc")                          # another real drop
    assert sel.patience == 2
    assert sel.stop_reason(4) == "early_stopping"


def test_a_permanent_plateau_still_terminates_via_stale_patience():
    """Ties are patience-neutral, so something else has to bound them."""
    sel = CheckpointSelector(metric="val_auc", auc_min_delta=0.01,
                             patience=4, min_epochs=1)
    assert sel.stale_limit == 8
    _run(sel, [0.70] * 9, "val_auc")
    assert sel.patience == 0
    assert sel.stop_reason(8) == "early_stopping_plateau"


def test_min_epochs_blocks_early_stopping():
    sel = CheckpointSelector(metric="val_loss", patience=1, min_epochs=4)
    _run(sel, [0.5, 0.9, 0.9], "val_loss")
    assert sel.patience >= 1
    assert sel.stop_reason(0) is None and sel.stop_reason(2) is None
    assert sel.stop_reason(3) == "early_stopping"


def test_raising_min_delta_can_only_delay_stopping_never_hasten_it():
    """Structural property that makes the guard safe to tune upward."""
    seq = [0.70, 0.69, 0.68, 0.67, 0.66, 0.65, 0.64]
    stops = []
    for d in (0.0, 0.005, 0.02, 0.5):
        s = CheckpointSelector(metric="val_auc", auc_min_delta=d, patience=3,
                               min_epochs=1, stale_patience=10 ** 6)
        first = None
        for i, sc in enumerate(seq):
            s.step({"val_auc": sc, "val_loss": 0.5})
            if first is None and s.stop_reason(i):
                first = i
        stops.append(math.inf if first is None else first)
    assert stops == sorted(stops), stops


# ══════════════════════════════════════════════════════════════════════════
# 5. NaN safety: there must ALWAYS be a checkpoint
# ══════════════════════════════════════════════════════════════════════════

def test_nan_auc_every_epoch_still_yields_a_checkpoint():
    """A degenerate gold subset makes every validation column single-class, so
    macro AUC is NaN forever.  `NaN > x` is False for every x, which once meant
    NO checkpoint was ever written and the run still exited 0."""
    assert (float("nan") > 0.0) is False, "the premise of the defect"
    sel = CheckpointSelector(metric="val_auc", auc_min_delta=0.0, patience=99)
    ds = _run(sel, [float("nan")] * 5, "val_auc")
    assert ds[0]["save"] is True, "the first epoch must materialise a checkpoint"
    assert sum(1 for d in ds if d["save"]) >= 1


def test_nan_primary_falls_back_to_val_loss_and_keeps_selecting():
    """Better than 'a' checkpoint: with the primary metric permanently NaN the
    fallback keeps CHOOSING, so the saved model is the best one seen."""
    sel = CheckpointSelector(metric="val_auc", auc_min_delta=0.0,
                             fallback_metric="val_loss", patience=99)
    losses = [0.90, 0.70, 0.50, 0.55]
    ds = []
    exists = False
    for i, l in enumerate(losses):
        d = sel.step({"val_auc": float("nan"), "val_loss": l},
                     checkpoint_exists=exists)
        exists = exists or d["save"]
        ds.append(d)
    assert all(d["fallback"] for d in ds), [d["fallback"] for d in ds]
    assert [d["status"] for d in ds] == [IMPROVED, IMPROVED, IMPROVED, REGRESSED]
    assert abs(sel.best_fallback - 0.50) < 1e-12
    assert not np.isfinite(sel.best), "the AUC incumbent must stay untouched"


def test_fallback_is_abandoned_once_the_primary_becomes_usable():
    """The two criteria must never be mixed mid-fold: once val_auc is finite,
    val_loss stops voting."""
    sel = CheckpointSelector(metric="val_auc", auc_min_delta=0.0, patience=99)
    d0 = sel.step({"val_auc": float("nan"), "val_loss": 0.9}, checkpoint_exists=False)
    assert d0["fallback"] and d0["metric"] == "val_loss"
    d1 = sel.step({"val_auc": 0.61, "val_loss": 0.8}, checkpoint_exists=True)
    assert not d1["fallback"] and d1["metric"] == "val_auc" and d1["status"] == IMPROVED
    # Even if val_auc goes NaN again later, the fallback does NOT come back.
    d2 = sel.step({"val_auc": float("nan"), "val_loss": 0.1}, checkpoint_exists=True)
    assert not d2["fallback"] and d2["status"] == UNDECIDABLE and not d2["save"]


def test_missing_checkpoint_forces_a_save_whatever_the_metric_did():
    sel = CheckpointSelector(metric="val_auc", auc_min_delta=0.0, patience=99)
    sel.step({"val_auc": 0.9, "val_loss": 0.1}, checkpoint_exists=True)
    d = sel.step({"val_auc": 0.1, "val_loss": 0.9}, checkpoint_exists=False)
    assert d["status"] == REGRESSED and d["save"] is True and d["forced"] is True
    assert abs(sel.best - 0.9) < 1e-12, "a forced save must not move the incumbent"


def test_nan_never_overwrites_a_good_incumbent():
    sel = CheckpointSelector(metric="val_auc", auc_min_delta=0.0, patience=99)
    sel.step({"val_auc": 0.88, "val_loss": 0.2}, checkpoint_exists=False)
    d = sel.step({"val_auc": float("nan"), "val_loss": 0.1}, checkpoint_exists=True)
    assert d["save"] is False and abs(sel.best - 0.88) < 1e-12


# ══════════════════════════════════════════════════════════════════════════
# 6. Legacy mode is unchanged
# ══════════════════════════════════════════════════════════════════════════

def _legacy_reference(scores, metric, patience_limit):
    """A literal transcription of the pre-fix epoch-loop arithmetic."""
    best = -math.inf if metric == "val_auc" else math.inf
    patience, exists, saves = 0, False, []
    for s in scores:
        improved = better(metric, s, best)
        if improved or not exists:
            best = s if improved else best
            patience = 0
            exists = True
            saves.append(True)
        else:
            patience += 1
            saves.append(False)
    return best, patience, saves


def test_legacy_mode_reproduces_the_old_arithmetic_exactly():
    seqs = [
        [0.70, 0.7003, 0.6997, 0.7002, 0.6998],
        [0.5, 0.5, 0.5, 0.5],
        [float("nan")] * 4,
        [float("nan"), 0.6, float("nan"), 0.61, 0.60],
        [0.0, 0.0, 0.1],
    ]
    for metric in ("val_auc", "val_loss"):
        for seq in seqs:
            sel = CheckpointSelector(metric=metric, auc_min_delta=0.05,
                                     loss_delta_k=9.0, patience=3, legacy=True)
            exists, saves = False, []
            for s in seq:
                d = sel.step({metric: s, "val_loss": s}, checkpoint_exists=exists)
                exists = exists or d["save"]
                saves.append(d["save"])
            ref_best, ref_pat, ref_saves = _legacy_reference(seq, metric, 3)
            assert saves == ref_saves, (metric, seq, saves, ref_saves)
            assert sel.patience == ref_pat, (metric, seq, sel.patience, ref_pat)
            same = (sel.best == ref_best) or (np.isnan(sel.best) and np.isnan(ref_best))
            assert same, (metric, seq, sel.best, ref_best)


def test_legacy_mode_ignores_min_delta_and_the_fallback():
    sel = CheckpointSelector(metric="val_auc", auc_min_delta=0.5, legacy=True)
    assert sel.auc_min_delta == 0.0 and sel.fallback_metric is None
    d = sel.step({"val_auc": float("nan"), "val_loss": 0.1}, checkpoint_exists=True)
    assert d["fallback"] is False and d["metric"] == "val_auc"


def test_selection_min_delta_zero_matches_legacy_saves():
    """The config-level legacy switch, without the flag."""
    seq = [0.70, 0.7003, 0.6997, 0.7002]
    sel = CheckpointSelector(metric="val_auc", override_min_delta=0.0,
                             auc_min_delta=0.05, patience=99)
    exists, saves = False, []
    for s in seq:
        d = sel.step({"val_auc": s, "val_loss": 0.5}, checkpoint_exists=exists)
        exists = exists or d["save"]
        saves.append(d["save"])
    _, _, ref = _legacy_reference(seq, "val_auc", 99)
    assert saves == ref, (saves, ref)


# ══════════════════════════════════════════════════════════════════════════
# 7. Weighted soft-target losses
# ══════════════════════════════════════════════════════════════════════════

def test_asl_reduces_exactly_to_the_old_formula_on_hard_targets():
    """The soft-target rewrite must be a strict generalisation: for y in {0,1}
    it has to reproduce the published ASL value to float precision."""
    torch.manual_seed(0)
    logits = torch.randn(16, 12) * 3.0
    y = (torch.rand(16, 12) < 0.4).float()
    gn, gp, clip = 4.0, 1.0, 0.05

    p = torch.sigmoid(logits)
    p_m = (p - clip).clamp(min=0)
    ref_pos = (1 - p).pow(gp) * (-torch.log(p.clamp(min=1e-12)))
    ref_neg = p_m.pow(gn) * (-torch.log((1 - p_m).clamp(min=1e-8)))
    ref = (y * ref_pos + (1 - y) * ref_neg).mean()

    got = AsymmetricLoss(gamma_neg=gn, gamma_pos=gp, clip=clip, ohem_ratio=1.0)(logits, y)
    assert torch.allclose(got, ref, atol=1e-5), (float(got), float(ref))


def test_asl_no_longer_treats_a_smoothed_zero_as_a_positive():
    """The bug: `safe.bool()` is True for 0.05, so every negative cell was
    scored with the POSITIVE branch and gamma_pos.  A target of 0.05 must sit
    ~20x nearer the pure-negative loss than the pure-positive one."""
    logits = torch.full((1, 1), 3.0)                  # confident positive
    asl = AsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0, clip=0.05, ohem_ratio=1.0)
    l_neg = float(asl(logits, torch.zeros(1, 1)))
    l_pos = float(asl(logits, torch.ones(1, 1)))
    l_smoothed = float(asl(logits, torch.full((1, 1), 0.05)))
    assert l_neg > l_pos, (l_neg, l_pos)              # wrong prediction costs more
    assert abs(l_smoothed - (0.95 * l_neg + 0.05 * l_pos)) < 1e-5
    assert abs(l_smoothed - l_neg) < abs(l_smoothed - l_pos)


def test_every_criterion_is_linear_in_soft_targets():
    """A soft target y must cost exactly y*L(1) + (1-y)*L(0) -- the expectation
    of the loss under Bernoulli(y).  Anything else is a hidden threshold.
    This is the property AsymmetricLoss lacked entirely before the rewrite (it
    branched on `safe.bool()`) and that MaskedFocalLoss met only approximately
    (it substituted a fractional y into p_t)."""
    torch.manual_seed(3)
    logits = torch.randn(8, 6)
    for crit in (MaskedBCEWithLogitsLoss(),
                 MaskedFocalLoss(gamma=2.0, alpha=0.25),
                 MaskedFocalLoss(gamma=0.0, alpha=-1.0),
                 AsymmetricLoss(ohem_ratio=1.0)):
        for y_val in (0.1, 0.3, 0.5, 0.7, 0.9):
            soft = float(crit(logits, torch.full_like(logits, y_val)))
            mix = (y_val * float(crit(logits, torch.ones_like(logits)))
                   + (1 - y_val) * float(crit(logits, torch.zeros_like(logits))))
            assert abs(soft - mix) < 1e-4, (type(crit).__name__, y_val, soft, mix)


def test_soft_target_loss_stays_between_the_two_hard_label_losses():
    """The bound the old focal formulation broke: substituting y=0.75 into
    (1 - p_t)^gamma gave a loss BELOW both L(0) and L(1)."""
    torch.manual_seed(4)
    logits = torch.randn(8, 6)
    for crit in (MaskedBCEWithLogitsLoss(),
                 MaskedFocalLoss(gamma=2.0, alpha=0.25),
                 AsymmetricLoss(ohem_ratio=1.0)):
        l0 = float(crit(logits, torch.zeros_like(logits)))
        l1 = float(crit(logits, torch.ones_like(logits)))
        lo, hi = min(l0, l1), max(l0, l1)
        for t in (0.0, 0.25, 0.5, 0.75, 1.0):
            v = float(crit(logits, torch.full_like(logits, t)))
            assert lo - 1e-6 <= v <= hi + 1e-6, (type(crit).__name__, t, v, lo, hi)


def test_a_soft_target_pushes_the_prediction_toward_itself():
    """The property that actually matters for optimisation: with y fixed at
    0.8, a prediction far below must be pushed UP and one far above pushed
    DOWN, for every criterion."""
    for crit in (MaskedBCEWithLogitsLoss(),
                 MaskedFocalLoss(gamma=2.0, alpha=0.5),
                 AsymmetricLoss(ohem_ratio=1.0)):
        for logit, want_up in ((-3.0, True), (3.0, False)):
            z = torch.full((1, 1), logit, requires_grad=True)
            crit(z, torch.full((1, 1), 0.8)).backward()
            g = float(z.grad)
            assert (g < 0) == want_up, (type(crit).__name__, logit, g)


def test_soft_targets_between_0_and_1_are_finite_and_ordered():
    logits = torch.zeros(1, 1)
    crit = MaskedBCEWithLogitsLoss()
    vals = [float(crit(logits, torch.full((1, 1), t))) for t in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert all(np.isfinite(v) for v in vals), vals
    assert abs(vals[0] - vals[4]) < 1e-6                 # symmetric at logit 0


def test_per_cell_weights_zero_out_a_cell_completely():
    logits = torch.tensor([[10.0, 10.0]])
    y = torch.tensor([[0.0, 0.0]])
    crit = MaskedBCEWithLogitsLoss()
    both = float(crit(logits, y))
    one = float(crit(logits, y, weight=torch.tensor([[1.0, 0.0]])))
    assert abs(both - one) < 1e-5, "a zero-weight cell must not change the mean"
    none = float(crit(logits, y, weight=torch.tensor([[0.0, 0.0]])))
    assert none == 0.0


def test_gold_outweighs_derived_in_the_gradient():
    """Priority 2's core requirement: a gold row must move the weights more
    than a derived row that disagrees with it."""
    logits = torch.zeros(2, 1, requires_grad=True)
    y = torch.tensor([[1.0], [0.0]])                 # row 0 gold, row 1 derived
    w = torch.tensor([[1.0], [0.3]])
    loss = MaskedBCEWithLogitsLoss()(logits, y, weight=w)
    loss.backward()
    g = logits.grad.flatten()
    assert abs(g[0]) > abs(g[1]) * 3 - 1e-6, g
    assert g[0] < 0 < g[1], "gold pushes up, derived pushes down"


def test_weights_and_soft_targets_compose():
    logits = torch.zeros(1, 2)
    y = torch.tensor([[0.8, 0.2]])
    w = torch.tensor([[1.0, 0.0]])
    crit = MaskedBCEWithLogitsLoss()
    got = float(crit(logits, y, weight=w))
    only = float(crit(logits[:, :1], y[:, :1]))
    assert abs(got - only) < 1e-6, (got, only)


def test_nan_targets_still_cost_nothing():
    logits = torch.randn(4, 3)
    y = torch.full((4, 3), float("nan"))
    for crit in (MaskedBCEWithLogitsLoss(), MaskedFocalLoss(),
                 AsymmetricLoss(ohem_ratio=1.0)):
        out = crit(logits, y)
        assert torch.isfinite(out) and float(out) == 0.0, type(crit).__name__


def test_build_criterion_does_not_smooth_twice():
    """train_one_epoch already clamps targets into [eps, 1-eps]; the loss must
    not smooth them a second time (0 -> 0.05 -> 0.025)."""
    crit, name = build_criterion(_cfg(loss="asl", label_smoothing=0.05))
    assert crit.label_smoothing == 0.0, name
    crit, _ = build_criterion(_cfg(loss="asl", label_smoothing=0.05,
                                   loss_label_smoothing=0.1))
    assert abs(crit.label_smoothing - 0.1) < 1e-12


# ══════════════════════════════════════════════════════════════════════════
# 8. Derived labels: weights, soft targets, gold-only validation
# ══════════════════════════════════════════════════════════════════════════

def _gold_frame(n=6):
    df = pd.DataFrame({"StudyInstanceUID": [f"g{i}" for i in range(n)]})
    for j, c in enumerate(TARGETS):
        df[c] = [(i + j) % 2 for i in range(n)]
    df["fold"] = [i % 3 for i in range(n)]
    df["is_gold"] = True
    df["is_pseudo"] = False
    return df


def _derived_frame(n=4):
    df = pd.DataFrame({"StudyInstanceUID": [f"d{i}" for i in range(n)]})
    for c in TARGETS:
        df[c] = np.linspace(0.05, 0.95, n)
    return df


def test_derived_cell_weights_are_soft_and_down_weighted():
    cfg = _cfg(labels_weight=0.3, gold_weight=1.0, soft_targets=True,
               labels_min_conf=0.0, labels_weight_column="")
    sub = _derived_frame(4)
    vals, w = derived_cell_weights(sub, TARGETS, cfg)
    assert vals.shape == (4, 12) and w.shape == (4, 12)
    assert np.allclose(w, 0.3)
    assert ((vals > 0) & (vals < 1)).any(), "soft targets must survive"
    assert vals.min() >= 0.0 and vals.max() <= 1.0


def test_soft_targets_false_hardens_them():
    cfg = _cfg(labels_weight=0.3, soft_targets=False, labels_min_conf=0.0,
               labels_weight_column="")
    vals, _ = derived_cell_weights(_derived_frame(4), TARGETS, cfg)
    assert set(np.unique(vals)) <= {0.0, 1.0}, np.unique(vals)


def test_confidence_gate_zeroes_the_weight_not_the_row():
    cfg = _cfg(labels_weight=0.3, soft_targets=True, labels_min_conf=0.8,
               labels_weight_column="")
    sub = _derived_frame(4)                       # values 0.05 .. 0.95
    vals, w = derived_cell_weights(sub, TARGETS, cfg)
    conf = np.abs(vals - 0.5) * 2
    assert np.all(w[conf < 0.8 - 1e-9] == 0.0)
    assert np.all(w[conf >= 0.8 - 1e-9] == 0.3)
    assert w.shape[0] == 4, "rows are kept; only their weight is zeroed"


def test_missing_target_columns_get_zero_weight():
    cfg = _cfg(labels_weight=0.3, labels_min_conf=0.0, labels_weight_column="")
    sub = _derived_frame(3).drop(columns=["Fracture", "Baker's"])
    vals, w = derived_cell_weights(sub, TARGETS, cfg)
    for c in ("Fracture", "Baker's"):
        assert np.all(w[:, TARGETS.index(c)] == 0.0)
    assert np.all(np.isfinite(vals)), "NaN must never reach the loader"


def test_per_cell_confidence_columns_scale_the_weight():
    """src/labels.py emits one "<label>__conf" per target; an UNMENTIONED
    finding arrives as (prob = class prior, confidence = 0.0) and must cost
    exactly nothing."""
    cfg = _cfg(labels_weight=0.4, labels_min_conf=0.0,
               labels_weight_column="", labels_conf_suffix="__conf")
    sub = _derived_frame(3)
    for j, c in enumerate(TARGETS):
        sub[c + "__conf"] = [1.0, 0.5, 0.0]          # confident / hedged / unmentioned
    vals, w = derived_cell_weights(sub, TARGETS, cfg)
    assert np.allclose(w[0], 0.4) and np.allclose(w[1], 0.2)
    assert np.allclose(w[2], 0.0), "an unmentioned finding must be weightless"
    # The soft target itself is untouched: confidence scales the WEIGHT, never
    # the value, so a 0.6 target stays 0.6 rather than being pulled toward 0.5.
    assert vals.min() >= 0.0 and vals.max() <= 1.0
    assert ((vals > 0) & (vals < 1)).any()


def test_confidence_gate_uses_explicit_confidence_when_present():
    cfg = _cfg(labels_weight=0.4, labels_min_conf=0.6, labels_weight_column="",
               labels_conf_suffix="__conf")
    sub = _derived_frame(3)
    for c in TARGETS:
        sub[c + "__conf"] = [0.9, 0.7, 0.3]
    _, w = derived_cell_weights(sub, TARGETS, cfg)
    assert np.all(w[0] > 0) and np.all(w[1] > 0)
    assert np.all(w[2] == 0.0), "below labels_min_conf must be gated out"


def test_confidence_suffix_absent_falls_back_to_the_probability_margin():
    cfg = _cfg(labels_weight=0.4, labels_min_conf=0.0, labels_weight_column="",
               labels_conf_suffix="__conf")
    vals, w = derived_cell_weights(_derived_frame(4), TARGETS, cfg)
    # No __conf columns -> flat labels_weight, NOT scaled by |p - 0.5|.
    assert np.allclose(w, 0.4), np.unique(w)


def test_per_row_confidence_column_scales_the_weight():
    cfg = _cfg(labels_weight=0.4, labels_min_conf=0.0,
               labels_weight_column="sample_weight")
    sub = _derived_frame(3)
    sub["sample_weight"] = [1.0, 0.5, 0.0]
    _, w = derived_cell_weights(sub, TARGETS, cfg)
    assert np.allclose(w[0], 0.4) and np.allclose(w[1], 0.2)
    assert np.allclose(w[2], 0.0)


def test_attach_derived_rows_is_a_noop_when_labels_csv_has_not_landed():
    gold = _gold_frame()
    w = np.ones((len(gold), 12), dtype=np.float32)
    out, ow = attach_derived_rows(gold, w, None, TARGETS, _cfg())
    assert out is gold and ow is w


def test_attach_derived_rows_marks_them_train_only_and_non_gold():
    gold = _gold_frame(6)
    w = np.ones((len(gold), 12), dtype=np.float32)
    out, ow = attach_derived_rows(gold, w, _derived_frame(4), TARGETS,
                                  _cfg(labels_weight=0.3, labels_min_conf=0.0,
                                       labels_weight_column=""),
                                  gold_ids=set(gold["StudyInstanceUID"]))
    assert len(out) == 10 and ow.shape == (10, 12)
    new = out.iloc[6:]
    assert (new["fold"] == -1).all(), "derived rows must never match a fold"
    assert (~_truthy(new["is_gold"])).all()
    assert np.allclose(ow[:6], 1.0) and np.allclose(ow[6:], 0.3)
    assert list(out.columns) == list(gold.columns), "schema must be preserved"


def test_a_gold_study_is_never_relabelled_from_its_report():
    gold = _gold_frame(3)
    derived = _derived_frame(2)
    derived.loc[0, "StudyInstanceUID"] = "g1"          # collides with gold
    out, ow = attach_derived_rows(gold, np.ones((3, 12), np.float32), derived,
                                  TARGETS, _cfg(labels_min_conf=0.0,
                                                labels_weight_column=""),
                                  gold_ids=set(gold["StudyInstanceUID"]))
    assert len(out) == 4, len(out)
    assert list(out["StudyInstanceUID"]).count("g1") == 1


def test_gold_weight_scales_only_the_gold_block():
    gold = _gold_frame(3)
    out, ow = attach_derived_rows(gold, np.ones((3, 12), np.float32),
                                  _derived_frame(2), TARGETS,
                                  _cfg(labels_weight=0.3, labels_min_conf=0.0,
                                       labels_weight_column=""),
                                  gold_ids=set(gold["StudyInstanceUID"]))
    ow = apply_gold_weight(out, ow, _cfg(gold_weight=2.0))
    assert np.allclose(ow[:3], 2.0) and np.allclose(ow[3:], 0.3)


def test_validation_ignores_non_gold_rows():
    """The essential Priority 2 invariant: validating against derived labels
    measures agreement with a keyword matcher, not with ground truth."""
    v = _gold_frame(4)
    v.loc[2:, "is_gold"] = False
    out = enforce_gold_only_validation(v, _cfg(val_gold_only=True), fold=0)
    assert len(out) == 2 and _truthy(out["is_gold"]).all()


def test_validation_gold_filter_can_be_disabled_but_shouts():
    v = _gold_frame(4)
    v.loc[2:, "is_gold"] = False
    out = enforce_gold_only_validation(v, _cfg(val_gold_only=False), fold=0)
    assert len(out) == 4


def test_a_pseudo_row_in_validation_is_fatal():
    v = _gold_frame(3)
    v.loc[1, "is_pseudo"] = True
    try:
        enforce_gold_only_validation(v, _cfg(), fold=0)
    except SystemExit as e:
        assert "validation split" in str(e)
        return
    raise AssertionError("a pseudo row in validation must abort the run")


def test_gold_only_validation_is_a_noop_without_a_provenance_column():
    v = _gold_frame(3).drop(columns=["is_gold"])
    assert len(enforce_gold_only_validation(v, _cfg(), fold=0)) == 3


def test_truthy_handles_bool_int_float_and_string_columns():
    assert list(_truthy(pd.Series([True, False]))) == [True, False]
    assert list(_truthy(pd.Series([1, 0, np.nan]))) == [True, False, False]
    assert list(_truthy(pd.Series(["True", "false", "", "0"]))) == \
        [True, False, False, False]


# ══════════════════════════════════════════════════════════════════════════
# 9. per_row_masked_bce (the paired sample behind the val_loss resolution)
# ══════════════════════════════════════════════════════════════════════════

def test_per_row_masked_bce_matches_the_torch_loss_per_row():
    torch.manual_seed(5)
    logits = torch.randn(7, 4)
    y = (torch.rand(7, 4) < 0.5).float()
    rows = per_row_masked_bce(logits.numpy(), y.numpy())
    assert rows.shape == (7,)
    crit = MaskedBCEWithLogitsLoss()
    for i in range(7):
        ref = float(crit(logits[i:i + 1], y[i:i + 1]))
        assert abs(rows[i] - ref) < 1e-5, (i, rows[i], ref)
    assert abs(rows.mean() - float(crit(logits, y))) < 1e-5


def test_per_row_masked_bce_drops_fully_masked_rows():
    logits = np.zeros((3, 2))
    y = np.array([[1.0, 0.0], [np.nan, np.nan], [0.0, 1.0]])
    rows = per_row_masked_bce(logits, y)
    assert rows.shape == (2,), rows


def test_per_row_masked_bce_is_stable_at_extreme_logits():
    rows = per_row_masked_bce(np.array([[80.0, -80.0]]), np.array([[0.0, 1.0]]))
    assert np.all(np.isfinite(rows)), rows
    assert abs(rows[0] - 80.0) < 1e-6, rows


# ══════════════════════════════════════════════════════════════════════════
# 10. Runtime guards (compile / AMP / EMA / clock) -- verified, not assumed
# ══════════════════════════════════════════════════════════════════════════

def test_torch_compile_is_gated_to_cuda_only():
    """On MPS the Inductor backend emits a Metal kernel over Apple's 32 KB
    threadgroup limit and every forward dies.  Checked structurally with the
    AST: every torch.compile call must be lexically inside an
    `if device.type == "cuda":` branch."""
    import ast
    tree = ast.parse(open(os.path.join(REPO, "src", "train.py")).read())

    def is_cuda_guard(node):
        t = node.test
        return (isinstance(t, ast.Compare)
                and isinstance(t.left, ast.Attribute) and t.left.attr == "type"
                and isinstance(t.left.value, ast.Name) and t.left.value.id == "device"
                and isinstance(t.ops[0], ast.Eq)
                and getattr(t.comparators[0], "value", None) == "cuda")

    def calls_compile(node):
        return any(isinstance(n, ast.Attribute) and n.attr == "compile"
                   and isinstance(n.value, ast.Name) and n.value.id == "torch"
                   for n in ast.walk(node))

    total = sum(1 for n in ast.walk(tree)
                if isinstance(n, ast.Attribute) and n.attr == "compile"
                and isinstance(n.value, ast.Name) and n.value.id == "torch")
    assert total >= 1, "torch.compile disappeared from train.py"

    guarded = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and is_cuda_guard(node):
            guarded += sum(1 for b in node.body
                           for n in ast.walk(b)
                           if isinstance(n, ast.Attribute) and n.attr == "compile"
                           and isinstance(n.value, ast.Name) and n.value.id == "torch")
    assert guarded == total, f"{total - guarded} torch.compile call(s) outside the CUDA guard"
    assert "torch.compile skipped on" in open(
        os.path.join(REPO, "src", "train.py")).read()


def test_amp_is_cuda_only_and_the_scaler_is_off_elsewhere():
    """A GradScaler object is never falsy, so `if scaler:` once silently
    enabled AMP on CPU.  resolve_amp must return the flag explicitly."""
    from src.train import resolve_amp
    for dev in ("cpu", "mps"):
        use_amp, dtype, need_scaler = resolve_amp(Config(), torch.device(dev))
        assert use_amp is False, dev
        assert dtype is torch.float32, dev
        assert need_scaler is False, dev
    disabled = torch.amp.GradScaler("cuda", enabled=False)
    assert bool(disabled) is True, "a GradScaler is truthy even when disabled"
    assert disabled.is_enabled() is False


def test_ema_weights_are_what_gets_validated_and_saved():
    """`eval_model` must be the SAME object handed to validate() and to
    save_best(), or the selected file is not the scored model."""
    src = open(os.path.join(REPO, "src", "train.py")).read()
    assert "eval_model = ema_model.module if ema_model is not None else model" in src
    body = src[src.index("eval_model = ema_model.module"):]
    body = body[:body.index("if interrupted:")]
    assert "eval_model," in body and "validate(" in body
    assert "save_best(eval_model, best_path)" in body
    # Every save_best CALL SITE must pass eval_model.  Checked with the AST so
    # the function definition and the prose in comments cannot match.
    import ast
    calls = [n for n in ast.walk(ast.parse(src))
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "save_best"]
    assert calls, "save_best is never called"
    for c in calls:
        first = c.args[0]
        assert isinstance(first, ast.Name) and first.id == "eval_model", \
            f"save_best called with {ast.dump(first)[:60]} -- raw weights must not be checkpointed"


def test_wallclock_default_is_sleep_inclusive_and_monotonic_is_opt_in():
    c = WallClock(10.0)
    assert c.mode == "wall"
    c.t0_wall -= 9.9 * 60                       # simulate a suspend
    assert c.expired(reserve_s=60), "Kaggle bills real time; sleep must count"

    m = WallClock(10.0, mode="monotonic")
    m.t0_wall -= 9.9 * 60
    assert not m.expired(reserve_s=60), "monotonic must ignore the suspend"
    assert WallClock(0.0).remaining() == math.inf


def main():
    print("\n=== src/train.py selection / weighting tests ===")
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and isinstance(fn, types.FunctionType):
            check(name[5:], fn)
    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    for n, e in FAILED:
        print(f"  - {n}: {e}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
