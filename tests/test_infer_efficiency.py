"""Correctness tests for the inference-efficiency work in src/infer.py.

A speedup that changes predictions is a bug, so every optimisation here is
paired with a test that pins the behaviour it must NOT change:

  * batching TTA views into one forward must return what per-view forwards
    returned, INCLUDING the medial/lateral permutation on mirrored views;
  * padding a short final batch must not touch the rows that are kept;
  * torch.inference_mode() must equal torch.no_grad() exactly;
  * the cascade must leave a confident study's rank bit-identical, and must
    never reorder confident studies against each other;
  * rank-averaging must rank per LABEL across studies, never the reverse;
  * a submission must survive corrupt / missing / NaN predictions.

Everything runs on tiny tensors and a 3-layer toy net -- no backbone is built,
nothing is downloaded.

Run:  python3 -m pytest tests/test_infer_efficiency.py -q
"""

import os
import sys
import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import infer as I


TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
           "Contusion", "Fracture"]


class Tiny(nn.Module):
    """Per-sample, position-sensitive, and deterministic.

    Position-sensitive matters: a net that ignores x would make the flip and
    zoom views identical and the laterality test vacuous.
    """

    def __init__(self, n_out=12, size=16):
        super().__init__()
        torch.manual_seed(0)
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.fc = nn.Linear(4 * size * size, n_out)

    def forward(self, x):
        return self.fc(torch.relu(self.conv(x)).flatten(1))


@pytest.fixture
def net():
    return Tiny().eval()


# ══════════════════════════════════════════════════════════════════════════
# Batched TTA == sequential TTA
# ══════════════════════════════════════════════════════════════════════════

def sequential_tta(model, x, tta, perm):
    """The pre-optimisation path: one forward per TTA view."""
    out = None
    for aug_fn, needs_swap, w in tta:
        p = torch.sigmoid(model(aug_fn(x))).float()
        if needs_swap and perm is not None:
            p = p[:, perm]
        out = p * w if out is None else out + p * w
    return out


@pytest.mark.parametrize("n_views", [1, 2, 3, 5])
def test_batched_tta_equals_sequential_tta(net, n_views):
    from src.kaggle_data import build_lateral_swap
    perm, _pairs, _ = build_lateral_swap(TARGETS)
    tta = I.build_tta(n_views, 16, 16)
    x = torch.rand(6, 3, 16, 16)

    with torch.no_grad():
        want = sequential_tta(net, x, tta, perm)
    with torch.inference_mode():
        got = I.tta_predict(net, x, False, torch.float32, tta, perm,
                            max_images=6 * n_views)

    assert torch.allclose(want, got, atol=1e-6), \
        f"batched TTA diverged by {(want - got).abs().max():.3e}"


def test_batched_tta_still_undoes_the_laterality_swap(net):
    """Regression guard for a bug that already shipped once.

    A mirrored knee's "Medial Meniscus" output describes the ORIGINAL image's
    lateral side. If the permutation is dropped, the mirror view pollutes the
    two laterality pairs while leaving every other column alone -- so this
    asserts on exactly those columns.
    """
    from src.kaggle_data import build_lateral_swap
    perm, pairs, _ = build_lateral_swap(TARGETS)
    assert perm is not None and len(pairs) == 2, f"expected 2 pairs, got {pairs}"

    tta = I.build_tta(2, 16, 16)                 # original + mirror
    x = torch.rand(4, 3, 16, 16)

    with torch.inference_mode():
        with_swap = I.tta_predict(net, x, False, torch.float32, tta, perm,
                                  max_images=8)
        without = I.tta_predict(net, x, False, torch.float32, tta, None,
                                max_images=8)

    swapped_cols = [TARGETS.index(c) for pair in pairs for c in pair]
    assert not torch.allclose(with_swap[:, swapped_cols],
                              without[:, swapped_cols], atol=1e-5), \
        "the lateral permutation had no effect -- it is not being applied"
    keep = [i for i in range(len(TARGETS)) if i not in swapped_cols]
    assert torch.allclose(with_swap[:, keep], without[:, keep], atol=1e-6), \
        "the permutation touched a column that has no medial/lateral partner"


def test_view_stack_rows_match_the_weighted_mean(net):
    """return_views must hand back the SAME views the mean was built from."""
    tta = I.build_tta(5, 16, 16)
    x = torch.rand(5, 3, 16, 16)
    with torch.inference_mode():
        mean, views = I.tta_predict(net, x, False, torch.float32, tta, None,
                                    max_images=25, return_views=True)
    assert views.shape == (5, 5, 12)
    w = torch.tensor(I.tta_prefix_weights(5)).view(-1, 1, 1)
    assert torch.allclose((views * w).sum(0), mean, atol=1e-6)


def test_view_grouping_respects_the_image_budget(net):
    """A small max_images must split the views, not silently ignore the cap."""
    tta = I.build_tta(5, 16, 16)
    x = torch.rand(8, 3, 16, 16)

    seen = []
    orig = net.forward
    net.forward = lambda t: (seen.append(int(t.shape[0])), orig(t))[1]
    with torch.inference_mode():
        got = I.tta_predict(net, x, False, torch.float32, tta, None, max_images=16)
    net.forward = orig

    assert max(seen) <= 16, f"a forward carried {max(seen)} images, cap was 16"
    assert sum(seen) == 8 * 5, "not every view was computed"
    with torch.inference_mode():
        want = I.tta_predict(net, x, False, torch.float32, tta, None, max_images=40)
    assert torch.allclose(got, want, atol=1e-6), "splitting views changed the answer"


def test_inference_mode_equals_no_grad(net):
    tta = I.build_tta(3, 16, 16)
    x = torch.rand(4, 3, 16, 16)
    with torch.no_grad():
        a = I.tta_predict(net, x, False, torch.float32, tta, None, max_images=12)
    with torch.inference_mode():
        b = I.tta_predict(net, x, False, torch.float32, tta, None, max_images=12)
    assert torch.equal(a.clone(), b.clone()), "inference_mode changed the output"


def test_padding_a_short_batch_does_not_move_the_kept_rows(net):
    """predict() pads the last batch to a fixed shape; the real rows must not move."""
    tta = I.build_tta(3, 16, 16)
    x = torch.rand(5, 3, 16, 16)
    with torch.inference_mode():
        plain = I.tta_predict(net, x, False, torch.float32, tta, None, max_images=24)
        padded_in = torch.cat([x, x[-1:].repeat(3, 1, 1, 1)], 0)
        padded = I.tta_predict(net, padded_in, False, torch.float32, tta, None,
                               max_images=24)[:5]
    assert torch.allclose(plain, padded, atol=1e-6)


# ══════════════════════════════════════════════════════════════════════════
# Forward-pass accounting
# ══════════════════════════════════════════════════════════════════════════

def test_forward_stats_count_one_call_for_all_views(net):
    tta = I.build_tta(5, 16, 16)
    x = torch.rand(16, 3, 16, 16)
    I.reset_forward_stats()
    with torch.inference_mode():
        I.tta_predict(net, x, False, torch.float32, tta, None, max_images=80)
    assert I.FORWARD_STATS["calls"] == 1, \
        f"expected 1 forward for 5 views, got {I.FORWARD_STATS['calls']}"
    assert I.FORWARD_STATS["images"] == 80


# ══════════════════════════════════════════════════════════════════════════
# Cascade
# ══════════════════════════════════════════════════════════════════════════

def test_cascade_leaves_confident_studies_bit_identical():
    rng = np.random.default_rng(0)
    base = rng.random((40, 12))
    extra = rng.random((40, 12))
    idx = np.array([3, 7, 11, 19, 25])

    merged = I.cascade_merge(base, idx, [base[idx], extra[idx]])
    plain = I.rank_normalise(base)

    keep = np.setdiff1d(np.arange(40), idx)
    assert np.array_equal(merged[keep], plain[keep]), \
        "the cascade moved a study the cheap model was confident about"


def test_cascade_reuses_exactly_the_rank_slots_of_the_subset():
    """The subset must keep the rank VALUES it already held -- only their
    assignment inside the subset may change. Otherwise the subset's ranks and
    the full-set ranks are on different scales and the global ordering breaks."""
    rng = np.random.default_rng(1)
    base = rng.random((30, 12))
    idx = np.array([1, 4, 9, 14, 22, 28])
    merged = I.cascade_merge(base, idx, [base[idx], rng.random((6, 12))])
    plain = I.rank_normalise(base)
    assert np.allclose(np.sort(merged[idx], axis=0), np.sort(plain[idx], axis=0)), \
        "the cascade invented rank values the subset did not previously occupy"


def test_cascade_with_one_model_is_a_no_op():
    rng = np.random.default_rng(2)
    base = rng.random((20, 12))
    idx = np.array([2, 5, 8])
    merged = I.cascade_merge(base, idx, [base[idx]])
    assert np.allclose(merged, I.rank_normalise(base)), \
        "fusing the cheap model with itself changed the ranking"


def _same_order(a, b):
    """Same pairwise ordering in every column, ties included -- which is exactly
    and only what ROC-AUC can distinguish."""
    for j in range(a.shape[1]):
        x, y = a[:, j], b[:, j]
        sx = np.sign(x[:, None] - x[None, :])
        sy = np.sign(y[:, None] - y[None, :])
        if not np.array_equal(sx, sy):
            return False
    return True


def _refines(merged, plain):
    """merged agrees with plain wherever plain expressed a strict order, and may
    only add an order where plain tied."""
    for j in range(merged.shape[1]):
        x, y = merged[:, j], plain[:, j]
        sx = np.sign(x[:, None] - x[None, :])
        sy = np.sign(y[:, None] - y[None, :])
        strict = sy != 0
        if not np.array_equal(sx[strict], sy[strict]):
            return False
    return True


def test_cascade_over_every_study_refines_a_plain_rank_average():
    """With the subset = every study, the cascade must reproduce the ensemble's
    ordering. It may REFINE a tie (rank_average ties whenever two models' ranks
    cross; the cascade breaks that with model 1) but must never contradict a
    strict ordering."""
    rng = np.random.default_rng(3)
    a, b = rng.random((25, 12)), rng.random((25, 12))
    idx = np.arange(25)
    merged = I.cascade_merge(a, idx, [a[idx], b[idx]])
    plain = I.rank_average([a, b])
    assert _refines(merged, plain)


def test_cascade_breaks_ties_with_the_cheap_models_ranking_not_array_order():
    """rank_average ties whenever two models' ranks cross. The tiebreak must be
    model 1's opinion, so the result does not depend on the order studies happen
    to sit in sample_submission.csv."""
    a = np.array([[0.1], [0.2], [0.3], [0.4]])
    b = np.array([[0.4], [0.3], [0.2], [0.1]])
    idx = np.arange(4)
    assert len(np.unique(I.rank_average([a, b]))) == 1, "setup: expected a total tie"

    merged = I.cascade_merge(a, idx, [a[idx], b[idx]])
    # With the ensemble indifferent, model 1's order must survive intact.
    assert list(np.argsort(merged[:, 0])) == [0, 1, 2, 3], merged.ravel()

    # Permuting the rows must permute the answer, not change it.
    p = np.array([2, 0, 3, 1])
    merged_p = I.cascade_merge(a[p], np.arange(4), [a[p], b[p]])
    assert np.allclose(np.sort(merged_p[:, 0]), np.sort(merged[:, 0]))
    assert list(np.argsort(merged_p[:, 0])) == list(np.argsort(a[p][:, 0]))


def test_cascade_only_refines_rank_average_when_the_extra_models_tie():
    """Regression from the sweep harness: cascade_merge and rank_average
    disagreed on 12 studies because ties are common at that size.

    Model 1's outputs are continuous sigmoids (distinct ranks, so distinct rank
    slots); the EXTRA models are the tie-prone ones here. Under those conditions
    the cascade must reproduce the ensemble order exactly, refining ties only.
    """
    rng = np.random.default_rng(7)
    for trial in range(25):
        a = rng.random((12, 12))                          # continuous: no ties
        b = rng.integers(0, 4, (12, 12)).astype(float)    # heavily tied
        idx = np.arange(12)
        merged = I.cascade_merge(a, idx, [a[idx], b[idx]])
        plain = I.rank_average([a, b])
        assert _refines(merged, plain), f"contradicted a strict order on trial {trial}"


def test_cascade_cannot_separate_studies_the_cheap_model_tied():
    """A documented limitation, pinned so it cannot change silently.

    The cascade re-slots onto model 1's rank grid. If model 1 tied two subset
    studies they share one slot value, so the ensemble's preference between them
    is unrepresentable. Model 1 emits continuous sigmoids, so this needs exactly
    equal predictions to trigger -- but it is real, and it is why the cascade is
    a re-ORDERING of model 1's ranks and not a replacement for them.
    """
    base = np.array([[0.5], [0.5], [0.9]])          # rows 0 and 1 tied
    extra = np.array([[0.1], [0.9], [0.5]])         # the ensemble prefers row 1
    merged = I.cascade_merge(base, np.arange(3), [base, extra])
    assert merged[0, 0] == merged[1, 0], \
        "expected the tied pair to stay on one slot; the invariant changed"


def test_cascade_never_moves_a_confident_study_across_a_subset_study():
    """The count of subset studies ranked below each confident study must not
    change -- that is what "the confident study's rank is untouched" means in
    pairwise terms."""
    rng = np.random.default_rng(11)
    base = rng.random((30, 12))
    idx = np.array([2, 5, 9, 13, 21, 27])
    keep = np.setdiff1d(np.arange(30), idx)
    plain = I.rank_normalise(base)
    merged = I.cascade_merge(base, idx, [base[idx], rng.random((6, 12))])
    before = (plain[idx][:, None, :] < plain[keep][None, :, :]).sum(axis=0)
    after = (merged[idx][:, None, :] < merged[keep][None, :, :]).sum(axis=0)
    assert np.array_equal(before, after), \
        "the cascade moved subset studies across a confident study"


def test_cascade_handles_an_empty_subset():
    base = np.random.default_rng(4).random((10, 12))
    assert np.allclose(I.cascade_merge(base, np.array([], int), []),
                       I.rank_normalise(base))


# ══════════════════════════════════════════════════════════════════════════
# Rank averaging axis  (per LABEL across studies)
# ══════════════════════════════════════════════════════════════════════════

def test_rank_normalise_ranks_studies_within_a_label_not_labels_within_a_study():
    # Column 0 ascending, column 1 descending. Ranking down the columns must
    # recover those two orders independently; ranking across rows would not.
    p = np.array([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]])
    r = I.rank_normalise(p)
    assert list(np.argsort(r[:, 0])) == [0, 1, 2]
    assert list(np.argsort(r[:, 1])) == [2, 1, 0]
    assert np.allclose(r.max(axis=0), 1.0)


def test_rank_average_is_invariant_to_per_model_calibration():
    rng = np.random.default_rng(5)
    p = rng.random((20, 12))
    squashed = 0.001 + 0.002 * p          # same order, wildly different scale
    assert np.allclose(I.rank_average([p, p]), I.rank_average([p, squashed]))


# ══════════════════════════════════════════════════════════════════════════
# Submission robustness
# ══════════════════════════════════════════════════════════════════════════

def test_write_submission_scrubs_nan_and_clips_range():
    import pandas as pd
    vals = np.array([[np.nan, np.inf, -np.inf, 2.5, -1.0, 0.5,
                      0.0, 1.0, 0.25, 0.75, 0.1, 0.9]])
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "submission.csv")
        I.write_submission(path, ["s1"], TARGETS, "StudyInstanceUID", vals)
        got = pd.read_csv(path)
    row = got[TARGETS].values[0]
    assert not np.isnan(row).any(), "a NaN reached the submission"
    assert np.isfinite(row).all(), "an inf reached the submission"
    assert (row >= 0).all() and (row <= 1).all(), f"out of [0,1]: {row}"
    assert row[0] == 0.5 and row[1] == 1.0 and row[2] == 0.0


def test_write_submission_keeps_template_row_order():
    import pandas as pd
    ids = [f"study_{i}" for i in range(5)][::-1]      # deliberately not sorted
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "submission.csv")
        I.write_submission(path, ids, TARGETS, "StudyInstanceUID",
                           np.full((5, 12), 0.5))
        got = pd.read_csv(path)
    assert got["StudyInstanceUID"].tolist() == ids
    assert list(got.columns) == ["StudyInstanceUID"] + TARGETS


# ══════════════════════════════════════════════════════════════════════════
# BatchSource  (decode once per preprocessing group)
# ══════════════════════════════════════════════════════════════════════════

class _CountingLoader:
    def __init__(self, batches):
        self.batches = batches
        self.passes = 0

    def __iter__(self):
        self.passes += 1
        for b in self.batches:
            yield b

    def __len__(self):
        return len(self.batches)


def test_batch_source_reuses_the_first_pass():
    batches = [torch.rand(4, 3, 8, 8) for _ in range(3)]
    loader = _CountingLoader(batches)
    src = I.BatchSource(loader, budget_bytes=10 * 2**20)

    first = [b.clone() for b in src]
    second = [b.clone() for b in src]

    assert loader.passes == 1, "the loader was iterated again after memoisation"
    assert src.cached
    for a, b in zip(first, second):
        assert torch.equal(a, b), "the memoised batch differs from the first pass"


def test_batch_source_does_not_memoise_a_partial_pass():
    """A calibration pass breaks early; installing that as the cache would make
    every later model see a truncated test set."""
    loader = _CountingLoader([torch.rand(4, 3, 8, 8) for _ in range(4)])
    src = I.BatchSource(loader, budget_bytes=10 * 2**20)
    for i, _b in enumerate(src):
        if i == 1:
            break
    assert not src.cached, "an early-stopped pass installed a truncated cache"
    assert len(list(src)) == 4, "the full pass did not see every batch"


def test_batch_source_respects_the_ram_budget():
    loader = _CountingLoader([torch.rand(4, 3, 64, 64) for _ in range(8)])
    src = I.BatchSource(loader, budget_bytes=1024)          # far too small
    list(src)
    assert not src.cached, "the RAM budget was ignored"
    assert loader.passes == 1
    list(src)
    assert loader.passes == 2, "an uncached source must re-read the loader"


def test_batch_source_with_zero_budget_never_caches():
    loader = _CountingLoader([torch.rand(2, 3, 8, 8) for _ in range(2)])
    src = I.BatchSource(loader, budget_bytes=0)
    list(src)
    list(src)
    assert not src.cached and loader.passes == 2


# ══════════════════════════════════════════════════════════════════════════
# AMP / device policy
# ══════════════════════════════════════════════════════════════════════════

def test_fp32_override_disables_autocast():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp, dtype = I.resolve_amp(dev, force_fp32=True)
    assert use_amp is False and dtype is torch.float32


def test_autocast_helper_is_a_true_noop_when_disabled():
    """autocast(enabled=False) is numerically a no-op but warns on CPU/MPS with
    dtype=float32; the helper must return a plain null context instead."""
    import contextlib
    ctx = I._autocast(False, "cpu", torch.float32)
    assert isinstance(ctx, contextlib.nullcontext().__class__)


def test_tta_prefix_weights_match_build_tta():
    for n in range(1, 6):
        built = [w for _fn, _s, w in I.build_tta(n, 16, 16)]
        assert np.allclose(built, I.tta_prefix_weights(n))
        assert abs(sum(built) - 1.0) < 1e-9


def test_build_tta_is_a_prefix_family():
    """The offline (N, M) sweep reconstructs every smaller M from one pass at
    M=5, which is only valid if the views are a prefix family."""
    x = torch.rand(2, 3, 16, 16)
    full = I.build_tta(5, 16, 16)
    for n in range(1, 6):
        part = I.build_tta(n, 16, 16)
        for i in range(n):
            assert torch.equal(part[i][0](x), full[i][0](x)), f"view {i} differs at n={n}"
            assert part[i][1] == full[i][1], f"swap flag differs at view {i}, n={n}"


# ══════════════════════════════════════════════════════════════════════════
# OOM classification
# ══════════════════════════════════════════════════════════════════════════

def test_oom_detection_catches_the_plain_runtime_error_form():
    assert I._is_oom(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))
    assert not I._is_oom(RuntimeError("shape mismatch"))
    assert not I._is_oom(ValueError("nope"))
