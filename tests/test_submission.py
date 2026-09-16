"""Inference / soup tests for src/infer.py and src/soup.py.

These are the two modules the rest of the suite does not reach, and they are the
two that decide whether a submission exists at all.  Everything here drives pure
functions and tiny tensors -- no backbone is built, nothing is downloaded -- so
it runs on the Kaggle image in about a second.

Several tests are REGRESSION tests for defects that shipped:

  * ``F.interpolate`` was called in the multi-scale TTA with no
    ``import torch.nn.functional as F``, so every submission run died with
    ``NameError`` on its first batch.  test_tta_zoom_executes covers it.
  * ``amp_dtype`` was hard-coded to bfloat16, which needs sm_80+; Kaggle's T4 is
    sm_75.  test_amp_dtype_matches_hardware covers it.
  * the submission's columns were rebuilt from a constant in the source instead
    of read from sample_submission.csv.  test_submission_shape_follows_template
    covers it.

Run:  python3 -m pytest tests/test_submission.py -q
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import infer as I
from src import soup as S


# ══════════════════════════════════════════════════════════════════════════
# TTA
# ══════════════════════════════════════════════════════════════════════════

def test_tta_zoom_executes():
    """Regression: the zoom view used F.interpolate with F never imported."""
    tta = I.build_tta(5, 32, 32)
    x = torch.rand(2, 3, 32, 32)
    for fn, _needs_swap, _w in tta:
        out = fn(x)
        assert out.shape == x.shape, f"a TTA view changed the shape: {out.shape}"


@pytest.mark.parametrize("n,expected", [(1, 1), (3, 3), (5, 5), (99, 5), (0, 1)])
def test_tta_count_is_honoured_and_clamped(n, expected):
    """tta_n is the wall-clock lever, so it has to actually change the work."""
    assert len(I.build_tta(n, 16, 16)) == expected


@pytest.mark.parametrize("n", [1, 3, 5])
def test_tta_weights_sum_to_one(n):
    """A weighted average whose weights do not sum to 1 rescales every logit."""
    assert abs(sum(w for *_, w in I.build_tta(n, 16, 16)) - 1.0) < 1e-9


def test_tta_original_view_is_first_and_unweighted_by_augmentation():
    """The un-augmented view is the only one from the true test distribution."""
    tta = I.build_tta(5, 16, 16)
    fn, needs_swap, weight = tta[0]
    x = torch.rand(1, 3, 16, 16)
    assert torch.equal(fn(x), x), "the first TTA view must be the identity"
    assert needs_swap is False
    assert weight == pytest.approx(0.5)


def test_tta_predict_applies_the_laterality_permutation():
    """A mirrored knee's 'medial' output describes the original's LATERAL side.

    The model here returns a constant vector, so any difference in the output
    can only come from the permutation being applied (or not).
    """
    n_cls = 4
    const = torch.tensor([[10.0, -10.0, 0.0, 0.0]])

    class Const(nn.Module):
        def forward(self, x):
            return const.repeat(x.shape[0], 1)

    perm = [1, 0, 2, 3]                      # swap columns 0 and 1
    tta = I.build_tta(2, 8, 8)               # identity + mirror
    out = I.tta_predict(Const(), torch.rand(1, 3, 8, 8), False, torch.float32,
                        tta, lateral_swap_perm=perm)
    p = torch.sigmoid(const)
    # identity contributes p, the mirror contributes p[perm]; weights 0.5/0.5
    expected = 0.5 * p + 0.5 * p[:, perm]
    assert torch.allclose(out, expected, atol=1e-6)


# ══════════════════════════════════════════════════════════════════════════
# Ensembling
# ══════════════════════════════════════════════════════════════════════════

def test_rank_normalise_is_monotone_invariant():
    """ROC-AUC only sees ranking, so a monotone rescale must be a no-op."""
    a = np.random.default_rng(0).normal(size=(50, 4))
    assert np.allclose(I.rank_normalise(a), I.rank_normalise(a * 3.0 + 7.0))


def test_rank_average_of_one_model_is_just_its_ranks():
    a = np.random.default_rng(1).normal(size=(20, 3))
    assert np.allclose(I.rank_average([a]), I.rank_normalise(a))


def test_agreement_accepts_equivalent_and_rejects_different():
    """The soup ships only if it would RANK the test set like the ensemble."""
    rng = np.random.default_rng(2)
    a = rng.normal(size=(200, 6))
    assert I.agreement(a, a) == pytest.approx(1.0)
    assert I.agreement(a, a * 3 + 7) == pytest.approx(1.0)      # AUC-equivalent
    assert I.agreement(a, a + rng.normal(size=a.shape) * 0.01) > 0.98
    assert I.agreement(a, rng.normal(size=a.shape)) < 0.5       # unrelated


def test_agreement_ignores_constant_columns():
    """A column with no variance has no ranking to compare; it must not NaN."""
    a = np.random.default_rng(3).normal(size=(30, 3))
    b = a.copy()
    a[:, 1] = 1.0
    b[:, 1] = 1.0
    assert np.isfinite(I.agreement(a, b))


# ══════════════════════════════════════════════════════════════════════════
# Submission I/O
# ══════════════════════════════════════════════════════════════════════════

def _template(tmp, ids, cols, id_col="StudyInstanceUID"):
    d = os.path.join(tmp, "data")
    os.makedirs(d, exist_ok=True)
    sub = pd.DataFrame({id_col: ids})
    for c in cols:
        sub[c] = 0.5
    sub.to_csv(os.path.join(d, "sample_submission.csv"), index=False)
    return d


def test_submission_shape_follows_template():
    """Regression: columns were rebuilt from a constant, not read from the file.

    The template deliberately uses a different column ORDER and a different id
    column name than the built-in KNEE_TARGETS list.
    """
    with tempfile.TemporaryDirectory() as tmp:
        cols = ["Fracture", "ACL", "Effusion"]
        d = _template(tmp, ["s1", "s2"], cols, id_col="StudyUID")
        cfg = type("C", (), {"data_dir": d})()
        ids, targets, id_col = I.load_submission_template(cfg)
        assert ids == ["s1", "s2"]
        assert targets == cols, "column ORDER must come from the template"
        assert id_col == "StudyUID"


def test_write_submission_is_atomic_and_leaves_no_temp():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "submission.csv")
        cols = ["a", "b"]
        I.write_submission(out, ["x", "y"], cols, "sid", np.zeros((2, 2), np.float32))
        assert os.path.exists(out)
        assert not os.path.exists(out + ".tmp"), "temp file left behind"
        got = pd.read_csv(out)
        assert list(got.columns) == ["sid"] + cols
        assert got["sid"].astype(str).tolist() == ["x", "y"]


def test_write_submission_overwrites_in_place():
    """Every model rewrites the file; a partial run must leave a valid CSV."""
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "submission.csv")
        cols = ["a"]
        I.write_submission(out, ["x"], cols, "sid", np.array([[0.1]], np.float32))
        I.write_submission(out, ["x"], cols, "sid", np.array([[0.9]], np.float32))
        assert pd.read_csv(out)["a"].iloc[0] == pytest.approx(0.9)


def test_missing_template_falls_back_to_test_csv():
    with tempfile.TemporaryDirectory() as tmp:
        d = os.path.join(tmp, "data")
        os.makedirs(d)
        pd.DataFrame({"StudyInstanceUID": ["a", "b"]}).to_csv(
            os.path.join(d, "test.csv"), index=False)
        cfg = type("C", (), {"data_dir": d})()
        ids, targets, id_col = I.load_submission_template(cfg)
        assert ids == ["a", "b"]
        assert len(targets) == 12, "fallback must still produce all 12 targets"


# ══════════════════════════════════════════════════════════════════════════
# AMP
# ══════════════════════════════════════════════════════════════════════════

def test_amp_dtype_matches_hardware():
    """Regression: bfloat16 was hard-coded, but it needs sm_80+ and Kaggle is T4."""
    use_amp, dtype = I.resolve_amp(torch.device("cpu"))
    assert use_amp is False and dtype is torch.float32
    if torch.cuda.is_available():
        use_amp, dtype = I.resolve_amp(torch.device("cuda"))
        major = torch.cuda.get_device_capability(0)[0]
        assert dtype is (torch.bfloat16 if major >= 8 else torch.float16)


# ══════════════════════════════════════════════════════════════════════════
# Soup
# ══════════════════════════════════════════════════════════════════════════

class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)
        self.bn = nn.BatchNorm1d(3)

    def forward(self, x):
        return self.bn(self.fc(x))


def _save_variants(tmp, n, scale=0.002):
    paths, states = [], []
    base = _Net().state_dict()
    for i in range(n):
        st = {k: (v + torch.randn_like(v) * scale if torch.is_floating_point(v) else v)
              for k, v in base.items()}
        p = os.path.join(tmp, f"fold_{i}.pt")
        torch.save(st, p)
        paths.append(p)
        states.append(st)
    return paths, states


def test_soup_equals_the_uniform_mean():
    with tempfile.TemporaryDirectory() as tmp:
        paths, states = _save_variants(tmp, 5)
        soup, n = S.average_states(paths, verbose=False)
        assert n == 5
        for k, v in states[0].items():
            if torch.is_floating_point(v):
                want = torch.stack([s[k].float() for s in states]).mean(0)
                assert torch.allclose(soup[k], want, atol=1e-6), k


def test_soup_takes_the_latest_integer_buffer_not_the_mean():
    """num_batches_tracked is a counter; averaging it is meaningless."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for i in range(3):
            net = _Net()
            net.bn.num_batches_tracked += (i + 1) * 10
            p = os.path.join(tmp, f"f{i}.pt")
            torch.save(net.state_dict(), p)
            paths.append(p)
        soup, _ = S.average_states(paths, verbose=False)
        assert soup["bn.num_batches_tracked"].dtype == torch.int64
        assert int(soup["bn.num_batches_tracked"]) == 30


def test_soup_refuses_mismatched_architectures():
    """Averaging two different nets silently produces a plausible-looking file."""
    with tempfile.TemporaryDirectory() as tmp:
        paths, _ = _save_variants(tmp, 1)
        bad = os.path.join(tmp, "bad.pt")
        torch.save(nn.Linear(4, 3).state_dict(), bad)
        with pytest.raises(ValueError):
            S.average_states([paths[0], bad], verbose=False)


def test_divergence_report_separates_near_from_far():
    """The guard exists to reject folds that left the shared basin."""
    with tempfile.TemporaryDirectory() as tmp:
        near, _ = _save_variants(tmp, 3, scale=0.001)
        assert max(S.divergence_report(near)) < 0.35
    with tempfile.TemporaryDirectory() as tmp:
        far = []
        for i in range(3):
            p = os.path.join(tmp, f"g{i}.pt")
            torch.save(_Net().state_dict(), p)      # independent inits
            far.append(p)
        assert max(S.divergence_report(far)) > 0.35


def test_soup_streams_and_does_not_hold_every_checkpoint():
    """Peak RAM must be ~2 state dicts, not N, or a 5-fold soup OOMs on Kaggle."""
    import tracemalloc
    with tempfile.TemporaryDirectory() as tmp:
        paths, _ = _save_variants(tmp, 2)
        tracemalloc.start()
        S.average_states(paths, verbose=False)
        _, peak2 = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        paths8, _ = _save_variants(tmp, 8)
        tracemalloc.start()
        S.average_states(paths8, verbose=False)
        _, peak8 = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    # 4x the checkpoints must not cost anything like 4x the peak.
    assert peak8 < peak2 * 2.5, f"peak grew {peak8 / peak2:.1f}x for 4x the folds"



# ══════════════════════════════════════════════════════════════════════════
# Out-of-fold reporting
# ══════════════════════════════════════════════════════════════════════════

def _write_oof(d, fold, y, logits, cols):
    np.savez(os.path.join(d, f"oof_fold_{fold}.npz"),
             ids=np.arange(fold * len(y), (fold + 1) * len(y)).astype(object),
             logits=np.asarray(logits, np.float32),
             targets=np.asarray(y, np.float32),
             target_cols=np.asarray(cols, dtype=object))


def test_oof_report_pools_across_folds():
    """Pooling must score every row once, not fold-by-fold."""
    with tempfile.TemporaryDirectory() as d:
        rng = np.random.default_rng(0)
        for f in range(4):
            y = np.zeros((10, 1), np.float32)
            y[:5, 0] = 1
            lg = (y[:, :1] * 2 + rng.normal(0, 0.4, (10, 1)))
            _write_oof(d, f, y, lg, ["A"])
        r = S.oof_report(d, verbose=False)
        assert r["n_folds"] == 4
        assert r["n_rows"] == 40, "every held-out row must appear exactly once"
        assert 0.5 < r["pooled_macro_auc"] <= 1.0


def test_pooling_makes_a_fold_sparse_label_computable():
    """The real defect in averaging per-fold AUCs.

    A label whose positives land in only one fold is UNCOMPUTABLE in the others
    (roc_auc_score needs both classes), so it silently drops out of those folds'
    macro averages: the per-fold mean is then an average over a DIFFERENT LABEL
    SET per fold, which is not comparable across configurations.

    Note the per-fold mean is not reliably higher or lower — with 10-13 rows a
    single fold's noise can move it either way, which is itself the argument for
    not comparing configurations on it.
    """
    with tempfile.TemporaryDirectory() as d:
        rng = np.random.default_rng(1)
        for f in range(5):
            y = np.zeros((12, 2), np.float32)
            y[:6, 0] = 1                      # computable in every fold
            if f == 0:
                y[:3, 1] = 1                  # positives in fold 0 only
            lg = np.stack([y[:, 0] * 2 + rng.normal(0, 0.5, 12),
                           y[:, 1] * 2 + rng.normal(0, 0.5, 12)], 1)
            _write_oof(d, f, y, lg, ["Common", "Rare"])

        # Per fold, 'Rare' is uncomputable in 4 of the 5.
        uncomputable = 0
        for f in range(5):
            z = np.load(os.path.join(d, f"oof_fold_{f}.npz"), allow_pickle=True)
            if len(np.unique(z["targets"][:, 1])) < 2:
                uncomputable += 1
        assert uncomputable == 4, "fixture should make 'Rare' fold-sparse"

        r = S.oof_report(d, verbose=False)
        auc, npos, n = r["per_label"]["Rare"]
        assert np.isfinite(auc), "pooling must make a fold-sparse label computable"
        assert (npos, n) == (3, 60), "scored on every pooled row, once"
        assert r["n_rows"] == 60


def test_oof_report_handles_missing_directory():
    with tempfile.TemporaryDirectory() as d:
        assert S.oof_report(d, verbose=False) is None
        assert S.clean_oof_auc(d) == (None, 0)


def test_clean_oof_auc_now_returns_the_pooled_number():
    with tempfile.TemporaryDirectory() as d:
        rng = np.random.default_rng(2)
        for f in range(3):
            y = np.zeros((10, 1), np.float32)
            y[:5, 0] = 1
            _write_oof(d, f, y, y[:, :1] * 2 + rng.normal(0, 0.4, (10, 1)), ["A"])
        pooled, k = S.clean_oof_auc(d)
        assert k == 3
        assert pooled == pytest.approx(S.oof_report(d, verbose=False)["pooled_macro_auc"])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
